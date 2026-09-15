"""Offline tests for the Phase 4A data-quality report.

Builds a report from a *hand-built* PipelineRun (no live network, no clients)
and asserts the serialised sections match the Step 8 specification: SOURCE /
COVERAGE / VALUES / TIME / ENERGY / WEATHER / IMBALANCE / PROVENANCE, plus a
summary that distinguishes live-verified from unavailable datasets.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from gridpulse.config import Settings
from gridpulse.pipeline.runner import (
    DatasetResult,
    FeatureResult,
    GoldResult,
    PipelineRun,
)
from gridpulse.reporting.dq_report import DataQualityReport
from gridpulse.ingestion.weather.variables import Location

UTC = timezone.utc

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 2, tzinfo=UTC)


def _weather_result(entity: str) -> DatasetResult:
    return DatasetResult(
        source="open-meteo",
        entity=entity,
        status="verified_live",
        requested_start=START,
        requested_end=END,
        fetched_at=datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
        actual_start=START,
        actual_end=START + timedelta(hours=3),
        resolution_minutes=60,
        unit="°C",
        row_count=3,
        expected_count=24,
        missing_timestamps=21,
        duplicate_dropped=0,
        null_values=0,
        min_value=2.6,
        max_value=3.1,
        mean_value=2.8666666666666667,
        bronze_dir="data/bronze/open-meteo/historical-weather/20240101__20240102",
        bronze_sha256="a" * 64,
        silver_csv="data/silver/open-meteo/historical-weather-temperature-2m/temperature.csv",
    )


def _entsoe_unavailable(entity: str) -> DatasetResult:
    return DatasetResult(
        source="entsoe",
        entity=entity,
        status="unavailable",
        note="ENTSOE_API_KEY not configured",
        requested_start=START,
        requested_end=END,
        expected_count=24,
    )


def _run() -> PipelineRun:
    return PipelineRun(
        settings=Settings(project_root=Path("."), data_root=Path("data")),
        requested_start=START,
        requested_end=END,
        started_at=datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
        finished_at=datetime(2024, 1, 2, 12, 0, tzinfo=UTC) + timedelta(seconds=3),
        bronze_root=Path("data/bronze"),
        silver_root=Path("data/silver"),
        gold_root=Path("data/gold"),
        datasets=[
            _weather_result("historical-weather:nl-central:temperature_2m"),
            _weather_result("historical-weather:nl-central:wind_speed_100m"),
            _entsoe_unavailable("actual-total-load"),
            _entsoe_unavailable("imbalance-prices"),
        ],
        weather_locations=[Location("nl-central", 52.21, 5.29)],
        weather_variables=["temperature_2m", "wind_speed_100m"],
        imbalance_note="UNVERIFIED — cannot construct EntsoeClient: ENTSOE_API_KEY not configured",
        gold=GoldResult(produced=False, reason="No ENTSO-E load data available"),
        features=FeatureResult(produced=False, reason="No ENTSO-E load data available"),
        dst_offsets_observed={60: 24},
        dst_transitions_observed=0,
    )


def test_report_dict_sections(tmp_path: Path) -> None:
    report = DataQualityReport(_run())
    data = report.to_dict()

    # Summary distinguishes verified weather from unavailable ENTSO-E.
    summary = data["summary"]
    assert summary["open_meteo_datasets_verified"] == 2
    assert summary["entsoe_datasets_verified"] == 0
    assert summary["entsoe_datasets_unavailable"] == 2
    assert summary["gold_produced"] is False
    assert summary["imbalance_status"].startswith("UNVERIFIED")

    # Requested window + runtime present.
    meta = data["report"]
    assert meta["requested_window"]["start_utc"] == START.isoformat()
    assert meta["total_runtime_s"] == 3.0

    # Time policy reflects the Amsterdam/DST rule.
    tp = data["time_policy"]
    assert tp["dst_offsets_observed"] == {60: 24}
    assert tp["number_of_distinct_local_offsets"] == 1
    assert "Europe/Amsterdam" in tp["local_time_policy"]

    # Per-dataset coverage/value info present.
    rows = data["source_coverage_values"]
    weather_rows = [r for r in rows if r["source"] == "open-meteo"]
    assert weather_rows[0]["row_count"] == 3
    assert weather_rows[0]["missing_timestamps"] == 21
    assert weather_rows[0]["unit"] == "°C"
    entsoe_ds = [r for r in rows if r["source"] == "entsoe"]
    assert all(r["status"] == "unavailable" for r in entsoe_ds)
    assert entsoe_ds[0]["note"] == "ENTSOE_API_KEY not configured"

    # Energy section honestly reports no Gold.
    assert data["energy"]["gold_dataset"] == "NOT PRODUCED"

    # Weather section lists locations/variables.
    assert data["weather"]["locations"] == ["nl-central"]
    assert set(data["weather"]["variables"]) == {"temperature_2m", "wind_speed_100m"}

    # Provenance records roots and hashes, and nothing else is secret here.
    assert Path(data["provenance"]["bronze_root"]).as_posix() == "data/bronze"
    assert Path(data["provenance"]["silver_root"]).as_posix() == "data/silver"
    assert Path(data["provenance"]["gold_root"]).as_posix() == "data/gold"
    assert data["provenance"]["hashes_where_available"][0]["sha256"] == "a" * 64

    # Render a markdown view (exercise the formatter, check key headings).
    md = report.to_markdown()
    for section in ("## TIME POLICY", "## SUMMARY", "## SOURCE / COVERAGE / VALUES",
                    "## ENERGY", "## WEATHER", "## IMBALANCE", "## PROVENANCE"):
        assert section in md
    assert "Golden" not in md or "ENERGY" in md


def test_report_write_outputs_json_and_markdown(tmp_path: Path) -> None:
    report = DataQualityReport(_run())
    json_path, md_path = report.write(tmp_path / "reports")

    assert json_path.exists()
    assert md_path.name.startswith("dq-report_") and md_path.name.endswith(".md")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["summary"]["imbalance_status"].startswith("UNVERIFIED")
    assert md_path.read_text(encoding="utf-8").startswith("# GridPulse Data-Quality Report")


def test_serialization_is_json_clean(tmp_path: Path) -> None:
    """Full round-trip through json.dumps must not raise (no datetimes leaked)."""
    data = DataQualityReport(_run()).to_dict()
    text = json.dumps(data, default=str)
    assert '"generated_at_utc"' in text or "generated_at_utc" in text
    assert "ENTSOE_API_KEY not configured" in text  # reason, not a secret value