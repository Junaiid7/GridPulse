"""Offline tests for the Phase 4A pipeline runner.

These tests drive the full Bronze → Silver → Gold → Features → DQ-report
path with stored synthetic fixtures and *fake clients* — no live network, no
API keys. They assert the real pipeline behaviour (paths, availability flags,
Gold/feature content) so the live run can be trusted when it mirrors the same
code path.

Verification level ladder (documented in the DQ report):
- offline fixture run  -> "verified via stored fixture"
- live run             -> "verified via live API"
- nothing available    -> "unavailable / assumed / documented limitation"
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest import mock

from gridpulse.config import Settings
from gridpulse.ingestion.common.models import FetchResult
from gridpulse.ingestion.weather.variables import NL_POINTS
from gridpulse.pipeline.runner import DatasetResult, PipelineRun, run_pipeline

UTC = UTC

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"

_DATA = {
    "load": (FIXTURES / "entsoe_actual_load.xml").read_bytes(),
    "generation": (FIXTURES / "entsoe_generation.xml").read_bytes(),
    "prices": (FIXTURES / "entsoe_dayahead_prices.xml").read_bytes(),
    "imbalance": (FIXTURES / "entsoe_imbalance_prices.xml").read_bytes(),
    "weather": (FIXTURES / "open_meteo_hourly.json").read_bytes(),
}

START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 2, tzinfo=UTC)


class FixtureEntsoeClient:
    """Fake ENTSO-E client serving canned FetchResults."""

    _ENTITY_KEY = {
        "actual-total-load": "load",
        "actual-generation-by-type": "generation",
        "dayahead-prices": "prices",
        "imbalance-prices": "imbalance",
    }

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self._payloads = payloads

    def _result(self, entity: str) -> FetchResult:
        key = self._ENTITY_KEY.get(entity, entity)
        return FetchResult(
            source="entsoe",
            entity=entity,
            start=START,
            end=END,
            retrieved_at=datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
            payload=self._payloads[key],
            content_type="application/xml",
            encoding="utf-8",
            url="https://web-api.tp.entsoe.eu/api",
            timezone="UTC",
        )

    def fetch_load(self, area, start, end) -> list[FetchResult]:
        return [self._result("actual-total-load")]

    def fetch_generation(self, area, start, end) -> list[FetchResult]:
        return [self._result("actual-generation-by-type")]

    def fetch_day_ahead_prices(self, area, start, end) -> list[FetchResult]:
        return [self._result("dayahead-prices")]

    def fetch_imbalance_prices(self, area, start, end) -> list[FetchResult]:
        return [self._result("imbalance-prices")]


class FixtureOpenMeteoClient:
    """Fake Open-Meteo client: same payload for every NL location."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def fetch_historical(self, location, start_date, end_date) -> list[FetchResult]:
        return [
            FetchResult(
                source="open-meteo",
                entity="historical-weather",
                start=datetime.combine(start_date, datetime.min.time(), tzinfo=UTC),
                end=datetime.combine(end_date, datetime.min.time(), tzinfo=UTC)
                + timedelta(days=1),
                retrieved_at=datetime(2024, 1, 2, 12, 0, tzinfo=UTC),
                payload=self._payload,
                content_type="application/json",
                encoding="utf-8",
                url="https://archive-api.open-meteo.com/v1/archive",
                identifiers=(f"location:{location.name}",),
                timezone="UTC",
                metadata={"location": location.name, "variables": "hourly"},
            )
        ]


def _settings(tmp_path: Path) -> Settings:
    return Settings(project_root=tmp_path, data_root=tmp_path)


def _run(tmp_path: Path, *, entsoe: bool = True) -> PipelineRun:
    om = FixtureOpenMeteoClient(_DATA["weather"])
    en = FixtureEntsoeClient(_DATA) if entsoe else None
    return run_pipeline(
        start=START,
        end=END,
        settings=_settings(tmp_path),
        open_meteo_client=om,
        entsoe_client=en,
        report_dir=tmp_path / "reports",
    )


def test_full_fixture_run_produces_all_tiers(tmp_path: Path) -> None:
    run = _run(tmp_path, entsoe=True)

    # Weather verified live (fixture) for every location-variable.
    wx = [d for d in run.datasets if d.source == "open-meteo"]
    assert wx
    assert all(d.status == "verified_live" for d in wx)
    assert len(run.weather_locations) == len(NL_POINTS)
    assert "temperature_2m" in run.weather_variables

    # ENTSO-E datasets available via fixture.
    en = [d for d in run.datasets if d.source == "entsoe"]
    assert run.entsoe_available
    assert any(d.entity == "actual-generation-by-type:B18" for d in en)

    # Imbalance path verified through the fixture.
    assert run.imbalance_note.startswith("VERIFIED")

    # Bronze / Silver artefacts exist on disk.
    assert any(p for p in run.datasets if p.bronze_dir and Path(p.bronze_dir).exists())
    assert any(p.silver_csv for p in run.datasets if Path(p.silver_csv).exists())

    # Gold produced with the expected union hour grid (00,01,02,03).
    assert run.gold.produced, run.gold.reason
    assert run.gold.row_count == 4
    assert run.gold.csv_path and Path(run.gold.csv_path).exists()
    assert run.gold.negative_residual_fraction == 0.0
    assert run.gold.residual_stats["min"] is not None

    # Features produced with as-of-guard columns.
    assert run.features.produced
    assert run.features.csv_path and Path(run.features.csv_path).exists()
    assert "lag_1h" in run.features.columns
    assert "weather_temperature_2m" in run.features.columns

    # DQ report written (JSON + Markdown).
    assert len(run.report_paths) == 2
    assert all(p.exists() for p in run.report_paths)
    assert run.ok  # no recorded errors


def test_full_fixture_run_feature_content(tmp_path: Path) -> None:
    """Golden content checks: calendar, lags, rolling, no leakage."""
    run = _run(tmp_path, entsoe=True)
    from gridpulse.transformation.csvio import read_table

    rows = read_table(Path(run.features.csv_path))
    assert len(rows) == 4
    first = rows[0]
    # First gold hour has no strictly-before history -> lags/rolling are empty.
    assert first["lag_1h"] == ""
    assert first["rolling_mean_24h"] == ""
    # Second gold hour's lag_1h equals the first residual (strictly before).
    second = rows[1]
    assert float(second["lag_1h"]) == float(first["residual_load_mw"])

    # No future observation leaks into weather features of the first hour.
    assert first["weather_temperature_2m"] == ""
    # 01:00 uses the 00:00 observation (strictly before).
    assert float(second["weather_temperature_2m"]) == 3.1

    # Calendar fields exist and are boolean-correct for a Monday (2024-01-01).
    assert first["is_weekend"] == "false"
    assert first["is_holiday"] == "true"  # New Year's Day


def test_gold_residual_matches_definition(tmp_path: Path) -> None:
    """residual = load − wind(B18+B19) − solar(B16), not clamped."""
    run = _run(tmp_path, entsoe=True)
    from gridpulse.transformation.csvio import read_table

    gold_rows = read_table(Path(run.gold.csv_path))
    hour00 = next(
        r for r in gold_rows if r["timestamp_utc"].startswith("2024-01-01T00:00")
    )
    # load = mean(4510,4561,4612,4689); wind = 1000+500; solar = 0
    expected = (4510 + 4561 + 4612 + 4689) / 4 - 1500.0
    assert abs(float(hour00["residual_load_mw"]) - expected) < 1e-9


def test_no_entsoe_run_still_writes_weather_and_report(tmp_path: Path) -> None:
    """Without ENTSO-E, datasets are UNAVAILABLE but weather + report exist."""
    run = _run(tmp_path, entsoe=False)

    assert not run.entsoe_available
    en = [d for d in run.datasets if d.source == "entsoe"]
    assert len(en) == 6  # imbalance + load + gen×3 + prices
    assert all(d.status == "unavailable" for d in en)
    core_notes = [d.note for d in en if d.entity != "imbalance-prices"]
    assert all(n == "ENTSOE_API_KEY not configured" for n in core_notes)

    # Imbalance explicitly UNVERIFIED, never fabricated.
    assert (
        run.imbalance_note
        == "UNVERIFIED — cannot construct EntsoeClient: ENTSOE_API_KEY not configured"
    )

    # Weather still ingested and aggregated.
    assert any(d.status == "verified_live" for d in run.datasets)

    # Gold / Features not produced, reasons honest.
    assert not run.gold.produced
    assert "ENTSOE" in run.gold.reason.upper()
    assert not run.features.produced

    # Report present despite partial data.
    assert len(run.report_paths) == 2
    report = json.loads(run.report_paths[0].read_text(encoding="utf-8"))
    assert report["summary"]["entsoe_datasets_unavailable"] == 6
    assert report["summary"]["gold_produced"] is False
    assert report["summary"]["imbalance_status"].startswith("UNVERIFIED")


def _fake_run(tmp_path: Path, *, entsoe_verified: bool) -> PipelineRun:
    datasets = []
    if entsoe_verified:
        datasets.append(
            DatasetResult(
                source="entsoe", entity="actual-total-load", status="verified_live"
            )
        )
    else:
        datasets.append(
            DatasetResult(
                source="entsoe", entity="actual-total-load", status="unavailable"
            )
        )
    return PipelineRun(
        settings=_settings(tmp_path),
        requested_start=START,
        requested_end=END,
        datasets=datasets,
        imbalance_note="UNVERIFIED — test",
        report_paths=[Path("dummy.json"), Path("dummy.md")],
    )


def test_cli_exit_codes_are_offline(tmp_path: Path) -> None:
    """CLI exit codes routed by patched runner — no live network in tests."""
    from gridpulse.pipeline.__main__ import main

    # Default (no ENTSO-E configured) + --require-entsoe -> exit 2
    with mock.patch(
        "gridpulse.pipeline.runner.run_pipeline",
        return_value=_fake_run(tmp_path, entsoe_verified=False),
    ):
        assert (
            main(["--start", "2024-01-01", "--end", "2024-01-02", "--require-entsoe"])
            == 2
        )

    # ENTSO-E present -> runs cleanly, exit 0
    with mock.patch(
        "gridpulse.pipeline.runner.run_pipeline",
        return_value=_fake_run(tmp_path, entsoe_verified=True),
    ):
        assert main(["--start", "2024-01-01", "--end", "2024-01-02"]) == 0

    # --require-gold unmet -> exit 3
    with mock.patch(
        "gridpulse.pipeline.runner.run_pipeline",
        return_value=_fake_run(tmp_path, entsoe_verified=True),
    ):
        assert (
            main(["--start", "2024-01-01", "--end", "2024-01-02", "--require-gold"])
            == 3
        )
