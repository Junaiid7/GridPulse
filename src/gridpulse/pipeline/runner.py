"""Pipeline runner: Bronze → Silver → Gold → Features → Data-quality report.

This module is the central orchestrator for a Phase 4A real-data run.

Design constraints (do not break):
- Uses existing Phase 2 clients (OpenMeteoClient, EntsoeClient) and
  existing Phase 3 transformations; adds nothing to them.
- Injects clients for tests (fake opener); no live network in unit tests.
- ENTSO-E steps are soft-fail: when no key is configured the run records
  each dataset as UNAVAILABLE and continues with weather artefacts.
- Reports are always written (even on partial data) for auditability.
- Secrets are never written to reports or metadata.
"""

from __future__ import annotations

import logging
import platform
import statistics
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .. import __version__
from ..config import ConfigurationError, Settings, get_settings
from ..ingestion.common.errors import (
    AuthError,
    EmptyResponseError,
    HttpError,
    IngestionError,
    NoDataError,
)
from ..ingestion.common.models import FetchResult, TimeSeries, ensure_utc
from ..ingestion.common.storage import write_bronze
from ..ingestion.common.validation import (
    Issue,
    ValidationReport,
    aggregate,
    validate_non_negative,
    validate_timestamps,
)
from ..ingestion.entsoe.client import EntsoeClient
from ..ingestion.entsoe.domains import NL
from ..ingestion.entsoe.imbalance import EntsoeImbalancePrices, ImbalancePricesSource
from ..ingestion.entsoe.parser import (
    parse_day_ahead_prices,
    parse_generation,
    parse_load,
)
from ..features.calendar import calendar_features
from ..features.rolling import rolling_mean, rolling_std
from ..features.weather import weather_features
from ..ingestion.weather.open_meteo import OpenMeteoClient
from ..ingestion.weather.parser import parse_historical_json
from ..ingestion.weather.variables import NL_POINTS, NON_NEGATIVE_CONTRACT, Location
from ..ingestion.wiring import get_entsoe_client, get_open_meteo_client
from ..reporting.dq_report import DataQualityReport
from ..transformation.aggregation import aggregate_locations, to_hourly
from ..transformation.csvio import read_table, write_table
from ..transformation.gold import GoldInputs, build_gold_hourly, write_gold
from ..transformation.gold.residual_load import SOLAR_PSR, WIND_PSR
from ..transformation.provenance import Provenance
from ..transformation.silver import clean_timeseries, write_silver
from ..transformation.times import AMSTERDAM_TZ, utc_offset_minutes

LOGGER = logging.getLogger("gridpulse.pipeline")


# ── Data structures ────────────────────────────────────────────────────


@dataclass
class DatasetResult:
    """Outcome of one attempted dataset fetch/transform."""

    source: str
    entity: str
    status: str = "pending"
    note: str = ""
    requested_start: datetime | None = None
    requested_end: datetime | None = None
    fetched_at: datetime | None = None
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    resolution_minutes: int | None = None
    unit: str | None = None
    row_count: int | None = None
    expected_count: int | None = None
    missing_timestamps: int | None = None
    duplicate_dropped: int | None = None
    null_values: int | None = None
    min_value: float | None = None
    max_value: float | None = None
    mean_value: float | None = None
    bronze_dir: str | None = None
    bronze_sha256: str | None = None
    silver_csv: str | None = None
    local_note: str = ""


@dataclass
class GoldResult:
    produced: bool = False
    reason: str = ""
    csv_path: str | None = None
    row_count: int = 0
    actual_start: datetime | None = None
    actual_end: datetime | None = None
    missing_counts: dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    residual_stats: dict[str, float | None] = field(default_factory=dict)
    negative_residual_fraction: float | None = None
    residual_definition: str = "load − wind (B18+B19) − solar (B16)"
    wind_psr: tuple[str, ...] = WIND_PSR
    solar_psr: tuple[str, ...] = SOLAR_PSR


@dataclass
class FeatureResult:
    produced: bool = False
    reason: str = ""
    csv_path: str | None = None
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    asof_policy: str = (
        "as_of = target_utc (start of the target hour). All lag/rolling/weather "
        "features use observations strictly before target; calendar features "
        "are calendar-only. No future information used. Limitation: weather "
        "variables are historical observations, not NWP forecasts."
    )


@dataclass
class PipelineRun:
    settings: Settings
    requested_start: datetime
    requested_end: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    gridpulse_version: str = __version__
    python_version: str = platform.python_version()
    bronze_root: Path = Path()
    silver_root: Path = Path()
    gold_root: Path = Path()
    datasets: list[DatasetResult] = field(default_factory=list)
    weather_locations: list[Location] = field(default_factory=list)
    weather_variables: list[str] = field(default_factory=list)
    weather_nl_hourly: dict[str, TimeSeries] = field(default_factory=dict)
    imbalance_note: str = "unknown"
    imbalance_evidence: str = ""
    gold: GoldResult = field(default_factory=GoldResult)
    features: FeatureResult = field(default_factory=FeatureResult)
    errors: list[str] = field(default_factory=list)
    report_paths: list[Path] = field(default_factory=list)
    dst_offsets_observed: dict[int, int] = field(default_factory=dict)
    dst_transitions_observed: int = 0

    @property
    def entsoe_available(self) -> bool:
        return any(d.status == "verified_live" for d in self.datasets if d.source == "entsoe")

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0 and self.report_paths


# ── Helpers ────────────────────────────────────────────────────────────


def _expected_hourly_count(start: datetime, end: datetime) -> int:
    return max(0, int((ensure_utc(end) - ensure_utc(start)).total_seconds() / 3600))


def _count_issues(report: ValidationReport, code: str) -> int:
    return sum(1 for i in report.issues if i.code == code)


def _series_stats(series: TimeSeries) -> tuple[float | None, float | None, float | None]:
    vals = [p.value for p in series.points]
    if not vals:
        return None, None, None
    return min(vals), max(vals), sum(vals) / len(vals)


def _silver_stats(records) -> tuple[float | None, float | None, float | None]:
    vals = [r.value for r in records]
    if not vals:
        return None, None, None
    return min(vals), max(vals), sum(vals) / len(vals)


def _value_at(history: list[tuple[datetime, float]], ts: datetime) -> float | None:
    """Return value exactly at ``ts`` from a sorted history list, or None."""
    for t, v in history:
        if t == ts:
            return v
    return None


def _compute_dst(transitions: list[tuple[datetime, int]]) -> tuple[dict[int, int], int]:
    """Given sorted UTC timestamps with their Amsterdam UTC-offset, count transitions."""
    if not transitions:
        return {}, 0
    offsets = {}
    n_transitions = 0
    prev_offset = transitions[0][1]
    for _, off in transitions:
        offsets[off] = offsets.get(off, 0) + 1
        if off != prev_offset:
            n_transitions += 1
            prev_offset = off
    return offsets, n_transitions


# ── Weather step ───────────────────────────────────────────────────────


def _run_weather(
    client: OpenMeteoClient,
    start: datetime,
    end: datetime,
    bronze_root: Path,
    silver_root: Path,
    logger: logging.Logger,
) -> tuple[list[DatasetResult], dict[tuple[str, str], TimeSeries], dict[str, TimeSeries], list[tuple[datetime, int]]]:
    """Fetch, bronze, parse, silver for every NL location × variable.

    Returns ``(results, by_loc_var, weather_nl_hourly, dst_offsets_raw)``.
    """
    results: list[DatasetResult] = []
    by_loc_var: dict[tuple[str, str], TimeSeries] = {}
    start_date = start.date()
    end_date = (ensure_utc(end) - timedelta(days=1)).date()  # inclusive date for Open-Meteo
    dst_transitions: list[tuple[datetime, int]] = []

    for loc in NL_POINTS:
        entity_base = f"historical-weather:{loc.name}"
        try:
            fetches: list[FetchResult] = client.fetch_historical(loc, start_date, end_date)
        except Exception as exc:
            results.append(
                DatasetResult(
                    source="open-meteo",
                    entity=entity_base,
                    status="failed",
                    note=f"fetch failed: {exc!r}",
                    requested_start=start,
                    requested_end=end,
                )
            )
            logger.warning("Open-Meteo fetch failed for %s: %r", loc.name, exc)
            continue

        for fr in fetches:
            write_bronze(bronze_root, fr)

        # Parse all variables from each fetch payload
        for fr in fetches:
            try:
                parsed = parse_historical_json(fr.payload)
            except Exception as exc:
                results.append(
                    DatasetResult(
                        source="open-meteo",
                        entity=entity_base,
                        status="failed",
                        note=f"parse failed: {exc!r}",
                        fetched_at=fr.retrieved_at,
                    )
                )
                continue

            for var, series in parsed.items():
                entity = f"historical-weather:{loc.name}:{var}"
                vreport = validate_timestamps(series)
                if var in NON_NEGATIVE_CONTRACT:
                    vreport = aggregate([vreport, validate_non_negative(series, reason="physics contract")])

                records, creport = clean_timeseries(series)
                silver_csv, _ = write_silver(
                    silver_root, records, series.source, entity, series.unit,
                    report=creport, generated_at=fr.retrieved_at,
                )

                mn, mx, avg = _series_stats(series)
                expected = _expected_hourly_count(start, end)
                results.append(
                    DatasetResult(
                        source="open-meteo",
                        entity=entity,
                        status="verified_live",
                        note="",
                        requested_start=start,
                        requested_end=end,
                        fetched_at=fr.retrieved_at,
                        actual_start=min(p.timestamp for p in series.points) if series.points else None,
                        actual_end=max(p.timestamp for p in series.points) if series.points else None,
                        resolution_minutes=series.resolution_minutes,
                        unit=series.unit,
                        row_count=len(records),
                        expected_count=expected,
                        missing_timestamps=_count_issues(vreport, "timestamp_gap"),
                        duplicate_dropped=_count_issues(creport, "duplicate_dropped"),
                        null_values=series.metadata.get("nulls"),
                        min_value=mn,
                        max_value=mx,
                        mean_value=avg,
                        bronze_dir=str(bronze_root / fr.source / fr.entity),
                        bronze_sha256=fr.sha256,
                        silver_csv=str(silver_csv),
                    )
                )

                by_loc_var[(loc.name, var)] = series

                # Collect DST offset info from timestamps
                for p in series.points:
                    off = utc_offset_minutes(p.timestamp)
                    dst_transitions.append((ensure_utc(p.timestamp), off))

    # Build NL aggregate hourly series per variable
    by_var: dict[str, dict[str, TimeSeries]] = {}
    for (loc_name, var), s in by_loc_var.items():
        by_var.setdefault(var, {})[loc_name] = s

    nl_hourly: dict[str, TimeSeries] = {}
    for var, locs in by_var.items():
        hourly_locs = {name: to_hourly(s) for name, s in locs.items()}
        nl_hourly[var] = aggregate_locations(hourly_locs)

    return results, by_loc_var, nl_hourly, dst_transitions


# ── Imbalance verification ─────────────────────────────────────────────


def _verify_imbalance(
    entsoe_client: EntsoeClient | None,
    start: datetime,
    end: datetime,
    imbalance_source: ImbalancePricesSource | None,
    bronze_root: Path,
    silver_root: Path,
    logger: logging.Logger,
) -> tuple[DatasetResult, str]:
    """Attempt NL imbalance-price verification per Step 3."""
    entity = "imbalance-prices"
    ds = DatasetResult(
        source="entsoe",
        entity=entity,
        status="unavailable",
        requested_start=start,
        requested_end=end,
    )

    if entsoe_client is None:
        ds.note = "UNVERIFIED — cannot construct EntsoeClient: ENTSOE_API_KEY not configured"
        return ds, ds.note

    if imbalance_source is None:
        imbalance_source = EntsoeImbalancePrices(entsoe_client)

    try:
        results: list[FetchResult] = imbalance_source.fetch(NL, start, end)
    except NoDataError as exc:
        ds.status = "unavailable"
        ds.note = f'UNAVAILABLE — ENTSO-E returned "No matching data found" for NL: {exc}'
        return ds, ds.note
    except (HttpError, AuthError, IngestionError) as exc:
        ds.status = "failed"
        ds.note = f"FAILED — {type(exc).__name__}: {exc}"
        return ds, ds.note
    except Exception as exc:
        ds.status = "failed"
        ds.note = f"FAILED — unexpected: {type(exc).__name__}: {exc}"
        return ds, ds.note

    if not results:
        ds.status = "unavailable"
        ds.note = "UNAVAILABLE — fetch returned empty result list"
        return ds, ds.note

    # Parse all payloads (may be ZIP)
    from ..ingestion.entsoe.parser import parse_imbalance_prices
    all_points = []
    for fr in results:
        write_bronze(bronze_root, fr)
        try:
            series = parse_imbalance_prices(fr.payload)
            all_points.extend(series.points)
        except Exception as exc:
            ds.status = "failed"
            ds.note = f"parse failed: {exc!r}"
            return ds, ds.note

    if not all_points:
        ds.status = "unavailable"
        ds.note = "VERIFIED reachable but no data points returned"
        return ds, ds.note

    ds.status = "verified_live"
    mn = min(p.value for p in all_points)
    mx = max(p.value for p in all_points)
    avg = sum(p.value for p in all_points) / len(all_points)
    ds.row_count = len(all_points)
    ds.min_value = mn
    ds.max_value = mx
    ds.mean_value = avg
    ds.unit = "EUR/MWh"
    ds.resolution_minutes = 60  # typical for A85
    ds.actual_start = min(p.timestamp for p in all_points)
    ds.actual_end = max(p.timestamp for p in all_points)
    ds.note = (
        f"VERIFIED — {len(all_points)} data points | "
        f"min={mn:.2f} max={mx:.2f} mean={avg:.2f} EUR/MWh | "
        f"window=[{ds.actual_start.isoformat()}, {ds.actual_end.isoformat()})"
    )
    return ds, ds.note


# ── ENTSO-E datasets step ──────────────────────────────────────────────


def _run_entsoe_dataset(
    entsoe_client: EntsoeClient,
    *,
    entity: str,
    fetcher,
    parser,
    psr_codes: Sequence[str] | None = None,
    start: datetime,
    end: datetime,
    bronze_root: Path,
    silver_root: Path,
) -> list[DatasetResult]:
    """Fetch, parse, bronze-write, and clean one ENTSO-E dataset (or per PSR)."""
    results: list[DatasetResult] = []
    expected = _expected_hourly_count(start, end)

    try:
        fetch_results = fetcher(entsoe_client, NL, start, end)
    except NoDataError as exc:
        results.append(
            DatasetResult(
                source="entsoe", entity=entity, status="unavailable",
                note=f"No matching data found for {entity}: {exc}",
                requested_start=start, requested_end=end,
                expected_count=expected,
            )
        )
        return results
    except (HttpError, AuthError, IngestionError) as exc:
        results.append(
            DatasetResult(
                source="entsoe", entity=entity, status="failed",
                note=f"{type(exc).__name__}: {exc}",
                requested_start=start, requested_end=end,
                expected_count=expected,
            )
        )
        return results
    except Exception as exc:
        results.append(
            DatasetResult(
                source="entsoe", entity=entity, status="failed",
                note=f"unexpected {type(exc).__name__}: {exc}",
                requested_start=start, requested_end=end,
                expected_count=expected,
            )
        )
        return results

    for fr in fetch_results:
        write_bronze(bronze_root, fr)
        try:
            if psr_codes is not None:
                psr_dict = parser(fr.payload)
                for code in psr_codes:
                    series = psr_dict.get(code)
                    if series is None:
                        results.append(
                            DatasetResult(
                                source="entsoe", entity=f"{entity}:{code}",
                                status="unavailable",
                                note=f"PSR code {code} not present in response",
                                requested_start=start, requested_end=end,
                                fetched_at=fr.retrieved_at,
                                expected_count=expected,
                            )
                        )
                        continue
                    series = to_hourly(series)
                    _write_silver_dataset(
                        series, fr, silver_root, f"{entity}:{code}", expected, start, end, results
                    )
            else:
                series = parser(fr.payload)
                series = to_hourly(series)
                _write_silver_dataset(
                    series, fr, silver_root, entity, expected, start, end, results
                )
        except Exception as exc:
            results.append(
                DatasetResult(
                    source="entsoe", entity=entity, status="failed",
                    note=f"parse/clean failed: {exc!r}",
                    requested_start=start, requested_end=end,
                    fetched_at=fr.retrieved_at,
                    expected_count=expected,
                )
            )

    return results


def _write_silver_dataset(
    series: TimeSeries,
    fr: FetchResult,
    silver_root: Path,
    entity: str,
    expected: int,
    start: datetime,
    end: datetime,
    results: list[DatasetResult],
) -> None:
    checks = [validate_timestamps(series)]
    if series.unit in ("MW", "EUR/MWh"):
        checks.append(validate_non_negative(series, reason="market contract"))
    vreport = aggregate(checks)
    records, creport = clean_timeseries(series)
    silver_csv, _ = write_silver(
        silver_root, records, series.source, entity, series.unit,
        report=creport, generated_at=fr.retrieved_at,
    )
    mn, mx, avg = _silver_stats(records)
    results.append(
        DatasetResult(
            source="entsoe",
            entity=entity,
            status="verified_live",
            requested_start=start,
            requested_end=end,
            fetched_at=fr.retrieved_at,
            actual_start=min(r.timestamp_utc for r in records) if records else None,
            actual_end=max(r.timestamp_utc for r in records) if records else None,
            resolution_minutes=series.resolution_minutes,
            unit=series.unit,
            row_count=len(records),
            expected_count=expected,
            missing_timestamps=_count_issues(vreport, "timestamp_gap"),
            duplicate_dropped=_count_issues(creport, "duplicate_dropped"),
            min_value=mn,
            max_value=mx,
            mean_value=avg,
            bronze_dir=str(fr.source) + "/" + fr.entity,
            bronze_sha256=fr.sha256,
            silver_csv=str(silver_csv),
        )
    )


def _run_entsoe(
    entsoe_client: EntsoeClient | None,
    start: datetime,
    end: datetime,
    bronze_root: Path,
    silver_root: Path,
    logger: logging.Logger,
) -> list[DatasetResult]:
    """Fetch all ENTSO-E core datasets (load, generation, prices)."""
    if entsoe_client is None:
        return _entsoe_unavailable(start, end)
    return (
        _run_entsoe_dataset(
            entsoe_client,
            entity="actual-total-load",
            fetcher=lambda c, a, s, e: c.fetch_load(a, s, e),
            parser=lambda b: parse_load(b),
            start=start, end=end,
            bronze_root=bronze_root, silver_root=silver_root,
        )
        + _run_entsoe_dataset(
            entsoe_client,
            entity="actual-generation-by-type",
            fetcher=lambda c, a, s, e: c.fetch_generation(a, s, e),
            parser=lambda b: parse_generation(b),
            psr_codes=list(WIND_PSR) + list(SOLAR_PSR),
            start=start, end=end,
            bronze_root=bronze_root, silver_root=silver_root,
        )
        + _run_entsoe_dataset(
            entsoe_client,
            entity="dayahead-prices",
            fetcher=lambda c, a, s, e: c.fetch_day_ahead_prices(a, s, e),
            parser=lambda b: parse_day_ahead_prices(b),
            start=start, end=end,
            bronze_root=bronze_root, silver_root=silver_root,
        )
    )


def _entsoe_unavailable(start: datetime, end: datetime) -> list[DatasetResult]:
    expected = _expected_hourly_count(start, end)
    return [
        DatasetResult(
            source="entsoe", entity=name, status="unavailable",
            note="ENTSOE_API_KEY not configured",
            requested_start=start, requested_end=end,
            expected_count=expected,
        )
        for name in ["actual-total-load", "actual-generation-by-type:B16",
                      "actual-generation-by-type:B18", "actual-generation-by-type:B19",
                      "dayahead-prices"]
    ]


# ── Gold step ──────────────────────────────────────────────────────────


def _build_gold(
    entsoe_datasets: list[DatasetResult],
    gold_root: Path,
    holiday_calendar,
    start: datetime,
    end: datetime,
    logger: logging.Logger,
) -> GoldResult:
    """Build Gold from available ENTSO-E silver artefacts."""
    # Collect hourly value maps from silver CSVs
    load_map: dict[datetime, float] = {}
    price_map: dict[datetime, float] = {}
    gen_by_psr: dict[str, dict[datetime, float]] = {}

    # Read each silver csv to populate maps
    for ds in entsoe_datasets:
        if ds.status != "verified_live" or not ds.silver_csv:
            continue
        try:
            rows = read_table(Path(ds.silver_csv))
        except Exception as exc:
            logger.warning("Cannot read silver %s: %r", ds.silver_csv, exc)
            continue
        ts_value: dict[datetime, float] = {}
        for row in rows:
            try:
                ts = ensure_utc(datetime.fromisoformat(row["timestamp_utc"]))
                val = float(row["value"])
            except Exception:
                continue
            ts_value[ts] = val
        if ds.entity == "actual-total-load":
            load_map.update(ts_value)
        elif ds.entity == "dayahead-prices":
            price_map.update(ts_value)
        elif ds.entity.startswith("actual-generation-by-type:"):
            code = ds.entity.split(":")[-1]
            gen_by_psr[code] = ts_value

    if not load_map:
        return GoldResult(
            produced=False,
            reason="No ENTSO-E load data available (ENTSOE_API_KEY not configured or no data returned)",
            actual_start=start,
            actual_end=end,
        )

    inputs = GoldInputs(
        load=load_map,
        generation_by_psr=gen_by_psr,
        day_ahead_price=price_map,
        holiday_calendar=holiday_calendar,
        sources={
            "load": "entsoe actual-total-load (A65/A16)",
            "price": "entsoe dayahead-prices (A44)",
            **{f"psr:{c}": f"entsoe actual-generation-by-type (A75) PSR={c}" for c in gen_by_psr},
        },
    )

    rows = build_gold_hourly(inputs)
    if not rows:
        return GoldResult(
            produced=False,
            reason="Gold builder returned 0 rows",
            actual_start=start,
            actual_end=end,
        )

    csv_path, _ = write_gold(
        gold_root,
        rows,
        generated_at=datetime.now(timezone.utc),
    )

    residual_vals = [r.residual_load_mw for r in rows if r.residual_load_mw is not None]
    neg = sum(1 for v in residual_vals if v < 0)
    rs: dict[str, float | None] = {}
    if residual_vals:
        rs["min"] = min(residual_vals)
        rs["max"] = max(residual_vals)
        rs["mean"] = sum(residual_vals) / len(residual_vals)

    missing = {
        "load_mw": sum(1 for r in rows if r.load_mw is None),
        "wind_mw": sum(1 for r in rows if r.wind_mw is None),
        "solar_mw": sum(1 for r in rows if r.solar_mw is None),
        "residual_load_mw": sum(1 for r in rows if r.residual_load_mw is None),
        "day_ahead_price_eur_mwh": sum(1 for r in rows if r.day_ahead_price_eur_mwh is None),
    }

    return GoldResult(
        produced=True,
        csv_path=str(csv_path),
        row_count=len(rows),
        actual_start=min(r.timestamp_utc for r in rows),
        actual_end=max(r.timestamp_utc for r in rows),
        missing_counts=missing,
        duplicates=0,
        residual_stats=rs,
        negative_residual_fraction=neg / len(residual_vals) if residual_vals else None,
    )


# ── Features step ──────────────────────────────────────────────────────

FEATURE_COLUMNS = [
    "target_utc",
    "target_local",
    "residual_load_mw",
    "hour",
    "day_of_week",
    "day_of_year",
    "month",
    "is_weekend",
    "is_holiday",
    "lag_1h",
    "lag_24h",
    "lag_48h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_mean_168h",
    "rolling_std_24h",
    "rolling_std_168h",
]


def _build_features(
    gold_result: GoldResult,
    weather_nl_hourly: dict[str, TimeSeries],
    holiday_calendar,
) -> FeatureResult:
    """Build the feature dataset from Gold + weather."""
    if not gold_result.produced or not gold_result.csv_path:
        return FeatureResult(produced=False, reason=gold_result.reason)

    rows = read_table(Path(gold_result.csv_path))
    if not rows:
        return FeatureResult(produced=False, reason="Gold CSV has 0 rows")

    # Build history from gold residual values
    history: list[tuple[datetime, float]] = []
    gold_rows_map: dict[datetime, dict] = {}
    for row in rows:
        try:
            ts = ensure_utc(datetime.fromisoformat(row["timestamp_utc"]))
        except Exception:
            continue
        gold_rows_map[ts] = row
        val = row.get("residual_load_mw")
        if val not in (None, ""):
            history.append((ts, float(val)))

    history.sort(key=lambda x: x[0])

    # Dynamic weather columns
    wx_cols = sorted(f"weather_{v}" for v in weather_nl_hourly)
    all_cols = FEATURE_COLUMNS + wx_cols
    feature_rows: list[list] = []

    for ts in sorted(gold_rows_map.keys()):
        row = gold_rows_map[ts]
        local = ensure_utc(ts).astimezone(AMSTERDAM_TZ)
        residual = row.get("residual_load_mw")
        residual_f = float(residual) if residual not in (None, "") else None

        cal = calendar_features(ts, holiday_calendar=holiday_calendar)

        # Lag features (value at target - N hours, must exist in history)
        lag_1 = _value_at(history, ts - timedelta(hours=1))
        lag_24 = _value_at(history, ts - timedelta(hours=24))
        lag_48 = _value_at(history, ts - timedelta(hours=48))
        lag_168 = _value_at(history, ts - timedelta(hours=168))

        # Rolling features (strictly before target)
        rm_24 = rolling_mean(history, ts, 24)
        rm_168 = rolling_mean(history, ts, 168)
        rs_24 = rolling_std(history, ts, 24)
        rs_168 = rolling_std(history, ts, 168)

        # Weather features (last observation strictly before target)
        wf = weather_features(weather_nl_hourly, ts)

        feature_rows.append([
            ts.isoformat(),
            local.isoformat(),
            residual_f,
            cal["hour"],
            cal["day_of_week"],
            cal["day_of_year"],
            cal["month"],
            cal["is_weekend"],
            cal.get("is_holiday", False),
            lag_1,
            lag_24,
            lag_48,
            lag_168,
            rm_24,
            rm_168,
            rs_24,
            rs_168,
            *[wf.get(v.replace("weather_", "")) for v in wx_cols],
        ])

    csv_path = Path(gold_result.csv_path).parent / "features.csv"
    meta_path = csv_path.with_suffix(".meta.json")
    write_table(
        csv_path,
        header=all_cols,
        rows=feature_rows,
        provenance=Provenance(
            tier="gold",
            source="derived+entsoe+open-meteo",
            entity="nl-hourly-features",
            unit="mixed",
        ),
        meta_path=meta_path,
        meta_extras={
            "columns": all_cols,
            "row_count": len(feature_rows),
            "asof_policy": (
                "as_of = target_utc (start of forecast hour). All lag/rolling/weather features "
                "use observations strictly before target. Calendar features derived from calendar "
                "only. No future information used. Limitation: weather variables are historical "
                "observations, not NWP forecasts — explicitly labelled for the forecasting phase."
            ),
            "weather_columns_are_historical": True,
        },
    )

    return FeatureResult(
        produced=True,
        csv_path=str(csv_path),
        row_count=len(feature_rows),
        columns=all_cols,
    )


# ── Main runner ────────────────────────────────────────────────────────


def run_pipeline(
    *,
    start: datetime,
    end: datetime,
    settings: Settings | None = None,
    open_meteo_client: OpenMeteoClient | None = None,
    entsoe_client: EntsoeClient | None = None,
    imbalance_source: ImbalancePricesSource | None = None,
    holiday_calendar: Any | None = None,
    report_dir: Path | None = None,
    logger: logging.Logger | None = None,
) -> PipelineRun:
    """Run the full Bronze → Silver → Gold → Features → report pipeline.

    This is the primary entry point.  Clients are injected via arguments
    for testability; when omitted, the default clients are wired from
    the centralised configuration.

    Returns a :class:`PipelineRun` carrying every artefact path and
    availability note for the data-quality report.
    """
    logger = logger or LOGGER
    settings = settings or get_settings()
    start = ensure_utc(start)
    end = ensure_utc(end)
    if end <= start:
        raise ValueError("end must be strictly after start")

    bronze_root = settings.bronze_dir
    silver_root = settings.silver_dir
    gold_root = settings.gold_dir
    report_out = report_dir or (settings.data_root / "reports")

    run = PipelineRun(
        settings=settings,
        requested_start=start,
        requested_end=end,
        started_at=datetime.now(timezone.utc),
        bronze_root=bronze_root,
        silver_root=silver_root,
        gold_root=gold_root,
        weather_locations=list(NL_POINTS),
    )

    # Wire clients
    if open_meteo_client is None:
        open_meteo_client = get_open_meteo_client(settings)

    entsoe_msg = ""
    if entsoe_client is None:
        try:
            entsoe_client = get_entsoe_client(settings)
        except ConfigurationError as exc:
            entsoe_msg = str(exc)
            entsoe_client = None

    if holiday_calendar is None:
        if settings.holidays_csv:
            from ..features.holiday import CsvHolidayCalendar
            holiday_calendar = CsvHolidayCalendar(settings.holidays_csv)
        else:
            from ..features.holiday import NetherlandsHolidayCalendar
            holiday_calendar = NetherlandsHolidayCalendar()

    logger.info("Pipeline run: %s -> %s (UTC)", start.isoformat(), end.isoformat())

    # ── Weather ──
    logger.info("Fetching weather...")
    wx_results, by_loc_var, wx_nl_hourly, dst_trans = _run_weather(
        open_meteo_client, start, end, bronze_root, silver_root, logger,
    )
    run.datasets.extend(wx_results)
    run.weather_variables = sorted(wx_nl_hourly.keys())
    run.weather_nl_hourly = wx_nl_hourly

    # DST observation counts
    run.dst_offsets_observed, run.dst_transitions_observed = _compute_dst(dst_trans)

    # ── Imbalance verification ──
    logger.info("Verifying imbalance...")
    imb_ds, imb_note = _verify_imbalance(
        entsoe_client, start, end, imbalance_source, bronze_root, silver_root, logger,
    )
    run.datasets.append(imb_ds)
    run.imbalance_note = imb_note

    # ── ENTSO-E core datasets ──
    logger.info("Fetching ENTSO-E datasets...")
    entsoe_results = _run_entsoe(entsoe_client, start, end, bronze_root, silver_root, logger)
    run.datasets.extend(entsoe_results)

    # ── Gold ──
    logger.info("Building Gold...")
    run.gold = _build_gold(entsoe_results, gold_root, holiday_calendar, start, end, logger)

    # ── Features ──
    logger.info("Building features...")
    run.features = _build_features(run.gold, wx_nl_hourly, holiday_calendar)

    # ── Report ──
    logger.info("Writing data-quality report...")
    run.finished_at = datetime.now(timezone.utc)
    try:
        report = DataQualityReport(run)
        json_path, md_path = report.write(report_out)
        run.report_paths = [json_path, md_path]
        logger.info("Report written: %s", json_path)
    except Exception as exc:
        run.errors.append(f"Report write failed: {exc!r}")
        logger.error("Failed to write report: %r", exc)

    # ── Summary ──
    wx_verified = sum(1 for d in wx_results if d.status == "verified_live")
    entsoe_verified = sum(1 for d in entsoe_results if d.status == "verified_live")
    entsoe_total = len(entsoe_results)
    summary_lines = [
        f"Open-Meteo: {wx_verified}/{len(wx_results)} location-variable datasets verified",
        f"ENTSO-E: {entsoe_verified}/{entsoe_total} datasets verified",
        f"Imbalance: {run.imbalance_note[:100]}",
        f"Gold: {'PRODUCED' if run.gold.produced else 'NOT PRODUCED'}",
        f"Features: {'PRODUCED' if run.features.produced else 'NOT PRODUCED'}",
    ]
    if run.entsoe_available and not entsoe_msg:
        summary_lines.append(f"ENTSO-E: API key configured - {len(entsoe_results)} datasets retrieved")
    else:
        summary_lines.append("ENTSO-E: API key NOT configured - ENTSO-E data unavailable")
    for line in summary_lines:
        logger.info("  %s", line)

    return run
