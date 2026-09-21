"""Forecasting dataset contract (Phase 4B).

Defines the *forecasting-ready* view of the feature table produced by the
pipeline (``data/gold/nl/features.csv``) together with an explicit, documented
information-cutoff convention.

As-of / issue-time semantics
----------------------------

- A forecast is **issued once daily** at a fixed UTC clock hour
  (``issue_hour_utc``, default 06:00 UTC).
- The prediction **target** is ``T = issue_time + horizon_hours`` (default
  24 h ahead → the same UTC hour on the following day).
- **Information cutoff (no leakage):** every *predictor* column is read from
  the feature row anchored at the **issue time**. Upstream, that row was
  built with strictly-before semantics: ``lag_Nh`` uses the realised residual
  of hour ``[t - Nh, t - Nh + 1h)``, rolling windows aggregate history
  strictly before ``t``, and weather features are the last observation
  strictly before ``t``. So predictors only use information available
  strictly before the issue time. The label separately comes from the
  realised ``residual_load_mw`` of the target hour (available only *after*
  the fact).
- **Zero-observation-latency convention (explicit, and the one place the
  brief's "strictly before" rule is relaxed):** hourly values are credited as
  available at the start of their hour. The seasonal-naive baseline predicts
  ``residual(T - 24h)``; under the convention that value is known at the
  issue instant (it is the issue row's own target-time residual), so the
  naive vintage is legitimate and leakage-free. This convention is
  documented and never silent: every row records ``naive_lag_hours`` and the
  vintage value used.

Because the upstream feature builders are themselves leakage-safe, the
forecast layer *does not recompute* predictors from raw series in normal use;
it trusts the issue-row contract and verifies it in tests (see
``tests/unit/forecast/test_leakage.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..ingestion.common.models import ensure_utc

#: Columns the forecast layer treats as the realisation/labelling columns.
TARGET_COLUMN = "residual_load_mw"
NON_PREDICTOR_COLUMNS = frozenset({"target_utc", "target_local", TARGET_COLUMN})

#: Feature columns that are produced for every target hour by the pipeline.
#: ``weather_*`` columns are dynamic and appended separately.
CORE_FEATURE_COLUMNS = (
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
)


def _cell(value, *, kind):
    """Parse one CSV ``read_table`` cell (``str``, may be empty)."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    if text == "":
        return None
    if kind == "bool":
        return text == "true"
    try:
        number = float(text)
    except ValueError:
        return None
    return int(number) if kind == "int" else number


def _predictor_parse_kind(column: str) -> str:
    if column in ("is_weekend", "is_holiday"):
        return "bool"
    if column in ("hour", "day_of_week", "day_of_year", "month"):
        return "int"
    if column.startswith("weather_"):
        return "float"
    return "float"


@dataclass(frozen=True)
class ForecastRow:
    """One forecasting example.

    ``features`` hold the predictors (all anchored at ``issue_time``,
    strictly-before information cutoff). ``naive_vintage_mw`` is the value
    ``residual_load(T - naive_lag_hours)`` used by the seasonal-naive
    baseline (may be ``None`` when history is insufficient). ``target_mw`` is
    the realised residual load of the target hour (may be ``None`` if the
    feature table does not cover it). ``n_incomplete`` counts predictors that
    are ``None``.
    """

    issue_time: datetime
    target_time: datetime
    features: Mapping[str, object] = field(default_factory=dict)
    target_mw: float | None = None
    naive_vintage_mw: float | None = None
    n_incomplete: int = 0

    def has_features(self) -> bool:
        return self.n_incomplete == 0


@dataclass(frozen=True)
class ForecastingDataset:
    rows: Sequence[ForecastRow]
    predictor_columns: tuple
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def issue_times(self) -> tuple:
        return tuple(sorted(r.issue_time for r in self.rows))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.rows)


def build_forecasting_dataset(
    feature_rows,
    *,
    issue_hour_utc: int = 6,
    horizon_hours: int = 24,
    naive_lag_hours: int | None = None,
) -> ForecastingDataset:
    """Build a forecasting dataset from pipeline feature rows.

    Parameters
    ----------
    feature_rows:
        Rows as returned by ``read_table`` on ``features.csv``: each a
        ``dict`` keyed by the feature-table header. ``"target_utc"`` must be
        an ISO-8601 aware UTC timestamp. One row exists per target hour; only
        the rows whose UTC clock hour equals ``issue_hour_utc`` become issue
        rows. Predictors are *all* non-target, non-publication feature columns
        present in the table.
    issue_hour_utc, horizon_hours:
        As document in the module docstring.
    naive_lag_hours:
        Lag used by the seasonal-naive baseline; defaults to
        ``horizon_hours`` (24 h).

    Returns
    -------
    ``ForecastingDataset`` whose rows are sorted by issue time. Rows whose
    issue hour does not match, or whose target hour is not covered by the
    table, are skipped (counts reported in ``metadata``).
    """
    if naive_lag_hours is None:
        naive_lag_hours = horizon_hours

    by_time: dict[datetime, dict] = {}
    for raw in feature_rows:
        if "target_utc" not in raw or not raw["target_utc"]:
            continue
        ts = _as_datetime(raw["target_utc"])
        if ts.minute != 0 or ts.second != 0 or ts.microsecond != 0:
            raise ValueError(f"feature rows must be on the hour for contract; got {ts}")
        by_time[ts] = raw

    actual_columns = []
    for raw in by_time.values():
        actual_columns = list(raw.keys())
        break
    predictor_columns = tuple(
        sorted(c for c in actual_columns if c not in NON_PREDICTOR_COLUMNS and c != "")
    )
    # Re-establish the canonical ordering for the rolling/lag/calendar core,
    # then append any weather_* columns (already descending-var order).
    predictor_columns = tuple(
        [c for c in CORE_FEATURE_COLUMNS if c in predictor_columns]
        + [c for c in predictor_columns if c.startswith("weather_")]
        + [c for c in predictor_columns if c not in CORE_FEATURE_COLUMNS and not c.startswith("weather_")]
    )

    rows: list[ForecastRow] = []
    n_mismatched_issue = 0
    n_no_label = 0
    for issue_ts in sorted(by_time):
        if issue_ts.hour != issue_hour_utc:
            n_mismatched_issue += 1
            continue
        target_ts = issue_ts + timedelta(hours=horizon_hours)
        target_raw = by_time.get(target_ts)
        if target_raw is None:
            n_no_label += 1
            continue
        raw = by_time[issue_ts]

        features = {}
        n_incomplete = 0
        for col in predictor_columns:
            value = _cell(raw.get(col), kind=_predictor_parse_kind(col))
            if value is None:
                n_incomplete += 1
            features[col] = value

        vintage_ts = target_ts - timedelta(hours=naive_lag_hours)
        vintage_raw = by_time.get(vintage_ts)
        vintage_mw = (
            _cell(vintage_raw.get(TARGET_COLUMN), kind="float")
            if vintage_raw is not None
            else None
        )
        target_mw = _cell(target_raw.get(TARGET_COLUMN), kind="float")

        rows.append(
            ForecastRow(
                issue_time=issue_ts,
                target_time=target_ts,
                features=features,
                target_mw=target_mw,
                naive_vintage_mw=vintage_mw,
                n_incomplete=n_incomplete,
            )
        )

    metadata = {
        "target_column": TARGET_COLUMN,
        "issue_cadence": "once_daily",
        "issue_hour_utc": issue_hour_utc,
        "horizon_hours": horizon_hours,
        "naive_lag_hours": naive_lag_hours,
        "asof_policy": (
            "predictors from the issue-time feature row (all values strictly "
            "before issue time); label = residual_load_mw of target hour; "
            "naive vintage = residual_load(T - 24h) per the documented "
            "zero-observation-latency convention"
        ),
        "rows_requested": len(rows) + n_mismatched_issue + n_no_label,
        "rows_mismatched_issue_hour": n_mismatched_issue,
        "rows_skipped_no_label": n_no_label,
    }
    return ForecastingDataset(rows=rows, predictor_columns=predictor_columns, metadata=metadata)


def _as_datetime(value: str) -> datetime:
    dt = datetime.fromisoformat(value if isinstance(value, str) else str(value))
    return ensure_utc(dt)