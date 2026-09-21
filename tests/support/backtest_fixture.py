"""Synthetic backtest fixtures for Phase 4D-B testing.

Reuses the synthetic electricity pipeline runner to provide the two inputs the
backtest needs:

- **feature rows** — via ``features_rows(run)`` (forecast predictors + labels);
- **gold hour rows** — via :func:`gold_hour_rows` (realised residual + day-ahead
  price per UTC hour, read from the pipeline's ``hourly.csv``), which is what
  both ``build_forecast_profiles`` and ``run_dispatch_backtest`` consume for
  dispatch prices and settlement.

Everything here is FIXTURE-VERIFIED (synthetic inputs, no real ENTSO-E data).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from gridpulse.transformation.csvio import read_table


def gold_hour_rows(pipeline_run, *, area: str = "nl") -> list[dict]:
    """Read the pipeline's gold hourly table (realised residual + prices).

    Uses ``run.gold.csv_path`` directly so it keeps working whether the pipeline
    was run into a scratch dir (tests/demo) or a real data root. Returns the
    ``read_table`` dicts (one per UTC hour), with ``timestamp_utc`` as an
    ISO string and ``residual_load_mw`` / ``day_ahead_price_eur_mwh`` as
    floats (or empty strings when missing, like every pipeline table).
    """
    if not pipeline_run.gold.produced or not pipeline_run.gold.csv_path:
        raise ValueError("pipeline produced no gold table; cannot read hourly.csv")
    return read_table(Path(pipeline_run.gold.csv_path))


def gold_by_timestamp(gold_rows: list[dict]) -> dict[datetime, dict]:
    """Index gold rows by their aware-UTC ``timestamp_utc``.

    Rows whose ``timestamp_utc`` is unparsable/naive are skipped (never
    guessed). The lookback key is normalised to aware UTC so callers can look
    up with aware datetimes directly.
    """
    from gridpulse.ingestion.common.models import ensure_utc

    out: dict[datetime, dict] = {}
    for row in gold_rows:
        raw = row.get("timestamp_utc")
        if not raw:
            continue
        try:
            ts = ensure_utc(datetime.fromisoformat(str(raw)))
        except (TypeError, ValueError):
            continue
        out[ts] = row
    return out


__all__ = ["gold_hour_rows", "gold_by_timestamp"]
