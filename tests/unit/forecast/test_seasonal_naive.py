"""Seasonal-naive baseline: persistence of residual(T − 24h), explicit
insufficient-history handling (None, never a silent fill)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from gridpulse.forecast.contract import build_forecasting_dataset
from gridpulse.forecast.evaluate import compute_point_metrics
from gridpulse.forecast.models import SeasonalNaiveModel

UTC = timezone.utc


def _periodic_feature_rows(n_days: int = 6) -> list[dict]:
    """Hourly feature rows with residual(t) = residual(t-24h) EXACTLY."""
    start = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    rows = []
    for hour in range(n_days * 24):
        t = start + timedelta(hours=hour)
        residual = round(1000.0 * math.sin(2 * math.pi * (t.hour - 6) / 24), 3)
        rows.append(
            {
                "target_utc": t.isoformat(),
                "target_local": t.isoformat(),
                "residual_load_mw": repr(residual),
                "hour": str(t.hour),
                "day_of_week": str(t.weekday()),
                "day_of_year": str(t.timetuple().tm_yday),
                "month": str(t.month),
                "is_weekend": "true" if t.weekday() >= 5 else "false",
                "is_holiday": "false",
                "lag_1h": repr(residual),  # not used by naive
                "lag_24h": repr(residual),
                "lag_48h": repr(residual),
                "lag_168h": repr(residual),
                "rolling_mean_24h": repr(residual),
                "rolling_mean_168h": repr(residual),
                "rolling_std_24h": "10.0",
                "rolling_std_168h": "10.0",
            }
        )
    return rows


def test_perfect_persistence_is_exact(feature_rows):
    ds = build_forecasting_dataset(_periodic_feature_rows())
    model = SeasonalNaiveModel(lag_hours=24)
    start, end = ds.rows[1].issue_time, ds.rows[-1].issue_time + timedelta(hours=1)
    model.fit(ds, start=start, end=end)
    preds = model.predict(ds, start=start, end=end)
    actuals = [r.target_mw for r in ds.rows if start <= r.issue_time < end]
    assert len(preds) == len(actuals)
    assert all(p is not None for p in preds)
    metrics = compute_point_metrics(actuals, preds)
    assert metrics["mae"] == pytest.approx(0.0, abs=1e-9)


def test_insufficient_history_returns_none_never_filled(feature_rows):
    rows = _periodic_feature_rows(n_days=6)
    # Give the middle issue row a missing residual -> no vintage -> None.
    middle = datetime(2024, 1, 2, 6, 0, tzinfo=UTC)
    for row in rows:
        if row["target_utc"] == middle.isoformat():
            row["residual_load_mw"] = ""
    ds = build_forecasting_dataset(rows)
    model = SeasonalNaiveModel(lag_hours=24)
    start, end = ds.rows[1].issue_time, ds.rows[-1].issue_time + timedelta(hours=1)
    preds = model.predict(ds, start=start, end=end)
    assert any(p is None for p in preds)  # explicitly undetermined
    # And metrics never count it as an error.
    actuals = [r.target_mw for r in ds.rows if start <= r.issue_time < end]
    metrics = compute_point_metrics(actuals, preds)
    assert metrics["n_valid_pairs"] == sum(p is not None for p in preds)
    assert metrics["n_predictions"] == len(preds)


def test_stateless_metadata():
    model = SeasonalNaiveModel(lag_hours=24)
    model.fit(None, start=None, end=None)  # stateless — fit is a no-op
    meta = model.metadata()
    assert meta["rule"] == "prediction(T) = residual_load(T - lag_hours)"
    assert meta["lag_hours"] == 24
    assert "None" in meta["insufficient_history"]  # documented behaviour