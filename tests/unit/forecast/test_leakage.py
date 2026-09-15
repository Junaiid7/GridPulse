"""THE leakage gate for Phase 4B.

Asserts the information-cutoff guarantee end-to-end: every predictor of an
issue row is built from data strictly-before the issue time; no predictor
depends on the (future) label of the target hour; the seasonal-naive vintage
is the issue row's own residual.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from gridpulse.forecast.contract import build_forecasting_dataset
from gridpulse.forecast.models import SeasonalNaiveModel
from gridpulse.forecast.split import chronological_split_by_fraction
from gridpulse.forecast.models.base import select_rows

UTC = timezone.utc


def _by_ts(rows):
    return {datetime.fromisoformat(r["target_utc"]): r for r in rows}


def test_lag24h_is_residual_at_issue_minus_24h(feature_rows):
    """lag_24h must equal the realised residual of (issue − 24h), never the target."""
    by_ts = _by_ts(feature_rows)
    ds = build_forecasting_dataset(feature_rows)
    for r in ds.rows:
        origin_ts = r.issue_time - timedelta(hours=24)
        if origin_ts not in by_ts:  # window-edge issues have no 24h history
            continue
        expected = float(by_ts[origin_ts]["residual_load_mw"])
        assert pytest.approx(r.features["lag_24h"]) == expected
        # And it must NOT be the label of the target hour.
        target_residual = float(by_ts[r.target_time]["residual_load_mw"])
        assert not math.isclose(target_residual, expected)


def test_all_lags_are_strictly_before_issue(feature_rows):
    """lag_1h / lag_48h / lag_168h reference the correct past residual hours."""
    by_ts = _by_ts(feature_rows)
    ds = build_forecasting_dataset(feature_rows)
    for r in ds.rows:
        for lag_col, lag_h in (("lag_1h", 1), ("lag_48h", 48), ("lag_168h", 168)):
            origin_ts = r.issue_time - timedelta(hours=lag_h)
            if origin_ts not in by_ts:
                continue
            expected = float(by_ts[origin_ts]["residual_load_mw"])
            assert pytest.approx(r.features[lag_col]) == expected


def test_rolling_windows_use_only_history_before_issue(feature_rows):
    by_ts = _by_ts(feature_rows)
    ds = build_forecasting_dataset(feature_rows)
    # rolling_mean_24h over the 24 values in [issue-24h, issue).
    for r in ds.rows:
        if r.issue_time - timedelta(hours=24) not in by_ts:
            continue
        window = [
            float(by_ts[r.issue_time - timedelta(hours=h)]["residual_load_mw"])
            for h in range(1, 25)
        ]
        assert pytest.approx(r.features["rolling_mean_24h"]) == sum(window) / 24.0


def test_predictors_are_invariant_to_the_target_label(feature_rows):
    """Editing the future label must not move any predictor of the issue row."""
    ds0 = build_forecasting_dataset(feature_rows)
    r0 = ds0.rows[len(ds0.rows) // 2]
    mutated = [dict(row) for row in feature_rows]
    for row in mutated:
        if row["target_utc"] == r0.target_time.isoformat():
            row["residual_load_mw"] = "99999.0"
    ds1 = build_forecasting_dataset(mutated)
    match = next(x for x in ds1.rows if x.issue_time == r0.issue_time)
    assert match.features == r0.features  # predictors unchanged
    assert match.target_mw == pytest.approx(99999.0)  # label reflects the edit


def test_naive_vintage_is_issue_row_residual(feature_rows):
    """Seasonal naive predicts residual(T − 24h) = the issue row's residual."""
    by_ts = _by_ts(feature_rows)
    ds = build_forecasting_dataset(feature_rows)
    model = SeasonalNaiveModel(lag_hours=24)
    window = (ds.rows[0].issue_time, ds.rows[-1].issue_time + timedelta(hours=1))
    preds = model.predict(ds, start=window[0], end=window[1])
    select = select_rows(ds, window[0], window[1])
    assert len(preds) == len(select)
    for r, p in zip(select, preds):
        expected = float(by_ts[r.issue_time]["residual_load_mw"])
        assert p is not None
        assert pytest.approx(p) == expected


def test_benchmark_split_rejects_leaks_via_chronology(feature_rows):
    """The split used by the benchmark is strictly chronological: train rows are
    all strictly-before validation rows, which are before test rows."""
    by_ts = _by_ts(feature_rows)
    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    train_issues = {r.issue_time for r in ds.rows if r.issue_time < split.train_end}
    test_issues = {r.issue_time for r in ds.rows if r.issue_time >= split.validation_end}
    assert train_issues and test_issues
    assert max(train_issues) < min(test_issues)