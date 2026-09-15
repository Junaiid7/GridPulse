"""Evaluation metrics and bootstrap CIs: exact known values, near-zero MAPE
guard, empty input, determinism under a fixed seed."""

from __future__ import annotations

import math

import pytest

from gridpulse.forecast.evaluate import (
    bootstrap_mae_ci,
    bootstrap_mae_difference_ci,
    compute_point_metrics,
)


def test_metric_values_are_exact():
    actuals = [10.0, 20.0, 30.0]
    preds = [12.0, 20.0, 27.0]
    m = compute_point_metrics(actuals, preds)
    assert m["n_predictions"] == 3
    assert m["n_valid_pairs"] == 3
    assert m["mae"] == pytest.approx(5 / 3)          # (2+0+3)/3
    assert m["rmse"] == pytest.approx(math.sqrt(13 / 3))  # (4+0+9)/3
    assert m["bias"] == pytest.approx(-1 / 3)        # (2+0-3)/3
    assert m["median_absolute_error"] == pytest.approx(2.0)  # sorted [0,2,3]
    assert m["mape"] == pytest.approx(10.0)          # (0.2+0+0.1)/3*100


def test_mape_guards_near_zero_actual():
    actuals = [0.0, 5.0]
    preds = [1.0, 5.0]
    m = compute_point_metrics(actuals, preds)
    assert m["mape"] == pytest.approx(0.0)      # only the 5.0 row is valid
    assert m["mape_invalid_count"] == 1
    assert m["mae"] == pytest.approx(0.5)       # |0-1|,|5-5|

    # ALL actuals near zero -> mape is None (never misleading), count recorded.
    m2 = compute_point_metrics([0.0, 0.0], [1.0, 2.0])
    assert m2["mape"] is None
    assert m2["mape_invalid_count"] == 2
    assert m2["n_valid_pairs"] == 2


def test_empty_input():
    m = compute_point_metrics([], [])
    assert m["n_predictions"] == 0
    assert all(m[k] is None for k in ("mae", "rmse", "mape", "bias", "median_absolute_error"))


def test_none_entries_are_skipped_not_errors():
    actuals = [1.0, None, 3.0]
    preds = [2.0, 0.0, None]
    m = compute_point_metrics(actuals, preds)
    assert m["n_predictions"] == 3
    assert m["n_valid_pairs"] == 1  # only the (1.0, 2.0) row counts
    assert m["mae"] == pytest.approx(1.0)


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        compute_point_metrics([1.0], [1.0, 2.0])


def test_bootstrap_ci_is_seed_deterministic():
    actuals = [10.0, 20.0, 30.0, 40.0, 50.0, 12.0, 33.0, 21.0]
    preds = [11.0, 19.0, 31.0, 41.0, 49.0, 13.0, 30.0, 20.0]
    a = bootstrap_mae_ci(actuals, preds, n_boot=500, seed=42)
    b = bootstrap_mae_ci(actuals, preds, n_boot=500, seed=42)
    assert a == b
    assert a["ci_low"] <= a["mae"] <= a["ci_high"]
    assert a["n_samples"] == 8


def test_difference_ci_sign_and_consistency():
    actuals = [10.0, 20.0, 30.0, 40.0, 50.0]
    pred_a = [16.0, 26.0, 36.0, 46.0, 56.0]  # systematically worse
    pred_b = [10.5, 19.5, 30.5, 39.5, 50.5]   # near-perfect
    d = bootstrap_mae_difference_ci(actuals, pred_a, pred_b, n_boot=500, seed=0)
    assert d["difference"] > 0       # A worse than B
    assert d["mae_a"] > d["mae_b"]
    assert d["ci_low"] <= d["ci_high"]
    # Determinism:
    d2 = bootstrap_mae_difference_ci(actuals, pred_a, pred_b, n_boot=500, seed=0)
    assert d == d2


def test_difference_ci_nonexclusive_pairs_dropped():
    actuals = [1.0, None, 3.0]
    pred_a = [2.0, 5.0, None]
    pred_b = [1.5, 2.0, 2.0]
    d = bootstrap_mae_difference_ci(actuals, pred_a, pred_b, n_boot=100, seed=1)
    assert d["n_samples"] == 1  # only row 0 survived (a,b,actual all defined)