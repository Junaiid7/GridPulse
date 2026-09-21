"""Phase 4D-B leak and isolation guarantees (FIXTURE-VERIFIED).

The backtest must be *measurement-clean*:

- forecasts are fit on the train window only (no offset model sees test rows);
- the dispatch decision never sees the realised residual — settlement is the
  only place realised data enters;
- the split is strictly chronological and half-open, shared by all 24 offsets;
- the input prices follow the documented 06:00 UTC price-availability
  convention and equal the gold day-ahead rows;
- squalid as the few synthetic days are, rerunning is bit-for-bit identical.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from gridpulse.forecast.contract import build_forecasting_dataset
from gridpulse.forecast.split import (
    chronological_split_by_fraction,
    partition_timestamps,
)
from gridpulse.optimization import (
    build_forecast_profiles,
    run_dispatch_backtest,
    settle_day,
)

UTC_ISSUE = 6


@pytest.fixture(scope="module")
def profiles(feature_rows):
    return build_forecast_profiles(feature_rows)


@pytest.fixture(scope="module")
def backtest(feature_rows, gold_rows):
    return run_dispatch_backtest(feature_rows, gold_rows)


def _offset_dataset(feature_rows, k):
    return build_forecasting_dataset(
        feature_rows,
        issue_hour_utc=UTC_ISSUE,
        horizon_hours=24 + k,
        naive_lag_hours=24,
    )


# ============================================================================
# Train-only fit
# ============================================================================
def test_train_only_fit_rows(feature_rows, profiles):
    """Every offset's ``n_train_rows`` equals its train-window rows only."""
    offsets = {o["offset"]: o for o in profiles.offsets}
    split = profiles.split
    for k in (0, 7, 12, 23):
        off = offsets[k]
        assert off["fitted"] is True
        ds = _offset_dataset(feature_rows, k)
        expected = sum(
            1
            for r in ds.rows
            if split.train_start <= r.issue_time < split.train_end
            and r.target_mw is not None
            and r.has_features()
        )
        assert off["n_train_rows"] == expected


def test_test_window_rows_are_predict_only(feature_rows, profiles):
    """``n_test_rows`` per offset == its issue rows in the shared test window."""
    offsets = {o["offset"]: o for o in profiles.offsets}
    split = profiles.split
    for k in range(24):
        ds = _offset_dataset(feature_rows, k)
        test_issues = partition_timestamps(ds.issue_times, split)["test"]
        actual = len(
            [r for r in ds.rows if split.test_start <= r.issue_time < split.test_end]
        )
        assert offsets[k]["n_test_rows"] == len(test_issues) == actual
        assert len(test_issues) >= 1
        # every test issue time lies in the test window proper
        for ts in test_issues:
            assert split.test_start <= ts < split.test_end


# ============================================================================
# Chronological, half-open, shared split
# ============================================================================
def test_split_half_open_contiguous_and_aware(profiles):
    split = profiles.split
    for dt in (
        split.train_start,
        split.train_end,
        split.validation_start,
        split.validation_end,
        split.test_start,
        split.test_end,
    ):
        assert dt.tzinfo is not None
    assert split.train_start < split.train_end
    assert split.train_end == split.validation_start
    assert split.validation_end == split.test_start
    assert split.test_start < split.test_end


def test_split_matches_recomputed_fraction_split(feature_rows, profiles):
    base = build_forecasting_dataset(
        feature_rows, issue_hour_utc=UTC_ISSUE, horizon_hours=24, naive_lag_hours=24
    )
    recomputed = chronological_split_by_fraction(base.issue_times)
    assert profiles.split == recomputed
    # end-exclusive test window = last issue time + 1h (hourly-grid convention)
    assert recomputed.test_end == max(base.issue_times) + timedelta(hours=1)
    assert recomputed.train_start == min(base.issue_times)


def test_dispatched_days_are_chronological(backtest, profiles):
    """Dispatch window == shared test window as recorded; dropped days are sane."""
    assert backtest.info["n_profiles"] == len(profiles.profiles)
    assert backtest.info["n_dispatch_days"] <= backtest.info["n_profiles_complete"]
    s = backtest.split
    assert s["train_start"] < s["validation_start"] < s["test_start"] < s["test_end"]
    # dropped-days keys are ISO timestamps in ascending order
    stamps = sorted(backtest.dropped_days)
    assert stamps == list(backtest.dropped_days)


# ============================================================================
# Settlement isolation (realised never reaches the decision)
# ============================================================================
def test_settlement_only_place_realised_enters(profiles, gold_by_ts):
    """Identical dispatch schedule & throughput for different realised vectors;
    only the settled cost (and possibly undercut count) changes."""
    prof = next(p for p in profiles.profiles.values() if p.complete)
    prices = [
        float(gold_by_ts[ts]["day_ahead_price_eur_mwh"]) for ts in prof.target_times
    ]
    realised_a = [float(gold_by_ts[ts]["residual_load_mw"]) for ts in prof.target_times]
    realised_b = [r + 25.0 for r in realised_a]  # different realised horizon

    from gridpulse.optimization import assemble_dispatch_input, run_dispatch

    inputs = assemble_dispatch_input(prof, prices)
    res = run_dispatch(inputs, strategy="lp_p50")

    sa = settle_day(res, realised_a, prices)
    sb = settle_day(res, realised_b, prices)
    # the schedule object is the same -> identical energy bookkeeping
    assert sa["throughput_charge_mwh"] == sb["throughput_charge_mwh"]
    assert sa["throughput_discharge_mwh"] == sb["throughput_discharge_mwh"]
    assert sa["energy_neutrality_delta_mwh"] == sb["energy_neutrality_delta_mwh"]
    # but the honest settled cost tracks the realised profile
    assert sa["cost_eur"] != sb["cost_eur"]


def test_realised_column_only_changes_settlement(feature_rows, gold_rows):
    """Doubling every gold *residual* value leaves schedules untouched: throughput
    is identical across the two backtests, only settled costs / undercut move."""
    gold_shifted = [
        {**row, "residual_load_mw": str(2.0 * float(row["residual_load_mw"]))}
        for row in gold_rows
    ]
    bt_base = run_dispatch_backtest(feature_rows, gold_rows)
    bt_shift = run_dispatch_backtest(feature_rows, gold_shifted)

    # dispatch decisions depend only on price + forecast profile
    for s_base, s_shift in zip(bt_base.strategies, bt_shift.strategies):
        assert s_base["strategy"] == s_shift["strategy"]
        assert s_base["total_throughput_charge_mwh"] == pytest.approx(
            s_shift["total_throughput_charge_mwh"]
        )
        assert s_base["total_throughput_discharge_mwh"] == pytest.approx(
            s_shift["total_throughput_discharge_mwh"]
        )
    # but settlement is honest: doubled realised demand costs more
    nb_base = next(s for s in bt_base.strategies if s["strategy"] == "no_battery")
    nb_shift = next(s for s in bt_shift.strategies if s["strategy"] == "no_battery")
    assert nb_shift["mean_daily_cost_eur"] > nb_base["mean_daily_cost_eur"]


# ============================================================================
# Price-availability convention (06:00 UTC)
# ============================================================================
def test_price_convention_recorded(backtest):
    info = backtest.info
    assert info["issue_cadence"] == "once_daily"
    assert info["issue_hour_utc"] == 6
    assert info["price_availability_convention"]
    assert "06:00 UTC" in info["price_availability_convention"]
    assert "convention" in info["price_availability_convention"].lower()


def test_input_prices_are_gold_values(feature_rows, gold_rows, gold_by_ts, profiles):
    """Dispatcher prices equal the gold day-ahead rows for the operating day."""
    from gridpulse.optimization import assemble_dispatch_input

    n = 0
    for prof in profiles.profiles.values():
        if not prof.complete:
            continue
        prices = [
            float(gold_by_ts[ts]["day_ahead_price_eur_mwh"]) for ts in prof.target_times
        ]
        inputs = assemble_dispatch_input(prof, prices)
        for ts, p in zip(inputs.target_times, inputs.price_eur_mwh):
            assert float(gold_by_ts[ts]["day_ahead_price_eur_mwh"]) == pytest.approx(p)
        n += 1
    assert n >= 1


# ============================================================================
# Determinism
# ============================================================================
def test_profiles_fully_deterministic(feature_rows):
    p1 = build_forecast_profiles(feature_rows)
    p2 = build_forecast_profiles(feature_rows)
    d1 = [dict(o) for o in p1.offsets]
    d2 = [dict(o) for o in p2.offsets]
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)
    assert [prof.to_dict() for _, prof in sorted(p1.profiles.items())] == [
        prof.to_dict() for _, prof in sorted(p2.profiles.items())
    ]
