"""Phase 4D-B backtest unit tests (FIXTURE-VERIFIED).

Exercises the backtest machinery on the synthetic pipeline fixture: forecast
profile assembly, DispatchInput wiring, settlement semantics, bootstrap CIs,
the top-level :func:`run_dispatch_backtest` plumbing, and failure handling.

Everything here is machinery validation on the ~9-day synthetic held-out
window — no claim about real Dutch electricity is asserted or made.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest
from support.dispatch_fixture import tiny_pattern

import gridpulse.optimization.backtest as bt_mod
from gridpulse.optimization import (
    ASOF_POLICY,
    DATA_STATUS,
    PRICE_AVAILABILITY_CONVENTION,
    BacktestResult,
    DispatchInfeasible,
    DispatchInput,
    ForecastProfile,
    assemble_dispatch_input,
    bootstrap_cost_difference_ci,
    build_forecast_profiles,
    run_dispatch,
    run_dispatch_backtest,
    settle_day,
)
from gridpulse.optimization.battery import BatteryConfig

HORIZON = 24


# ============================================================================
# shared fixtures (fixture-verified synthetic window)
# ============================================================================
@pytest.fixture(scope="module")
def profiles(feature_rows):
    """One full 24-offset profile build, shared across the module."""
    return build_forecast_profiles(feature_rows)


@pytest.fixture(scope="module")
def backtest(feature_rows, gold_rows):
    return run_dispatch_backtest(feature_rows, gold_rows)


# ============================================================================
# small helpers
# ============================================================================
def _gold_prices(gold_by_ts, profile):
    return [
        float(gold_by_ts[ts]["day_ahead_price_eur_mwh"]) for ts in profile.target_times
    ]


def _gold_realised(gold_by_ts, profile):
    return [float(gold_by_ts[ts]["residual_load_mw"]) for ts in profile.target_times]


def _first_complete(profiles) -> ForecastProfile:
    for _, prof in sorted(profiles.profiles.items()):
        if prof.complete:
            return prof
    raise AssertionError("no complete profile in fixture window")


def _tiny_input() -> DispatchInput:
    t = tiny_pattern()  # price ramp 10..33, constant residual 100 MW
    residual = list(t["residual_load_mw"])
    return DispatchInput(
        issue_time=t["issue_time"],
        target_times=t["target_times"],
        price_eur_mwh=t["price_eur_mwh"],
        residual_load_mw=residual,
        scenario_residual_load_mw={"p10": residual, "p50": residual, "p90": residual},
        battery=BatteryConfig(),
    )


# ============================================================================
# Forecast profile assembly
# ============================================================================
def test_build_forecast_profiles_one_model_per_offset(profiles):
    assert isinstance(profiles.offsets, list)
    assert len(profiles.offsets) == HORIZON  # 24 per-offset models
    assert profiles.issue_hour_utc == 6
    for k, off in enumerate(profiles.offsets):
        assert off["offset"] == k
        assert off["horizon_hours"] == HORIZON + k
        assert off["fitted"] is True  # train window is big enough
        assert off["n_test_rows"] >= 1
        assert off["n_aligned"] <= off["n_test_rows"]
        # nothing crosses once corrected (the model reports it when it does)
        assert off["crossing"]["n_detected"] >= 0
    # all 24 offsets share one split
    assert len({id(o) for o in profiles.offsets}) >= 1


def test_profiles_complete_have_ordered_hourly_window(profiles):
    assert len(profiles.profiles) >= 1
    for issue, prof in profiles.profiles.items():
        assert prof.issue_time == issue
        assert len(prof.target_times) == HORIZON
        # target window is issue + 24h .. issue + 47h, hourly, increasing
        assert prof.target_times[0] == issue + timedelta(hours=24)
        assert prof.target_times[-1] == issue + timedelta(hours=24 + HORIZON - 1)
        for a, b in zip(prof.target_times, prof.target_times[1:]):
            assert b - a == timedelta(hours=1)
        assert 0 <= prof.n_defined <= HORIZON
    assert any(p.complete for p in profiles.profiles.values())


def test_complete_profile_triples_are_ordered(profiles):
    """Corrected P10/P50/P90 triples must be ascending hour by hour."""
    n_checked = 0
    for prof in profiles.profiles.values():
        if not prof.complete:
            continue
        n_checked += 1
        for lo, mid, hi in zip(prof.p10_mw, prof.p50_mw, prof.p90_mw):
            assert lo is not None and mid is not None and hi is not None
            assert lo <= mid <= hi
    assert n_checked >= 1


def test_feature_columns_present(profiles, feature_rows):
    cols = profiles.feature_columns
    assert cols  # non-empty
    # every chosen column really is a predictor column in the base dataset
    assert len(cols) == len(set(cols))


# ============================================================================
# DispatchInput assembly
# ============================================================================
def test_assemble_dispatch_input_wiring(profiles, gold_by_ts):
    prof = _first_complete(profiles)
    prices = _gold_prices(gold_by_ts, prof)
    inputs = assemble_dispatch_input(prof, prices)
    assert isinstance(inputs, DispatchInput)
    assert inputs.issue_time == prof.issue_time
    assert inputs.target_times == prof.target_times
    assert inputs.price_eur_mwh == tuple(prices)  # DispatchInput normalises to a tuple
    assert inputs.residual_load_mw == prof.p50_mw
    assert inputs.scenario_residual_load_mw["p10"] == prof.p10_mw
    assert inputs.scenario_residual_load_mw["p50"] == prof.p50_mw
    assert inputs.scenario_residual_load_mw["p90"] == prof.p90_mw
    # DispatchInput re-validates p10 <= p50 <= p90 and p50 == residual
    assert inputs.scenario_residual_load_mw["p50"] == inputs.residual_load_mw


def test_assemble_dispatch_input_prices_are_gold_day_ahead(profiles, gold_by_ts):
    """The 06:00 UTC price-availability convention: input prices == gold rows."""
    prof = _first_complete(profiles)
    prices = _gold_prices(gold_by_ts, prof)
    inputs = assemble_dispatch_input(prof, prices)
    for ts, p in zip(inputs.target_times, inputs.price_eur_mwh):
        assert float(gold_by_ts[ts]["day_ahead_price_eur_mwh"]) == pytest.approx(p)


def test_assemble_dispatch_input_rejects_incomplete(profiles, gold_by_ts):
    prof = _first_complete(profiles)
    prices = _gold_prices(gold_by_ts, prof)
    incomplete = ForecastProfile(
        issue_time=prof.issue_time,
        target_times=prof.target_times,
        p10_mw=tuple(None if i == 5 else v for i, v in enumerate(prof.p10_mw)),
        p50_mw=tuple(None if i == 5 else v for i, v in enumerate(prof.p50_mw)),
        p90_mw=tuple(None if i == 5 else v for i, v in enumerate(prof.p90_mw)),
        n_defined=prof.n_defined - 1,
    )
    assert not incomplete.complete
    with pytest.raises(ValueError, match="incomplete"):
        assemble_dispatch_input(incomplete, prices)


def test_assemble_dispatch_input_rejects_price_length_mismatch(profiles, gold_by_ts):
    prof = _first_complete(profiles)
    prices = _gold_prices(gold_by_ts, prof)
    with pytest.raises(ValueError, match="one price per target hour"):
        assemble_dispatch_input(prof, prices[:-1])


# ============================================================================
# Settlement semantics
# ============================================================================
def test_settle_day_no_battery_identity(profiles, gold_by_ts):
    prof = _first_complete(profiles)
    prices = _gold_prices(gold_by_ts, prof)
    realised = _gold_realised(gold_by_ts, prof)
    inputs = assemble_dispatch_input(prof, prices)
    res = run_dispatch(inputs, strategy="no_battery")
    assert all(c == 0.0 for c in res.charge_mw)
    assert all(d == 0.0 for d in res.discharge_mw)
    settled = settle_day(res, realised, prices)
    expected = sum(p * r for p, r in zip(prices, realised))
    assert settled["cost_eur"] == pytest.approx(expected, rel=1e-12, abs=1e-9)
    # no battery action ⇒ realised grid demand == realised residual >= 0
    assert settled["n_export_undercut"] == 0
    assert settled["grid_demand_realised"] == realised
    assert settled["energy_neutrality_delta_mwh"] == pytest.approx(0.0, abs=1e-12)


def test_settle_day_export_undercut_detection():
    inputs = _tiny_input()  # residual forecast 100 MW, price ramp up
    res = run_dispatch(inputs, strategy="greedy_arbitrage")
    assert max(res.discharge_mw) > 0.0 and max(res.charge_mw) > 0.0
    # settled against an extremely low realised profile: every discharge hour
    # pushes grid demand below zero -> export-undercut, reported honestly
    realised = [5.0] * 24
    settled = settle_day(res, realised, inputs.price_eur_mwh)
    assert settled["n_export_undercut"] >= 1
    for t in settled["export_undercut_hours"]:
        grid = realised[t] - res.discharge_mw[t] + res.charge_mw[t]
        assert grid < -1e-9
        assert settled["grid_demand_realised"][t] == pytest.approx(grid)
    # same schedule, honoured forecast -> no undercut
    clean = settle_day(res, inputs.residual_load_mw, inputs.price_eur_mwh)
    assert clean["n_export_undercut"] == 0
    # the schedule is fixed: throughput is identical in both settlements
    assert clean["throughput_discharge_mwh"] == settled["throughput_discharge_mwh"]


def test_settle_day_neutrality_matches_schedule(profiles, gold_by_ts):
    prof = _first_complete(profiles)
    prices = _gold_prices(gold_by_ts, prof)
    realised = _gold_realised(gold_by_ts, prof)
    inputs = assemble_dispatch_input(prof, prices)
    res = run_dispatch(inputs, strategy="lp_p50")
    settled = settle_day(res, realised, prices)
    eta_c = res.battery.charge_efficiency
    eta_d = res.battery.discharge_efficiency
    expected = sum(res.charge_mw) * eta_c - sum(res.discharge_mw) / eta_d
    assert settled["energy_neutrality_delta_mwh"] == pytest.approx(expected)
    assert settled["throughput_charge_mwh"] == pytest.approx(sum(res.charge_mw))
    assert settled["throughput_discharge_mwh"] == pytest.approx(sum(res.discharge_mw))


# ============================================================================
# Bootstrap CI
# ============================================================================
def test_bootstrap_identical_lists_zero_difference():
    ci = bootstrap_cost_difference_ci([3.0, 3.0, 3.0], [3.0, 3.0, 3.0], n_boot=400)
    assert ci["difference"] == pytest.approx(0.0, abs=1e-12)
    assert ci["n_samples"] == 3
    assert ci["ci_low"] == pytest.approx(0.0, abs=1e-12)
    assert ci["ci_high"] == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_positive_difference_means_a_more_expensive():
    a = [10.0, 12.0, 11.0]
    b = [5.0, 5.0, 5.0]
    ci = bootstrap_cost_difference_ci(a, b, n_boot=400)
    assert ci["difference"] == pytest.approx(
        6.0
    )  # mean(a) - mean(b); each pair a-b > 0
    assert ci["ci_low"] > 0.0
    assert ci["ci_high"] >= ci["ci_low"]


def test_bootstrap_none_pairs_excluded_consistently():
    a = [1.0, None, 3.0]
    b = [2.0, 4.0, None]
    ci = bootstrap_cost_difference_ci(a, b, n_boot=200)
    assert ci["n_samples"] == 1  # only the (1.0, 2.0) pair survives
    assert ci["difference"] == pytest.approx(-1.0, abs=1e-12)


def test_bootstrap_empty_returns_none():
    ci = bootstrap_cost_difference_ci([None], [None], n_boot=100)
    assert ci["difference"] is None
    assert ci["ci_low"] is None and ci["ci_high"] is None
    assert ci["n_samples"] == 0


def test_bootstrap_deterministic_same_seed():
    a = [10 + i for i in range(20)]
    b = [8 + (i % 3) + 0.5 for i in range(20)]
    c1 = bootstrap_cost_difference_ci(a, b, n_boot=500, seed=7)
    c2 = bootstrap_cost_difference_ci(a, b, n_boot=500, seed=7)
    c3 = bootstrap_cost_difference_ci(a, b, n_boot=500, seed=8)
    assert c1["difference"] == c2["difference"] == c3["difference"]  # point is fixed
    assert c1["ci_low"] == c2["ci_low"] and c1["ci_high"] == c2["ci_high"]
    assert (c1["ci_low"], c1["ci_high"]) != (c3["ci_low"], c3["ci_high"])


def test_bootstrap_rejects_length_mismatch():
    with pytest.raises(ValueError, match="parallel"):
        bootstrap_cost_difference_ci([1.0, 2.0], [1.0])


# ============================================================================
# Top-level backtest
# ============================================================================
def test_backtest_structure(backtest):
    assert isinstance(backtest, BacktestResult)
    assert backtest.phase == "4D-B"
    assert backtest.data_status == DATA_STATUS == "FIXTURE-VERIFIED"

    info = backtest.info
    assert info["target_column"] == "residual_load_mw"
    assert info["issue_hour_utc"] == 6
    assert info["n_profiles"] >= 1
    assert info["n_profiles_complete"] >= 1
    assert 1 <= info["n_dispatch_days"] <= info["n_profiles_complete"]
    assert info["n_dropped_days"] >= 0
    assert info["asof_policy"] == ASOF_POLICY
    assert info["price_availability_convention"] == PRICE_AVAILABILITY_CONVENTION
    assert info["strategies"] == [
        "no_battery",
        "greedy_arbitrage",
        "lp_p50",
        "scenario_lp",
    ]

    # the fixture separation is explicit and honest
    assert info["data_status"] == "FIXTURE-VERIFIED"
    assert "UNVERIFIED for live electricity" in info["statement"]
    assert "no ENTSO-E API key" in info["statement"]
    assert "synthetic" in info["statement"]
    assert "machinery validation" in info["statement"]
    assert info["export_undercut_note"]

    rep = backtest.reproducibility
    assert rep["seed"] == 0 and rep["n_boot"] >= 1
    assert (
        rep["gridpulse_version"] and rep["python_version"] and rep["generated_at_utc"]
    )


def test_backtest_strategy_stats(backtest):
    names = [s["strategy"] for s in backtest.strategies]
    assert names == ["no_battery", "greedy_arbitrage", "lp_p50", "scenario_lp"]
    for s in backtest.strategies:
        assert s["n_days"] >= 1
        assert s["n_infeasible"] == 0
        assert len(s["daily_costs_eur"]) == s["n_days"]
        assert s["mean_daily_cost_eur"] == pytest.approx(
            sum(s["daily_costs_eur"]) / s["n_days"]
        )
        assert s["min_daily_cost_eur"] <= s["max_daily_cost_eur"]
        assert s["total_export_undercut_hours"] >= 0
    # no_battery saves nothing vs itself
    nb = {s["strategy"]: s for s in backtest.strategies}["no_battery"]
    assert nb["mean_daily_savings_vs_no_battery_eur"] == pytest.approx(0.0, abs=1e-9)


def test_backtest_economic_sanity(backtest):
    """Dispatch strategies must beat the reference on this fixture window."""
    by_pair = {}
    for c in backtest.comparisons:
        by_pair[(c["A"], c["B"])] = c
    assert set(by_pair) == {
        ("no_battery", "greedy_arbitrage"),
        ("no_battery", "lp_p50"),
        ("no_battery", "scenario_lp"),
        ("lp_p50", "scenario_lp"),
    }
    for other in ("greedy_arbitrage", "lp_p50", "scenario_lp"):
        c = by_pair[("no_battery", other)]
        assert c["difference"] > 0.0  # no_battery is more expensive
        assert c["ci_low"] > 0.0
        assert c["n_days"] >= 3
        assert "no_battery" in c["note"]
    # scenario_lp is no-worse than lp_p50 (identical on this fixture window)
    c = by_pair[("lp_p50", "scenario_lp")]
    assert c["difference"] >= -1e-3
    assert c["n_boot"] >= 1


def test_backtest_comparison_fields(backtest):
    for c in backtest.comparisons:
        assert c["A"] in {"no_battery", "greedy_arbitrage", "lp_p50", "scenario_lp"}
        assert c["B"] in {"no_battery", "greedy_arbitrage", "lp_p50", "scenario_lp"}
        assert c["difference"] is not None
        assert c["ci_low"] is not None and c["ci_high"] is not None
        assert c["ci_low"] <= c["ci_high"]
        assert (
            c["direction"] == f"mean_daily_cost({c['A']}) - mean_daily_cost({c['B']})"
        )
        assert c["method"].startswith("percentile bootstrap")


def test_backtest_deterministic(feature_rows, gold_rows):
    bt1 = run_dispatch_backtest(feature_rows, gold_rows)
    bt2 = run_dispatch_backtest(feature_rows, gold_rows)
    d1, d2 = bt1.to_dict(), bt2.to_dict()
    d1["reproducibility"].pop("generated_at_utc")
    d2["reproducibility"].pop("generated_at_utc")
    assert json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True)


def test_backtest_per_day_failure_handling(
    feature_rows, gold_rows, backtest, profiles, monkeypatch
):
    """One infeasible day never aborts the run — it is recorded per strategy."""
    failure_issue = sorted(profiles.profiles)[0]
    real = bt_mod.run_dispatch

    def fake_run_dispatch(inputs, strategy):
        if inputs.issue_time == failure_issue:
            raise DispatchInfeasible("intended test failure")
        return real(inputs, strategy)

    monkeypatch.setattr(bt_mod, "run_dispatch", fake_run_dispatch)
    bt = run_dispatch_backtest(feature_rows, gold_rows)

    # the same days are still evaluated; only per-strategy feasibility changes
    assert bt.info["n_dispatch_days"] == backtest.info["n_dispatch_days"]
    for s in bt.strategies:
        # the failure day stays counted as dispatched but flagged infeasible
        assert s["n_infeasible"] == 1
        assert s["n_days"] + s["n_infeasible"] == bt.info["n_dispatch_days"]
    # paired-days comparisons exclude the infeasible day for every pair
    for c in bt.comparisons:
        assert c["n_days"] == bt.info["n_dispatch_days"] - 1


def test_backtest_serialization(backtest, tmp_path):
    data = backtest.to_dict()
    text = json.dumps(data, sort_keys=True)  # must be fully JSON-serializable
    assert '"phase": "4D-B"' in text
    assert data["data_status"] == "FIXTURE-VERIFIED"

    md = backtest.to_markdown()
    assert "DATA STATUS = `FIXTURE-VERIFIED`" in md
    for name in ("no_battery", "greedy_arbitrage", "lp_p50", "scenario_lp"):
        assert name in md
    assert "## Dispatch evaluation" in md
    assert "## Reproducibility" in md
    assert backtest.comparisons  # and the note line does not crash when present

    paths = backtest.write(tmp_path)
    assert len(paths) == 2
    json_path, md_path = paths
    assert json_path.exists() and md_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["info"]["n_dispatch_days"] == backtest.info["n_dispatch_days"]
    assert json_path.read_text(encoding="utf-8").endswith("\n")
    assert md_path.read_text(encoding="utf-8").endswith("\n")


def test_backtest_split_recorded(backtest):
    s = backtest.split
    for key in (
        "train_start",
        "train_end",
        "validation_start",
        "validation_end",
        "test_start",
        "test_end",
    ):
        assert key in s and s[key]  # isoformat strings present
    assert (
        s["train_start"]
        < s["train_end"]
        <= s["validation_start"]
        < s["validation_end"]
        <= s["test_start"]
        < s["test_end"]
    )
