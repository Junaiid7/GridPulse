"""Dispatch strategy tests (Phase 4D-A).

Tests the four baseline strategies (no_battery, greedy_arbitrage, lp_p50,
scenario_lp) across physical constraints, determinism, infeasibility, and
behavioral scenarios.
"""

import pytest

from gridpulse.optimization import (
    STRATEGIES,
    BatteryConfig,
    DispatchInfeasible,
    DispatchInput,
    run_dispatch,
)
from gridpulse.optimization.battery import BatteryModel
from tests.support.dispatch_fixture import synthetic_dispatch_input, tiny_pattern


def _make_input(**overrides):
    """Helper: create DispatchInput from tiny_pattern defaults."""
    base = tiny_pattern(24)
    base.update(overrides)
    return DispatchInput(
        issue_time=base["issue_time"],
        target_times=base["target_times"],
        price_eur_mwh=base["price_eur_mwh"],
        residual_load_mw=base["residual_load_mw"],
        battery=overrides.get("battery", BatteryConfig()),
    )


# -- Tests 1–3: SOC bounds / charge cap / discharge cap -----------------------


@pytest.mark.parametrize("strategy", ["greedy_arbitrage", "lp_p50"])
def test_soc_bounds_respected(strategy):
    """SOC stays within [soc_min, soc_max] for all hours."""
    inputs = synthetic_dispatch_input()
    result = run_dispatch(inputs, strategy=strategy)
    bat = inputs.battery

    for i, soc in enumerate(result.soc_mwh):
        assert bat.soc_min_mwh <= soc <= bat.soc_max_mwh, (
            f"hour {i} SOC {soc} outside [{bat.soc_min_mwh}, {bat.soc_max_mwh}]"
        )


@pytest.mark.parametrize("strategy", ["greedy_arbitrage", "lp_p50"])
def test_charge_power_cap_respected(strategy):
    """Charge never exceeds max_charge_power_mw."""
    inputs = synthetic_dispatch_input()
    result = run_dispatch(inputs, strategy=strategy)
    max_ch = inputs.battery.max_charge_power_mw

    for i, ch in enumerate(result.charge_mw):
        assert 0.0 <= ch <= max_ch + 1e-6, f"hour {i} charge {ch} exceeds cap {max_ch}"


@pytest.mark.parametrize("strategy", ["greedy_arbitrage", "lp_p50"])
def test_discharge_power_cap_respected(strategy):
    """Discharge never exceeds max_discharge_power_mw."""
    inputs = synthetic_dispatch_input()
    result = run_dispatch(inputs, strategy=strategy)
    max_dis = inputs.battery.max_discharge_power_mw

    for i, dis in enumerate(result.discharge_mw):
        assert 0.0 <= dis <= max_dis + 1e-6, (
            f"hour {i} discharge {dis} exceeds cap {max_dis}"
        )


# -- Test 4: Initial SOC respected -------------------------------------------


@pytest.mark.parametrize("strategy", ["greedy_arbitrage", "lp_p50"])
def test_initial_soc_respected(strategy):
    """End-of-hour-0 SOC matches the transition equation starting from initial SOC."""
    inputs = synthetic_dispatch_input()
    result = run_dispatch(inputs, strategy=strategy)
    bat = inputs.battery

    eta_c = bat.charge_efficiency
    eta_d = bat.discharge_efficiency
    ch0 = result.charge_mw[0]
    dis0 = result.discharge_mw[0]
    expected_soc0 = bat.initial_soc_mwh + ch0 * eta_c - dis0 / eta_d

    assert result.soc_mwh[0] == pytest.approx(expected_soc0, rel=1e-6)


# -- Test 5: Transition recomputed == returned SOC ---------------------------


@pytest.mark.parametrize("strategy", ["greedy_arbitrage", "lp_p50"])
def test_soc_transition_recomputed_matches_returned(strategy):
    """BatteryModel.state_vector recomputes the same SOC trajectory."""
    inputs = synthetic_dispatch_input()
    result = run_dispatch(inputs, strategy=strategy)

    model = BatteryModel(inputs.battery)
    recomputed = model.state_vector(
        result.charge_mw,
        result.discharge_mw,
        start_soc_mwh=inputs.battery.initial_soc_mwh,
    )

    for i, (expected, actual) in enumerate(zip(recomputed, result.soc_mwh)):
        assert expected == pytest.approx(actual, rel=1e-6), f"hour {i} mismatch"


# -- Tests 6–9: Input validation errors --------------------------------------


def test_empty_target_times_rejected():
    """Empty target_times raises ValueError."""
    with pytest.raises(ValueError, match="must contain exactly 24"):
        DispatchInput(
            issue_time=tiny_pattern()["issue_time"],
            target_times=[],
            price_eur_mwh=tuple(50.0 for _ in range(24)),
            residual_load_mw=tuple(100.0 for _ in range(24)),
        )


def test_23_hour_horizon_rejected():
    """23-hour horizon raises ValueError."""
    base = tiny_pattern(23)
    with pytest.raises(ValueError, match="must contain exactly 24"):
        DispatchInput(
            issue_time=base["issue_time"],
            target_times=base["target_times"],
            price_eur_mwh=base["price_eur_mwh"],
            residual_load_mw=base["residual_load_mw"],
        )


def test_non_finite_price_rejected():
    """Non-finite price raises ValueError."""
    base = tiny_pattern(24)
    prices = list(base["price_eur_mwh"])
    prices[5] = float("inf")
    with pytest.raises(ValueError, match="price_eur_mwh must contain only finite"):
        DispatchInput(
            issue_time=base["issue_time"],
            target_times=base["target_times"],
            price_eur_mwh=tuple(prices),
            residual_load_mw=base["residual_load_mw"],
        )


def test_scenario_ordering_violation_rejected():
    """P10 > P50 ordering violation raises ValueError."""
    inputs = synthetic_dispatch_input()
    scenarios = dict(inputs.scenario_residual_load_mw)
    bad_p10 = list(scenarios["p10"])
    bad_p10[0] = scenarios["p50"][0] + 10.0  # violate ordering

    with pytest.raises(ValueError, match="scenario ordering violated"):
        DispatchInput(
            issue_time=inputs.issue_time,
            target_times=inputs.target_times,
            price_eur_mwh=inputs.price_eur_mwh,
            residual_load_mw=inputs.residual_load_mw,
            scenario_residual_load_mw={
                "p10": tuple(bad_p10),
                "p50": scenarios["p50"],
                "p90": scenarios["p90"],
            },
        )


# -- Test 10: Determinism ---------------------------------------------------


@pytest.mark.parametrize(
    "strategy", ["no_battery", "greedy_arbitrage", "lp_p50", "scenario_lp"]
)
def test_determinism(strategy):
    """Running the same strategy twice produces identical to_dict() output."""
    inputs = synthetic_dispatch_input()
    if strategy == "scenario_lp" and not inputs.has_scenarios:
        pytest.skip("scenario_lp requires scenarios")

    r1 = run_dispatch(inputs, strategy=strategy)
    r2 = run_dispatch(inputs, strategy=strategy)

    assert r1.to_dict() == r2.to_dict()


# -- Test 11: No-battery baseline -------------------------------------------


def test_no_battery_zero_action_and_cost():
    """no_battery strategy: zero charge/discharge, cost = sum(price * residual)."""
    inputs = synthetic_dispatch_input()
    result = run_dispatch(inputs, strategy="no_battery")

    assert all(ch == 0.0 for ch in result.charge_mw)
    assert all(dis == 0.0 for dis in result.discharge_mw)
    assert all(
        soc == pytest.approx(inputs.battery.initial_soc_mwh) for soc in result.soc_mwh
    )

    expected_cost = sum(
        p * r for p, r in zip(inputs.price_eur_mwh, inputs.residual_load_mw)
    )
    assert result.simulated_cost_eur == pytest.approx(expected_cost)


# -- Test 12: Greedy respects constraints ------------------------------------


def test_greedy_respects_physical_constraints():
    """Greedy heuristic respects power/SOC/no-export constraints."""
    # Low→high price ramp, constant residual 100 MW.
    inputs = _make_input()
    result = run_dispatch(inputs, strategy="greedy_arbitrage")
    bat = inputs.battery

    # Power caps.
    for ch in result.charge_mw:
        assert 0 <= ch <= bat.max_charge_power_mw
    for dis in result.discharge_mw:
        assert 0 <= dis <= bat.max_discharge_power_mw

    # SOC bounds.
    for soc in result.soc_mwh:
        assert bat.soc_min_mwh <= soc <= bat.soc_max_mwh

    # No-export: grid demand >= 0.
    for gd, r in zip(result.grid_demand_mw, inputs.residual_load_mw):
        assert gd >= -1e-6, f"grid demand {gd} < 0 (no-export violated)"


# -- Test 13: LP satisfies all constraints ----------------------------------


def test_lp_satisfies_all_constraints():
    """LP solution respects SOC transition, bounds, and no-export constraints."""
    inputs = synthetic_dispatch_input()
    result = run_dispatch(inputs, strategy="lp_p50")
    bat = inputs.battery

    # SOC transition equation.
    model = BatteryModel(bat)
    soc_expected = model.state_vector(
        result.charge_mw, result.discharge_mw, start_soc_mwh=bat.initial_soc_mwh
    )
    for i, (actual, expected) in enumerate(zip(result.soc_mwh, soc_expected)):
        assert actual == pytest.approx(expected, rel=1e-5), f"SOC mismatch at {i}"

    # SOC bounds.
    for i, soc in enumerate(result.soc_mwh):
        assert bat.soc_min_mwh - 1e-6 <= soc <= bat.soc_max_mwh + 1e-6

    # Power caps.
    for ch in result.charge_mw:
        assert 0 <= ch <= bat.max_charge_power_mw + 1e-6
    for dis in result.discharge_mw:
        assert 0 <= dis <= bat.max_discharge_power_mw + 1e-6

    # No-export (P50).
    for t, (gd, r) in enumerate(zip(result.grid_demand_mw, inputs.residual_load_mw)):
        expected_gd = r - result.discharge_mw[t] + result.charge_mw[t]
        assert gd == pytest.approx(expected_gd, rel=1e-5)
        assert gd >= -1e-6, f"grid demand {gd} < 0 at hour {t}"


# -- Test 14: Scenario LP per-scenario costs -------------------------------


def test_scenario_lp_per_scenario_costs():
    """Scenario LP reports per-scenario costs and weighted total."""
    inputs = synthetic_dispatch_input()
    if not inputs.has_scenarios:
        pytest.skip("requires scenario_residual_load_mw")

    result = run_dispatch(inputs, strategy="scenario_lp")

    assert "p10" in result.scenario_costs_eur
    assert "p50" in result.scenario_costs_eur
    assert "p90" in result.scenario_costs_eur
    assert "weighted_total" in result.scenario_costs_eur

    # Weighted total should equal simulated_cost_eur.
    assert result.scenario_costs_eur["weighted_total"] == pytest.approx(
        result.simulated_cost_eur, rel=1e-5
    )


# -- Test 16: Low→high ramp behavior ----------------------------------------


@pytest.mark.parametrize("strategy", ["greedy_arbitrage", "lp_p50"])
def test_low_high_ramp_charges_cheap_discharges_expensive(strategy):
    """On a low→high price ramp, strategies net-charge early, net-discharge late."""
    # Price ramp: 10 → 33 EUR/MWh, constant residual 100 MW.
    inputs = _make_input()
    result = run_dispatch(inputs, strategy=strategy)

    # Early hours (cheap): expect net charge (ch > dis).
    early_charge = sum(result.charge_mw[:8])
    early_discharge = sum(result.discharge_mw[:8])

    # Late hours (expensive): expect net discharge (dis > ch).
    late_charge = sum(result.charge_mw[16:])
    late_discharge = sum(result.discharge_mw[16:])

    assert early_charge > early_discharge + 1e-3, (
        "expected net charge in early cheap hours"
    )
    assert late_discharge > late_charge + 1e-3, (
        "expected net discharge in late expensive hours"
    )


# -- Test 17: Infeasibility -------------------------------------------------


def test_infeasible_terminal_soc_raises():
    """Physically-unreachable terminal SOC raises DispatchInfeasible (not silent)."""
    # Battery: 50 MWh, SOC 10–90%, initial 50% = 25 MWh, max charge 0.5 MW.
    # Max SOC reachable in 24 h ≈ 25 + 24·0.5·sqrt(0.9) ≈ 36.4 MWh, but the
    # forced terminal 90% = 45 MWh. Unreachable → LP infeasible after 24 h.
    bat = BatteryConfig(
        capacity_mwh=50.0,
        max_charge_power_mw=0.5,
        max_discharge_power_mw=10.0,
        soc_min=0.10,
        soc_max=0.90,
        initial_soc=0.50,
    )
    inputs = synthetic_dispatch_input(battery=bat)

    inputs = DispatchInput(
        issue_time=inputs.issue_time,
        target_times=inputs.target_times,
        price_eur_mwh=inputs.price_eur_mwh,
        residual_load_mw=inputs.residual_load_mw,
        battery=bat,
        terminal_soc=0.90,
    )

    with pytest.raises(DispatchInfeasible):
        run_dispatch(inputs, strategy="lp_p50")


# -- Test 18: Grid demand >= 0 (no-export default) --------------------------


@pytest.mark.parametrize("strategy", ["greedy_arbitrage", "lp_p50", "scenario_lp"])
def test_grid_demand_non_negative(strategy):
    """Default no-export constraint ensures grid demand >= 0 in all hours."""
    inputs = synthetic_dispatch_input()
    if strategy == "scenario_lp" and not inputs.has_scenarios:
        pytest.skip("scenario_lp requires scenarios")

    result = run_dispatch(inputs, strategy=strategy)

    for t, gd in enumerate(result.grid_demand_mw):
        assert gd >= -1e-6, f"grid demand {gd} < 0 at hour {t} (no-export violated)"


# -- No-simultaneity check --------------------------------------------------


@pytest.mark.parametrize("strategy", ["greedy_arbitrage", "lp_p50", "scenario_lp"])
def test_no_simultaneous_charge_and_discharge(strategy):
    """No hour has both charge > 0 and discharge > 0 (within tolerance)."""
    inputs = synthetic_dispatch_input()
    if strategy == "scenario_lp" and not inputs.has_scenarios:
        pytest.skip("scenario_lp requires scenarios")

    result = run_dispatch(inputs, strategy=strategy)
    tol = 1e-6

    for t, (ch, dis) in enumerate(zip(result.charge_mw, result.discharge_mw)):
        simultaneous = ch > tol and dis > tol
        assert not simultaneous, (
            f"hour {t} has simultaneous charge={ch} and discharge={dis}"
        )


# -- Strategy registry ------------------------------------------------------


def test_strategies_registry():
    """STRATEGIES contains the four expected strategy names."""
    expected = {"no_battery", "greedy_arbitrage", "lp_p50", "scenario_lp"}
    assert STRATEGIES == expected


def test_invalid_strategy_raises():
    """run_dispatch raises ValueError for unknown strategy."""
    inputs = synthetic_dispatch_input()
    with pytest.raises(ValueError, match="unknown strategy"):
        run_dispatch(inputs, strategy="invalid_strategy")
