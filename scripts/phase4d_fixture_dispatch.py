"""Deterministic dispatch demo over all 4 baseline strategies (Phase 4D-A).

Runs on the synthetic electricity fixture and prints a compact summary for
each strategy: simulated cost, solver status, and total energy throughput.

DATA STATUS: FIXTURE-VERIFIED (synthetic inputs, no real ENTSO-E data).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

# --- path setup (mirrors phase4c script) ----------------------------------------
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from gridpulse.optimization import (
    DispatchInput,
    DispatchResult,
    run_dispatch,
)

# --- fixture builder -----------------------------------------------------------

def _make_fixture_inputs() -> list[tuple[str, DispatchInput]]:
    """Build DispatchInput instances for each demo scenario."""
    issue_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    target_times = tuple(
        datetime(2024, 1, 1, 6, 0, tzinfo=UTC) + timedelta(hours=h)
        for h in range(24)
    )

    # Synthetic price: low->high ramp (morning solar flush -> evening peak).
    # Simple monotone ramp, hand-checkable.
    price_eur_mwh = tuple(10.0 + float(h) * 2.5 for h in range(24))

    # Synthetic residual load: flat-ish with a 2 MW afternoon dip.
    residual_base = 100.0
    residual_load_mw = tuple(
        residual_base - 2.0 if 12 <= h <= 15 else residual_base
        for h in range(24)
    )

    # Scenario spread: P10/P50/P90 = residual +-/ 10%.
    p50 = residual_load_mw
    p10 = tuple(r * 0.90 for r in p50)
    p90 = tuple(r * 1.10 for r in p50)

    common = dict(
        issue_time=issue_time,
        target_times=target_times,
        price_eur_mwh=price_eur_mwh,
    )

    return [
        ("deterministic", DispatchInput(
            **common, residual_load_mw=p50,
            scenario_residual_load_mw={},
        )),
        ("scenario", DispatchInput(
            **common, residual_load_mw=p50,
            scenario_residual_load_mw={"p10": p10, "p50": p50, "p90": p90},
        )),
    ]


# --- compact table printer ----------------------------------------------------

def _print_schedule(result: DispatchResult, inputs: DispatchInput, tag: str) -> None:
    bat = inputs.battery
    print(f"\n{'-' * 80}")
    print(f"  STRATEGY: {result.strategy}  ({tag})")
    print(f"  DATA STATUS: {result.data_status}")
    print(f"  Simulated cost: EUR{result.simulated_cost_eur:,.2f}")
    print(f"  Solver: {result.solver}")
    if result.message:
        print(f"  Message: {result.message}")

    # Per-scenario costs for scenario strategies.
    if result.scenario_costs_eur and "weighted_total" in result.scenario_costs_eur:
        print("  Scenario costs: ", end="")
        for k, v in result.scenario_costs_eur.items():
            print(f"{k}=EUR{v:,.2f}  ", end="")
        print()

    # Throughput.
    total_ch = sum(result.charge_mw)
    total_dis = sum(result.discharge_mw)
    print(f"  Throughput: charge={total_ch:.2f} MWh  discharge={total_dis:.2f} MWh")

    # 24-hour schedule (compact, hour-aligned).
    print(f"  {'Hour':>4}  {'Price':>6}  {'Load':>5}  {'Charge':>7}  {'Disch':>7}  {'SOC':>7}  {'Grid':>7}")
    for i in range(24):
        h = i
        print(
            f"  {h:>4}  {inputs.price_eur_mwh[i]:>6.1f}  "
            f"{inputs.residual_load_mw[i]:>5.1f}  "
            f"{result.charge_mw[i]:>7.2f}  {result.discharge_mw[i]:>7.2f}  "
            f"{result.soc_mwh[i]:>7.2f}  {result.grid_demand_mw[i]:>7.2f}"
        )

    # Sanity checks.
    soc_min = bat.soc_min_mwh
    soc_max = bat.soc_max_mwh
    max_ch = bat.max_charge_power_mw
    max_dis = bat.max_discharge_power_mw

    ok_bounds = all(soc_min <= s <= soc_max + 1e-6 for s in result.soc_mwh)
    ok_ch = all(0 <= c <= max_ch + 1e-6 for c in result.charge_mw)
    ok_dis = all(0 <= d <= max_dis + 1e-6 for d in result.discharge_mw)
    ok_no_neg = all(g >= -1e-6 for g in result.grid_demand_mw)

    print(f"\n  Checks: SOC_bounds={'OK' if ok_bounds else 'FAIL'}  "
          f"charge_cap={'OK' if ok_ch else 'FAIL'}  "
          f"discharge_cap={'OK' if ok_dis else 'FAIL'}  "
          f"no_negative_grid={'OK' if ok_no_neg else 'FAIL'}")


# --- main ----------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("  PHASE 4D-A: BATTERY DISPATCH OPTIMISATION DEMO")
    print("  DATA STATUS: FIXTURE-VERIFIED (synthetic inputs)")
    print("  P10/P90 spread: synthetic +/-10% uncertainty band")
    print("  This is a RESEARCH SIMULATION, not grid control.")
    print("=" * 80)

    inputs_list = _make_fixture_inputs()
    results_summary: dict[str, dict] = {}

    for tag, inputs in inputs_list:
        bat = inputs.battery
        print(f"\n{'#' * 80}")
        print(f"  INPUT: tag={tag}  battery={bat.capacity_mwh} MWh / "
              f"{bat.max_charge_power_mw} MW / eta_rt={bat.round_trip_efficiency}")
        print(f"  Horizon: {inputs.target_times[0]} -> {inputs.target_times[-1]}")
        print(f"  Price range: EUR{min(inputs.price_eur_mwh):.1f} - EUR{max(inputs.price_eur_mwh):.1f}")

        strategies_for_run: list[str] = ["no_battery", "greedy_arbitrage", "lp_p50"]
        if inputs.has_scenarios:
            strategies_for_run.append("scenario_lp")

        for strat in strategies_for_run:
            try:
                result = run_dispatch(inputs, strategy=strat)
                _print_schedule(result, inputs, tag)
                results_summary[f"{tag}/{strat}"] = {
                    "cost": round(result.simulated_cost_eur, 2),
                    "status": result.status,
                    "throughput_charge": round(sum(result.charge_mw), 2),
                    "throughput_discharge": round(sum(result.discharge_mw), 2),
                }
            except Exception as e:
                print(f"\n  [{strat}] FAILED: {e}")

    # --- summary table ---------------------------------------------------------
    print(f"\n{'#' * 80}")
    print("  SUMMARY")
    print(f"{'#' * 80}")
    print(f"  {'Strategy':<25}  {'Cost (EUR)':>10}  {'Charge':>8}  {'Disch':>8}  {'Status'}")
    print(f"  {'-' * 25}  {'-' * 10}  {'-' * 8}  {'-' * 8}  {'-' * 10}")
    for key, val in results_summary.items():
        print(f"  {key:<25}  {val['cost']:>10,.2f}  {val['throughput_charge']:>8.2f}  "
              f"{val['throughput_discharge']:>8.2f}  {val['status']}")

    print(f"\n{'=' * 80}")
    print("  DEMO COMPLETE. All results are FIXTURE-VERIFIED.")
    print("  P10/P90 spread is synthetic (+/-10% of P50 residual load).")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
