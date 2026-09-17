# Battery Dispatch Optimisation (Phase 4D-A)

This is a **research/simulation** optimiser. Given a 24 h-ahead residual-load
profile and day-ahead electricity prices, it produces **hypothetical** battery
charge/discharge schedules. It does **not** control a real battery, grid or
market, and must never be read as such.

Everything here is **FIXTURE-VERIFIED** — it runs on hand-crafted synthetic
electricity fixtures in the offline suite. No live ENTSO-E data is used or
claimed (see `real-data-run.md` for Phase 4A's honest report). The P10/P90
scenario spread in the demo/tests is a **synthetic ±10% band**, not a real
forecast model.

## Research purpose

The day-ahead price and today's residual-load forecast are both uncertain at
dispatch time. Phase 4D-A builds the low-level motion (a battery model), the
optimisation machinery (LP + heuristics), and four comparable baselines so that
later phases (4D-B backtesting) can answer: *does uncertainty-aware dispatch
beat point-forecast dispatch in expectation, and what is it worth?*

## Battery assumptions (defaults)

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Capacity | `capacity_mwh` | 50 MWh |
| Max charge power | `max_charge_power_mw` | 10 MW |
| Max discharge power | `max_discharge_power_mw` | 10 MW |
| Round-trip efficiency | `round_trip_efficiency` | 0.90 |
| SOC bounds | `soc_min` / `soc_max` | 0.10 / 0.90 (fractions) |
| Initial SOC | `initial_soc` | 0.50 (fraction) |
| Charge / discharge efficiency | `charge_efficiency` / `discharge_efficiency` | `sqrt(0.90)` each |

The round-trip loss is split **evenly** between the two directions by default:
`charge_efficiency * discharge_efficiency == round_trip_efficiency` (with
`charge_efficiency = discharge_efficiency = sqrt(eta_rt)`). Callers may override,
but the product must equal the configured round-trip efficiency (validated with
tolerance `ETA_TOL = 1e-9`).

Powers are in MW over a one-hour period, so `power (MW) · 1 h = energy (MWh)`.

## Sign conventions

- **Positive residual load** = demand exceeds wind/solar generation.
- **Battery discharge** reduces net grid demand (it serves load).
- **Battery charge** increases net grid demand (it consumes power).
- Net grid demand per hour: `g[t] = residual_load[t] − discharge[t] + charge[t]`.

## SOC equation

Exact, one linear equation per hour:

```
SOC[t+1] = SOC[t] + charge[t] * eta_c − discharge[t] / eta_d
```

`discharge[t] / eta_d` is the *withdrawal from the battery* required to deliver
`discharge[t]` MW of grid power: because `eta_d < 1`, one grid-MWh of discharge
consumes more than one battery-MWh. `BatteryModel.state_vector(ch, dis)`
implements exactly this and is used by both the greedy heuristic and the LP for
deterministic verification.

## Constraints

- **Power limits**: `0 ≤ charge[t] ≤ P_charge`, `0 ≤ discharge[t] ≤ P_discharge`.
- **SOC limits**: `soc_min_mwh ≤ SOC[t] ≤ soc_max_mwh`.
- **Transition** (equality, one per hour): the SOC equation above.
- **No-export (default)**: `charge[t] − discharge[t] ≤ residual_load[t]`, so
  `g[t] ≥ 0` in every hour — the battery never pushes net demand below zero.
  `curtailment_allowed=True` instead documents an explicit export interpretation.
- **Terminal SOC** (default = initial SOC, a net-neutral horizon):
  `SOC[23] = terminal_soc_mwh`. This keeps the simulated 24 h energy-neutral.

## Objective

Minimise the simulated energy cost

```
Σ_t price[t] · g[t],   g[t] = residual[t] − discharge[t] + charge[t]
```

The constant term (`price[t] · residual[t]`) does not depend on the decision, so
the LP coefficients are `+price[t]` on `charge[t]` and `−price[t]` on
`discharge[t]`. With scenario weights summing to 1 the expected-cost coefficients
are identical (see the scenario LP below).

## The four baseline strategies

1. **`no_battery`** — zero action. Cost = `Σ price[t] · residual[t]`. The
   reference "do nothing" baseline.
2. **`greedy_arbitrage`** — a deterministic price-median two-phase heuristic.
   Hours below the horizon median are *charge candidates* (cheapest first);
   above-median hours are *discharge candidates* (most expensive first). Within
   each group, action is allocated proportional to distance from the median,
   capped by power, SOC headroom/floor, and (under no-export) the residual load
   of the hour. A reconciliation pass raises terminal SOC toward
   `inputs.terminal_soc` where the *replayed* trajectory has real headroom.
   **Reference only, not optimal.**
3. **`lp_p50`** — forecast-driven **linear program** using the P50 residual-load
   profile via `scipy.optimize.linprog(method="highs")` (deterministic HiGHS).
4. **`scenario_lp`** — the **uncertainty-aware** variant. Uses P10/P50/P90
   scenario loads with weights `(0.25, 0.50, 0.25)` (the Phase 4C convention).
   **One SHARED charge/discharge/SOC decision** is required to be feasible for
   every scenario; the no-export constraint is enforced per-scenario. The
   objective is the weighted expected cost `Σ_s w_s Σ_t price[t] · g_s[t]`.
   `DispatchResult.scenario_costs_eur` reports each per-scenario cost. This is
   *uncertainty-aware scenario dispatch*, not financial-risk optimisation — and
   the scenarios are residual-load scenarios only, never silently treated as
   price scenarios.

`run_dispatch(inputs, strategy)` is the public facade over all four;
`STRATEGIES = {"no_battery", "greedy_arbitrage", "lp_p50", "scenario_lp"}`.

All strategies are **deterministic**: HiGHS is deterministic on fixed input, and
the greedy's step ordering is chronological/price-sorted with no randomness.

### No simultaneous charge and discharge

Charge and discharge are separate LP variables (no MIP). For any feasible
solution with `charge[t] > 0` **and** `discharge[t] > 0` in an hour, reducing
both by the same MW leaves the cost unchanged (the two net out) and strictly
relaxes SOC headroom, so an optimal zero-simultaneity solution always exists for
finite prices — HiGHS (vertex solver) returns one. `DispatchResult` includes the
verification; any hour that triggers both raises rather than silently shipping
nonsense.

## Input / output contracts

Construct a `DispatchInput` (in `gridpulse.optimization.contract`):

- exactly 24 hourly, strictly-increasing, aware-UTC `target_times`;
- 24 finite, non-negative `price_eur_mwh`;
- 24 finite `residual_load_mw` (the P50 / system profile);
- optional all-or-none `scenario_residual_load_mw` with keys `p10/p50/p90`,
  per-hour `p10 ≤ p50 ≤ p90` (ordering enforced at the boundary, matching Phase
  4C), and `scenario["p50"] == residual_load_mw`;
- `battery` config (defaults to the 50 MWh / 10 MW above);
- `curtailment_allowed` (default `False`); `terminal_soc` (default
  `battery.initial_soc`).

Everything invalid is **rejected with `ValueError`**, never silently repaired,
and missing/`NaN` values are never fabricated.

`DispatchResult` carries the schedule vectors (`charge_mw`, `discharge_mw`,
`soc_mwh` — SOC at the *end* of each hour), the headline `grid_demand_mw`,
`simulated_cost_eur`, `scenario_costs_eur`, the battery config, solver status
(`"optimal"` / `"ok"`), solver name and message, and `to_dict()` — the
machine-readable form consumed by Phase 4D-B backtesting.

On solver failure (e.g. an unreachable terminal SOC) the LP raises
`DispatchInfeasible(ValueError)` carrying the solver message — failures are never
silent.

## What P10/P50/P90 mean

Consistent with Phase 4C: `p10` is the *10th percentile* of the day / residual
load (low-load scenario), `p90` the *90th* (high-load scenario), `p50` the median
(equal to the deterministic profile). Scenario weights favour the median
(0.50) with symmetric tails (0.25 each) per the Phase 4C convention.

## Why simulation, not control

The dispatcher is a research object: it **recommends** a hypothetical schedule
under the assumptions above (linear efficiencies, no cycling/soak degradation,
perfect 1 h resolution, no reserve-market participation, no grid-congestion
constraints beyond no-export). Real batteries face nonlinear degradation, ramp
limits, and market rules this model deliberately omits. Any live application
would need its own engineering review and hardware-in-the-loop validation.

## Relationship to Phase 4C

Phase 4C produces P10/P50/P90 **quantile forecasts** of residual load
(`QuantileForecast`, ordering enforced). Phase 4D-A *consumes* that shape: the
P50 profile drives `lp_p50`, and the P10/P50/P90 triple drives `scenario_lp`
through the same ordering rules. In production the scenario profiles would come
from Phase 4C's predictive distribution; offline, `tests/support/dispatch_fixture.py`
builds an explicitly synthetic ±10% band so the suite is FIXTURE-VERIFIED.

Phase 4D-B consumes `DispatchResult.to_dict()` row-by-row to run historical
backtests across strategies — the per-offset forecast-profile harness,
settlement semantics and fixture-vs-live separation are documented in
`docs/backtest.md`.

## Repository layout

| File | Role |
|------|------|
| `src/gridpulse/optimization/__init__.py` | public exports + entry-point doc |
| `src/gridpulse/optimization/battery.py` | `BatteryConfig`, `BatteryModel`, validation |
| `src/gridpulse/optimization/contract.py` | `DispatchInput`, `DispatchResult`, `DispatchInfeasible` |
| `src/gridpulse/optimization/strategies.py` | LP builder, greedy, the four strategies, `run_dispatch` |
| `tests/support/dispatch_fixture.py` | synthetic dispatch inputs (FIXTURE-VERIFIED) |
| `tests/unit/optimization/*.py` | battery / contract / strategy tests |
| `scripts/phase4d_fixture_dispatch.py` | deterministic demo over all four strategies |