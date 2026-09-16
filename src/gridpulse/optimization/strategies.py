r"""Battery-dispatch strategies (Phase 4D-A).

Four deterministic baselines over a :class:`DispatchInput`:

1. ``no_battery``       — zero battery action; the residual-load cost as-is.
2. ``greedy_arbitrage`` — price-median two-phase heuristic (reference, not
   optimal): charge during relatively cheap hours, discharge during
   relatively expensive hours, subject to exactly the same battery
   constraints as the LP.
3. ``lp_p50``          — forecast-driven linear program using the P50
   residual-load profile (scipy.optimize.linprog, HiGHS).
4. ``scenario_lp``     — uncertainty-aware scenario LP using P10/P50/P90
   with weights ``SCENARIO_WEIGHTS = (0.25, 0.50, 0.25)`` (Phase 4C
   convention). One SHARED charge/discharge/SOC decision is feasible for
   every scenario and the objective is the weighted expected cost. This
   is *uncertainty-aware scenario dispatch*, not financial-risk
   optimisation, and the scenarios are residual-load scenarios only —
   never silently treated as price scenarios.

LP formulation
--------------

Variables per hour *t* (ordered ``[ch0 dis0 soc0 ch1 dis1 soc1 ...]``):

===============  ================================================
variable         meaning
===============  ================================================
``ch[t]``        charge power (MW), 0..max_charge_power
``dis[t]``       discharge power (MW), 0..max_discharge_power
``soc[t]``       SOC at the *end* of hour *t* (MWh), within bounds
===============  ================================================

Objective (minimise simulated energy cost)

.. math::

    \sum_t \mathrm{price}[t] \cdot g[t],
    \qquad g[t] = \mathrm{residual}[t] - \mathrm{dis}[t] + \mathrm{ch}[t]

After dropping the constant term (``price[t] * residual[t]`` does not depend
on the decision), the linear coefficients are ``+price[t]`` on ``ch[t]`` and
``-price[t]`` on ``dis[t]``. With scenario weights :math:`w_s` summing to 1
the expected-cost coefficients are identical to the single-scenario case; the
scenarios couple the problem through the per-scenario "no negative net grid
demand" constraints below.

Constraints

- SOC transition (equality, one per hour):

  .. math::

      \mathrm{soc}[t] - \mathrm{soc}[t-1]
        - \eta_c \,\mathrm{ch}[t] + \mathrm{dis}[t]/\eta_d = 0

  with :math:`\mathrm{soc}[-1] = \mathrm{initial\_soc\_mwh}`. Here
  :math:`\eta_c \cdot \eta_d = \eta_{rt}` (see :mod:`.battery`).
- SOC bounds: ``soc_min_mwh <= soc[t] <= soc_max_mwh``.
- Power bounds: ``0 <= ch[t] <= max_charge``, ``0 <= dis[t] <= max_discharge``.
- No negative net grid demand (when ``curtailment_allowed=False``, the
  default): for every scenario *s* and hour: ``ch[t] - dis[t]
  <= residual_s[t]``, i.e. :math:`g_s[t] \\ge 0`. When
  ``curtailment_allowed=True`` this constraint is dropped and the export
  interpretation is documented.
- Terminal SOC (default ``terminal_soc = initial_soc``, net-neutral horizon):
  ``soc[23] >= terminal_soc_mwh``.
- No-simultaneity: separate ``ch[t]``/``dis[t]`` LP variables allow
  simultaneous charge and discharge *mathematically*. We do not add
  mixed-integer complexity: for any feasible solution with both positive in
  an hour, reducing both by an equal MW keeps the objective identical and
  strictly relaxes SOC headroom, so an optimal zero-simultaneity solution
  always exists for any finite price (the constant and linear terms in
  ``price[t]`` cancel). HiGHS returns such a vertex, and the result builder
  defends against a degenerate return by verifying no hour has both ``> tol``
  and raising otherwise — simultaneity is never silently buried.
"""

from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

from scipy.optimize import linprog

from .battery import BatteryModel
from .contract import (
    HORIZON_HOURS,
    SCENARIO_KEYS,
    SCENARIO_WEIGHTS,
    DispatchInfeasible,
    DispatchInput,
    DispatchResult,
)

#: An action is effectively "off" below this many MW.
ACTION_TOL = 1e-6

STRATEGIES = frozenset({"no_battery", "greedy_arbitrage", "lp_p50", "scenario_lp"})

SOLVER_LABEL = "scipy.optimize.linprog(method='highs')"


# --------------------------------------------------------------------------
# LP builder / solver
# --------------------------------------------------------------------------
def _lp_vectors(inputs: DispatchInput, scenarios: Mapping[str, Sequence[float]], weights):
    """A_x/A/b/c for the LP.

    ``scenarios`` maps scenario key -> 24-vector of residual loads (for the
    single-scenario LP it is ``{"p50": residual}``). ``weights`` aligns with
    ``sorted(scenarios)`` and must sum to 1 (``SCENARIO_WEIGHTS`` for the
    scenario LP, ``(1.0,)`` for a single profile). Returns
    ``(c, A_ub, b_ub, A_eq, b_eq, bounds, constant_term)`` where
    ``constant_term`` is ``sum_t price[t] * sum_s w_s residual_s[t]`` (the
    part of the objective that does not depend on the decision).
    """
    n = len(inputs.target_times)
    battery = inputs.battery
    model = BatteryModel(battery)
    eta_c = model.charge_efficiency
    eta_d_inv = 1.0 / model.discharge_efficiency

    price = inputs.price_eur_mwh
    nvar = n * 3  # ch, dis, soc per hour

    c = [0.0] * nvar
    for t in range(n):
        c[3 * t + 0] = price[t]  # ch[t]
        c[3 * t + 1] = -price[t]  # dis[t]

    # equality rows: transition soc[t] - soc[t-1] - eta_c*ch[t] + (1/eta_d)*dis[t] = 0
    A_eq = [[0.0] * nvar for _ in range(n)]
    b_eq = [0.0] * n
    for t in range(n):
        row = A_eq[t]
        row[3 * t + 0] = -eta_c  # ch[t]
        row[3 * t + 1] = eta_d_inv  # dis[t]
        row[3 * t + 2] = 1.0  # soc[t]
        if t > 0:
            row[3 * (t - 1) + 2] = -1.0  # - soc[t-1]
        else:
            b_eq[t] = battery.initial_soc_mwh  # soc[0] - ... = initial

    # inequality rows
    A_ub = []
    b_ub = []

    # no-export: ch[t] - dis[t] <= residual_s[t]  for every scenario
    if not inputs.curtailment_allowed:
        for s in sorted(scenarios):
            residual = scenarios[s]
            for t in range(n):
                row = [0.0] * nvar
                row[3 * t + 0] = 1.0  # ch[t]
                row[3 * t + 1] = -1.0  # - dis[t]
                A_ub.append(row)
                b_ub.append(float(residual[t]))

    # terminal: - soc[n-1] <= - terminal_soc_mwh
    terminal_mwh = battery.capacity_mwh * inputs.terminal_soc
    row = [0.0] * nvar
    row[3 * (n - 1) + 2] = -1.0
    A_ub.append(row)
    b_ub.append(-terminal_mwh)

    bounds = [
        (0.0, battery.max_charge_power_mw),  # ch[t]
        (0.0, battery.max_discharge_power_mw),  # dis[t]
        (battery.soc_min_mwh, battery.soc_max_mwh),  # soc[t]
    ] * n

    order = sorted(scenarios)
    if abs(sum(weights) - 1.0) > 1e-9:
        raise DispatchInfeasible(f"scenario weights must sum to 1; got {sum(weights)}")
    if len(order) != len(weights):
        raise DispatchInfeasible(
            f"expected {len(weights)} scenario weight(s) for {order}; got {order}"
        )
    constant_term = sum(
        price[t] * sum(w * scenarios[s][t] for s, w in zip(order, weights))
        for t in range(n)
    )
    return c, A_ub, b_ub, A_eq, b_eq, bounds, constant_term


def _solve_lp(inputs: DispatchInput, scenarios: Mapping[str, Sequence[float]], weights=None) -> dict:
    """Build and solve the LP; raise :class:`DispatchInfeasible` on failure.

    ``weights`` aligns with ``sorted(scenarios)`` and defaults to
    ``(1.0,) * len(scenarios)``. Returns a dict with ``ch``, ``dis``, ``soc``
    (lists of floats), ``cost`` (total simulated cost including the constant
    term), ``status`` and ``message``.
    """
    if weights is None:
        weights = tuple(1.0 for _ in scenarios)
    c, A_ub, b_ub, A_eq, b_eq, bounds, constant = _lp_vectors(inputs, scenarios, weights)
    result = linprog(
        c,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if A_ub else None,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        raise DispatchInfeasible(
            "dispatch LP did not solve: "
            f"{result.message if result.message else 'unknown solver error'} "
            f"(status={result.status})"
        )
    x = [float(v) for v in result.x]
    n = len(inputs.target_times)
    ch = [x[3 * t + 0] for t in range(n)]
    dis = [x[3 * t + 1] for t in range(n)]
    soc = [x[3 * t + 2] for t in range(n)]
    # Recompute the decision-independent residual cost so scenes are uniform.
    price = inputs.price_eur_mwh
    variable_cost = sum(price[t] * (ch[t] - dis[t]) for t in range(n))
    cost = constant + variable_cost
    return {
        "ch": ch,
        "dis": dis,
        "soc": soc,
        "cost": cost,
        "status": "optimal",
        "message": result.message,
    }


# --------------------------------------------------------------------------
# NO-BATTERY baseline
# --------------------------------------------------------------------------
def _strategy_no_battery(inputs: DispatchInput) -> DispatchResult:
    ch = [0.0] * HORIZON_HOURS
    dis = [0.0] * HORIZON_HOURS
    cost = sum(p * r for p, r in zip(inputs.price_eur_mwh, inputs.residual_load_mw))
    grid = [float(r) for r in inputs.residual_load_mw]
    return _build_result(
        inputs,
        strategy="no_battery",
        ch=ch,
        dis=dis,
        status="ok",
        solver="none",
        message="no battery action by construction",
        cost=cost,
    )


# --------------------------------------------------------------------------
# GREEDY arbitrage baseline
# --------------------------------------------------------------------------
def _greedy_arbitrage(inputs: DispatchInput) -> dict:
    """Deterministic price-median two-phase greedy.

    - Hours with price below the horizon median are *charge candidates* (the
      cheaper, the higher priority); above-median hours are *discharge
      candidates*.
    - Within each group the action is allocated proportional to the distance
      from the median (cheapest charges most, most expensive discharges most),
      capped by power limits, SOC headroom/floor and, when no-export is
      enforced, the residual load of the hour.
    - A reconciliation pass raises the terminal SOC to ``inputs.terminal_soc``
      by adding charge at the cheapest remaining hours if needed.

    Deterministic by construction (no randomness; ties resolved by
    chronological order). Not optimal — it is a reference heuristic that
    satisfies all battery and grid constraints.
    """
    price = list(inputs.price_eur_mwh)
    residual = list(inputs.residual_load_mw)
    median = _median(price)
    return _greedy_core(inputs, price, residual, median)


def _median(values: Sequence[float]) -> float:
    s = sorted(values)
    m = len(s) // 2
    if len(s) % 2 == 1:
        return s[m]
    return (s[m - 1] + s[m]) / 2.0


def _greedy_core(inputs: DispatchInput, price: list, residual: list, median: float) -> dict:
    battery = inputs.battery
    model = BatteryModel(battery)
    n = len(inputs.target_times)
    eta_c = model.charge_efficiency

    ch = [0.0] * n
    dis = [0.0] * n
    soc = battery.initial_soc_mwh

    # Determine charge/discharge ordering and priority weights.
    charge_hours = [
        t for t in range(n) if price[t] <= median
    ]
    discharge_hours = [
        t for t in range(n) if price[t] > median
    ]
    # Priority: within charge group, cheaper first (smaller price); within
    # discharge group, more expensive first (larger price).
    charge_hours.sort(key=lambda t: price[t])
    discharge_hours.sort(key=lambda t: -price[t])

    # ---- Phase 1: charge candidates (cheap hours), honouring SOC headroom.
    for t in charge_hours:
        headroom = battery.soc_max_mwh - soc
        if headroom <= ACTION_TOL:
            break
        max_charge_energy = headroom / eta_c
        # Cap so we never exceed max power.
        cap = min(battery.max_charge_power_mw, max_charge_energy)
        # Weight by distance from median so cheapest get the most.
        spread = median - price[t]
        total_spread = max(median - price[charge_hours[0]], 1e-9) if charge_hours else 1e-9
        weight = spread / total_spread
        action = cap * weight if weight > 0 else 0.0
        action = min(action, battery.max_charge_power_mw, max_charge_energy)
        ch[t] = action
        soc = soc + action * eta_c

    # ---- Phase 2: discharge candidates (expensive hours), honouring SOC
    # ---- floor, power cap and (no-export) residual cap.
    for t in discharge_hours:
        floor_headroom = soc - battery.soc_min_mwh
        if floor_headroom <= ACTION_TOL:
            continue
        max_discharge_energy = floor_headroom * model.discharge_efficiency
        cap = min(battery.max_discharge_power_mw, max_discharge_energy)
        if not inputs.curtailment_allowed:
            cap = min(cap, max(0.0, residual[t]))
        spread = price[t] - median
        total_spread = max(price[discharge_hours[0]] - median, 1e-9) if discharge_hours else 1e-9
        weight = spread / total_spread
        action = cap * weight if weight > 0 else 0.0
        action = min(action, battery.max_discharge_power_mw, max_discharge_energy)
        if not inputs.curtailment_allowed:
            action = min(action, max(0.0, residual[t]))
        dis[t] = action
        soc = soc - action / model.discharge_efficiency

    # ---- Phase 3: terminal-SOC reconciliation (net-neutral horizon).
    #
    # The running ``soc`` variable above reflects energy *net of discharge*, so
    # it can be low even while the replayed trajectory ``socs`` still peaks at
    # soc_max during the charge hours. Adding charge to an hour that is already
    # at the SOC cap in trajectory-space would overcharge the battery (SOC past
    # soc_max / capacity). So headroom here is computed from the *replayed*
    # trajectory, not from the running variable.
    target = battery.capacity_mwh * inputs.terminal_soc
    socs = model.state_vector(ch, dis, start_soc_mwh=battery.initial_soc_mwh)
    if socs[-1] < target - ACTION_TOL:
        remaining = sorted(charge_hours, key=lambda t: price[t])
        for t in remaining:
            if socs[-1] >= target - ACTION_TOL:
                break
            power_room = battery.max_charge_power_mw - ch[t]
            # Adding energy at hour t raises every subsequent hour by the same
            # amount, so the binding headroom is the *minimum* gap over the
            # forward trajectory, not the gap at hour t alone.
            forward_headroom = min(
                battery.soc_max_mwh - v for v in socs[t:]
            )
            if power_room <= ACTION_TOL or forward_headroom <= ACTION_TOL:
                continue
            add_energy = min(power_room * eta_c, forward_headroom, target - socs[-1])
            if add_energy <= ACTION_TOL:
                continue
            add_power = add_energy / eta_c
            ch[t] += add_power
            # Raising the SOC at the end of hour t lifts every subsequent hour
            # by the same amount (no other action changes in between).
            for j in range(t, len(socs)):
                socs[j] += add_energy
    soc = socs[-1]

    cost = _cost_of(inputs, ch, dis)
    return {"ch": ch, "dis": dis, "soc": model.state_vector(ch, dis), "cost": cost,
            "status": "ok", "message": "deterministic greedy heuristic"}


def _cost_of(inputs: DispatchInput, ch: Sequence[float], dis: Sequence[float]) -> float:
    return sum(
        p * (r - d + c)
        for p, r, c, d in zip(
            inputs.price_eur_mwh, inputs.residual_load_mw, ch, dis
        )
    )


def _strategy_greedy(inputs: DispatchInput) -> DispatchResult:
    price = list(inputs.price_eur_mwh)
    residual = list(inputs.residual_load_mw)
    median = _median(price)
    out = _greedy_core(inputs, price, residual, median)
    return _build_result(
        inputs,
        strategy="greedy_arbitrage",
        ch=out["ch"],
        dis=out["dis"],
        status="ok",
        solver="heuristic-price-median",
        message="deterministic greedy; not optimal",
        cost=out["cost"],
    )


# --------------------------------------------------------------------------
# LP strategies
# --------------------------------------------------------------------------
def _strategy_lp_p50(inputs: DispatchInput) -> DispatchResult:
    out = _solve_lp(inputs, {"p50": inputs.residual_load_mw}, weights=(1.0,))
    _verify_no_simultaneity(out["ch"], out["dis"])
    return _build_result(
        inputs,
        strategy="lp_p50",
        ch=out["ch"],
        dis=out["dis"],
        status=out["status"],
        solver=SOLVER_LABEL,
        message=out["message"],
        cost=out["cost"],
    )


def _strategy_scenario_lp(inputs: DispatchInput) -> DispatchResult:
    if not inputs.has_scenarios:
        raise DispatchInfeasible(
            "scenario_lp requires scenario_residual_load_mw (p10/p50/p90) on the input"
        )
    scenarios = dict(inputs.scenario_residual_load_mw)
    out = _solve_lp(inputs, scenarios, weights=SCENARIO_WEIGHTS)
    _verify_no_simultaneity(out["ch"], out["dis"])
    # Per-scenario cost of the SHARED decision.
    scenario_costs = {}
    for key in SCENARIO_KEYS:
        residual = scenarios[key]
        scenario_costs[key] = sum(
            p * (r - d + c)
            for p, r, c, d in zip(inputs.price_eur_mwh, residual, out["ch"], out["dis"])
        )
    total = sum(w * scenario_costs[k] for k, w in zip(SCENARIO_KEYS, SCENARIO_WEIGHTS))
    scenario_costs["weighted_total"] = total
    cost = out["cost"]
    result = _build_result(
        inputs,
        strategy="scenario_lp",
        ch=out["ch"],
        dis=out["dis"],
        status=out["status"],
        solver=SOLVER_LABEL,
        message=out["message"],
        cost=cost,
    )
    object.__setattr__(result, "scenario_costs_eur", scenario_costs)
    return result


def _verify_no_simultaneity(ch: Sequence[float], dis: Sequence[float]) -> None:
    offenders = [t for t in range(len(ch)) if ch[t] > ACTION_TOL and dis[t] > ACTION_TOL]
    if offenders:
        raise DispatchInfeasible(
            "LP returned simultaneous charge and discharge at hours "
            f"{offenders}; refusing to return a non-physical schedule"
        )


# --------------------------------------------------------------------------
# result assembly
# --------------------------------------------------------------------------
def _build_result(
    inputs: DispatchInput,
    *,
    strategy: str,
    ch: Sequence[float],
    dis: Sequence[float],
    status: str,
    solver: str,
    message: Optional[str],
    cost: float,
) -> DispatchResult:
    n = len(inputs.target_times)
    if len(ch) != n or len(dis) != n:
        raise DispatchInfeasible("internal: ch/dis length mismatch")
    soc = BatteryModel(inputs.battery).state_vector(ch, dis)
    headline_residual = list(inputs.residual_load_mw)
    # Headline grid demand: P50-based for deterministic, weighted-mean for scenario.
    grid = []
    if inputs.has_scenarios and strategy == "scenario_lp":
        weighted = [
            sum(w * inputs.scenario_residual_load_mw[k][t]
                for k, w in zip(SCENARIO_KEYS, SCENARIO_WEIGHTS))
            for t in range(n)
        ]
    else:
        weighted = headline_residual
    for t in range(n):
        grid.append(weighted[t] - dis[t] + ch[t])

    if strategy == "scenario_lp":
        # Populated by the caller (_strategy_scenario_lp) after building the
        # result, because per-scenario costs use the shared decision.
        scenario_costs = {}
    else:
        # Deterministic strategies: the single "scenario" is the P50 profile;
        # its cost equals the headline simulated cost.
        scenario_costs = {"p50": float(cost), "weighted_total": float(cost)}

    return DispatchResult(
        strategy=strategy,
        issue_time=inputs.issue_time,
        target_times=tuple(inputs.target_times),
        charge_mw=tuple(ch),
        discharge_mw=tuple(dis),
        soc_mwh=tuple(soc),
        grid_demand_mw=tuple(grid),
        residual_load_mw=tuple(headline_residual),
        scenario_residual_load_mw=dict(inputs.scenario_residual_load_mw),
        scenario_costs_eur=scenario_costs,
        simulated_cost_eur=float(cost),
        battery=inputs.battery,
        status=status,
        solver=solver,
        message=message,
    )


# --------------------------------------------------------------------------
# public facade
# --------------------------------------------------------------------------
def run_dispatch(inputs: DispatchInput, strategy: str) -> DispatchResult:
    """Run one deterministic dispatch strategy over ``inputs``."""
    if strategy == "no_battery":
        return _strategy_no_battery(inputs)
    if strategy == "greedy_arbitrage":
        return _strategy_greedy(inputs)
    if strategy == "lp_p50":
        return _strategy_lp_p50(inputs)
    if strategy == "scenario_lp":
        return _strategy_scenario_lp(inputs)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {sorted(STRATEGIES)}")


__all__ = [
    "ACTION_TOL",
    "STRATEGIES",
    "SOLVER_LABEL",
    "run_dispatch",
]