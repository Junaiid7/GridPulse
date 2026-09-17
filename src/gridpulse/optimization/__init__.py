"""Battery storage dispatch optimisation (Phase 4D-A).

A **research/simulation** optimizer producing hypothetical battery charge/discharge
schedules given a 24-hour residual-load forecast and day-ahead electricity prices.
This is explicitly NOT real grid/battery/market control.

Core contracts
--------------
- :class:`BatteryConfig` — validated battery parameters (50 MWh / 10 MW defaults).
- :class:`BatteryModel` — deterministic SOC state machinery.
- :class:`DispatchInput` — validated 24-hour problem: prices, residual load,
  optional P10/P50/P90 scenario loads, battery config.
- :class:`DispatchResult` — structured outcome carrying charge/discharge/SOC
  schedule, simulated cost, solver status, and scenario breakdown.
- :class:`DispatchInfeasible` — raised on solver failure (never silent).

Entry point
-----------
:func:`run_dispatch` — public facade dispatching to four baseline strategies:

1. ``"no_battery"`` — zero action baseline.
2. ``"greedy_arbitrage"`` — price-median heuristic.
3. ``"lp_p50"`` — forecast-driven LP using the P50 residual profile.
4. ``"scenario_lp"`` — uncertainty-aware LP with P10/P50/P90 scenarios.

Available strategies are exposed in :data:`STRATEGIES`.

Phase 4D-B (backtesting) — :func:`run_dispatch_backtest` connects the Phase 4C
per-offset P10/P50/P90 forecasts to :func:`run_dispatch` over a chronological
test window and settles every schedule against *realised* gold residual + price
(battery inflection value vs :class:`BatteryConfig` defaults). All backtest
output is ``FIXTURE-VERIFIED``; see :mod:`gridpulse.optimization.backtest` for
the documented conventions and the fixture-vs-live separation.

Usage example
-------------
::

    from gridpulse.optimization import BatteryConfig, DispatchInput, run_dispatch

    inputs = DispatchInput(
        issue_time=...,
        target_times=[...],  # 24 hourly UTC timestamps
        price_eur_mwh=[...],  # 24 day-ahead prices
        residual_load_mw=[...],  # 24 P50 residual-load values
    )
    result = run_dispatch(inputs, strategy="lp_p50")
    print(result.simulated_cost_eur, result.to_dict())

See :mod:`gridpulse.optimization.contract` for input validation rules and
:mod:`gridpulse.optimization.strategies` for LP formulation details.
"""

from .battery import BatteryConfig, BatteryModel, ETA_TOL
from .backtest import (
    ASOF_POLICY,
    DATA_STATUS,
    PRICE_AVAILABILITY_CONVENTION,
    BacktestResult,
    ForecastProfile,
    ForecastProfilesResult,
    assemble_dispatch_input,
    bootstrap_cost_difference_ci,
    build_forecast_profiles,
    run_dispatch_backtest,
    settle_day,
)
from .contract import (
    HORIZON_HOURS,
    PRICE_EPS,
    SCENARIO_KEYS,
    SCENARIO_WEIGHTS,
    DispatchInfeasible,
    DispatchInput,
    DispatchResult,
)
from .strategies import STRATEGIES, run_dispatch

__all__ = [
    "BatteryConfig",
    "BatteryModel",
    "ETA_TOL",
    "HORIZON_HOURS",
    "PRICE_EPS",
    "SCENARIO_KEYS",
    "SCENARIO_WEIGHTS",
    "DispatchInput",
    "DispatchResult",
    "DispatchInfeasible",
    "run_dispatch",
    "STRATEGIES",
    # Phase 4D-B backtesting
    "run_dispatch_backtest",
    "build_forecast_profiles",
    "ForecastProfile",
    "ForecastProfilesResult",
    "assemble_dispatch_input",
    "settle_day",
    "bootstrap_cost_difference_ci",
    "BacktestResult",
    "DATA_STATUS",
    "PRICE_AVAILABILITY_CONVENTION",
    "ASOF_POLICY",
]
