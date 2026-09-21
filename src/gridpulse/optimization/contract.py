"""Battery-dispatch optimisation contracts (Phase 4D-A).

Two dataclasses form the public surface of the dispatch engine:

- :class:`DispatchInput` — a validated 24-hour optimisation problem: hourly
  target timestamps, day-ahead electricity price, a residual-load profile
  (P50 / system) and optionally P10/P50/P90 scenario-load profiles, plus the
  battery configuration.
- :class:`DispatchResult` — a structured outcome carrying enough for testing
  now and historical backtesting in Phase 4D-B.

Conventions mirrored from the forecast layer (Phase 4B/4C):

- timestamps are **aware UTC**, hourly, strictly increasing, exactly 24 of
  them (``HORIZON_HOURS``);
- P10 <= P50 <= P90 ordering is enforced at the boundary, exactly like
  ``QuantileForecast`` (Phase 4C) — an invalid triple raises ``ValueError``
  and is never silently reordered;
- missing/non-finite values are **rejected**, never fabricated (no invented
  prices, no invented loads);
- the information-cutoff *semantics* are inherited from the caller: the
  residual-load profile (and any P10/P50/P90 spread) is a forecast available
  at ``issue_time``; the price vector is the day-ahead market price known when
  a day-ahead dispatch decision is made. The optimizer itself only performs
  arithmetic on the values it is given.

Sign convention (documented once here, used everywhere):

- positive residual load  = demand exceeds wind/solar generation;
- battery discharge       = reduces net grid demand (service);
- battery charge          = increases net grid demand (consumption);
- net grid demand ``g[t] = residual_load[t] - discharge[t] + charge[t]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..ingestion.common.models import ensure_utc
from .battery import BatteryConfig

#: Periods in one day-ahead dispatch horizon.
HORIZON_HOURS = 24

#: Scenario keys accepted for scenario residual loads (Phase 4C convention).
SCENARIO_KEYS = ("p10", "p50", "p90")

#: Default scenario weights (Phase 4C convention): P10 0.25 / P50 0.50 / P90 0.25.
SCENARIO_WEIGHTS = (0.25, 0.50, 0.25)

#: Tolerance under which a price is treated as zero-safe.
PRICE_EPS = 1e-9


def _check_finite_vector(values, *, name: str, tol: float | None = None) -> tuple[float, ...]:
    out = tuple(float(v) for v in values)
    if len(out) != HORIZON_HOURS:
        raise ValueError(
            f"{name} must have exactly {HORIZON_HOURS} entries; got {len(out)}"
        )
    for v in out:
        if not math.isfinite(v):
            raise ValueError(f"{name} must contain only finite values; got {v!r}")
        if tol is not None and v < tol:
            raise ValueError(f"{name} must be >= {tol}; got {v!r}")
    return out


def _check_target_times(times: Sequence[datetime]) -> tuple[datetime, ...]:
    if len(times) != HORIZON_HOURS:
        raise ValueError(
            f"target_times must contain exactly {HORIZON_HOURS} hourly timestamps; "
            f"got {len(times)}"
        )
    out = tuple(times)
    for t in out:
        aware = ensure_utc(t)
        if aware.minute != 0 or aware.second != 0 or aware.microsecond != 0:
            raise ValueError(f"target timestamp must be on the hour; got {t!r}")
    for i in range(1, len(out)):
        prev, cur = ensure_utc(out[i - 1]), ensure_utc(out[i])
        if cur <= prev:
            raise ValueError(
                f"target_times must be strictly increasing; period {i} ({cur}) "
                f"does not follow {prev}"
            )
        if cur - prev != timedelta(hours=1):
            raise ValueError(
                f"target_times must be spaced exactly 1 hour apart; period {i} "
                f"has gap {cur - prev}"
            )
    return tuple(ensure_utc(t) for t in out)


@dataclass(frozen=True)
class DispatchInput:
    """A validated 24-hour battery-dispatch problem.

    Fields
    ------
    issue_time:
        Forecast issue instant (aware UTC) — when the residual-load forecast
        used here would have been produced. Informational anchor only.
    target_times:
        Exactly :data:`HORIZON_HOURS` hourly, strictly-increasing, aware-UTC
        timestamps of the dispatch horizon.
    price_eur_mwh:
        Hourly day-ahead electricity price, one per target hour, finite and
        non-negative.
    residual_load_mw:
        The P50 / system residual-load profile, one per target hour, finite.
    scenario_residual_load_mw:
        Optional P10/P50/P90 scenario loads. All-or-none; each 24 long;
        per-hour ``p10 <= p50 <= p90``; ``scenario["p50"]`` must equal
        ``residual_load_mw``. Used by the uncertainty-aware scenario strategy.
    battery:
        Battery configuration (defaults = the generic 50 MWh / 10 MW battery).
    curtailment_allowed:
        When ``False`` (default) net grid demand is constrained ``>= 0`` in
        every scenario (no export / no curtailment). When ``True`` negative net
        demand is permitted and documented as an explicit export interpretation.
    terminal_soc:
        Fraction of capacity the battery must have at the *end* of the horizon
        (in ``[soc_min, soc_max]``). Defaults to ``battery.initial_soc``,
        making the simulated horizon energy-neutral (conservative for the cost
        figure). Pass ``soc_min`` explicitly for a maximally free terminal.
    """

    issue_time: datetime
    target_times: Sequence[datetime]
    price_eur_mwh: Sequence[float]
    residual_load_mw: Sequence[float]
    scenario_residual_load_mw: Mapping[str, Sequence[float]] = field(default_factory=dict)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    curtailment_allowed: bool = False
    terminal_soc: float | None = None

    def __post_init__(self):
        if not isinstance(self.issue_time, datetime):
            raise TypeError(f"issue_time must be a datetime; got {type(self.issue_time).__name__}")
        ensure_utc(self.issue_time)  # raises on naive — never misinterpreted

        object.__setattr__(self, "target_times", _check_target_times(self.target_times))
        object.__setattr__(self, "price_eur_mwh", _check_finite_vector(
            self.price_eur_mwh, name="price_eur_mwh", tol=0.0))
        object.__setattr__(self, "residual_load_mw", _check_finite_vector(
            self.residual_load_mw, name="residual_load_mw"))

        scenarios = dict(self.scenario_residual_load_mw)
        if scenarios and set(scenarios) != set(SCENARIO_KEYS):
            raise ValueError(
                f"scenario_residual_load_mw must provide exactly the keys "
                f"{SCENARIO_KEYS}; got {sorted(scenarios)}"
            )
        if scenarios:
            checked = {}
            for key in SCENARIO_KEYS:
                checked[key] = _check_finite_vector(scenarios[key], name=f"scenario_{key}")
            for i in range(HORIZON_HOURS):
                p10 = checked["p10"][i]
                p50 = checked["p50"][i]
                p90 = checked["p90"][i]
                if not (p10 <= p50 <= p90):
                    raise ValueError(
                        f"scenario ordering violated at hour {i}: "
                        f"p10={p10!r} p50={p50!r} p90={p90!r} "
                        "(P10<=P50<=P90 required; invalid triples are rejected, "
                        "never silently reordered)"
                    )
            if checked["p50"] != self.residual_load_mw:
                raise ValueError(
                    "scenario_residual_load_mw['p50'] must equal residual_load_mw"
                )
            object.__setattr__(self, "scenario_residual_load_mw", checked)

        if isinstance(self.battery, BatteryConfig):
            battery = self.battery
        else:  # pragma: no cover - type guard
            raise TypeError(f"battery must be a BatteryConfig; got {type(self.battery).__name__}")

        terminal = self.terminal_soc
        if terminal is None:
            terminal = battery.initial_soc
        terminal = float(terminal)
        if not math.isfinite(terminal):
            raise ValueError(f"terminal_soc must be finite; got {terminal!r}")
        if not battery.soc_min <= terminal <= battery.soc_max:
            raise ValueError(
                f"terminal_soc ({terminal}) must lie in [soc_min, soc_max] "
                f"= [{battery.soc_min}, {battery.soc_max}]"
            )
        object.__setattr__(self, "battery", battery)
        object.__setattr__(self, "curtailment_allowed", bool(self.curtailment_allowed))
        object.__setattr__(self, "terminal_soc", terminal)

    @property
    def horizon_hours(self) -> int:
        return len(self.target_times)

    @property
    def has_scenarios(self) -> bool:
        return bool(self.scenario_residual_load_mw)

    @property
    def nominal_scenario_weights(self) -> tuple[float, float, float]:
        return SCENARIO_WEIGHTS


@dataclass(frozen=True)
class DispatchResult:
    """Structured output of one dispatch strategy.

    Fields are laid out for direct consumption by Phase 4D-B backtesting:
    ``to_dict()`` is the machine-readable form and includes every field below.
    """

    strategy: str
    issue_time: datetime
    target_times: tuple
    charge_mw: tuple
    discharge_mw: tuple
    soc_mwh: tuple                       # end-of-hour SOC trajectory, len 24
    grid_demand_mw: tuple                # headline net grid demand per hour
    residual_load_mw: tuple              # P50 / system profile used for the headline
    scenario_residual_load_mw: Mapping   # empty for deterministic strategies
    scenario_costs_eur: Mapping          # per-scenario + weighted_total
    simulated_cost_eur: float            # sum(price[t] * grid_demand[t])
    battery: BatteryConfig
    status: str                          # "optimal" | "ok"
    solver: str
    message: str | None
    data_status: str = "FIXTURE-VERIFIED"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "issue_time": self.issue_time.isoformat(),
            "target_times": [t.isoformat() for t in self.target_times],
            "charge_mw": list(self.charge_mw),
            "discharge_mw": list(self.discharge_mw),
            "soc_mwh": list(self.soc_mwh),
            "grid_demand_mw": list(self.grid_demand_mw),
            "residual_load_mw": list(self.residual_load_mw),
            "scenario_residual_load_mw": {
                k: list(v) for k, v in self.scenario_residual_load_mw.items()
            },
            "scenario_costs_eur": dict(self.scenario_costs_eur),
            "simulated_cost_eur": self.simulated_cost_eur,
            "battery": self.battery.to_dict(),
            "status": self.status,
            "solver": self.solver,
            "message": self.message,
            "data_status": self.data_status,
        }


#: Purpose-specific exception for solver failure — never hidden.
class DispatchInfeasible(ValueError):
    """Raised when the dispatch LP is infeasible (or the solver otherwise
    fails). Carries the solver message so the failure is never silent."""


__all__ = [
    "HORIZON_HOURS",
    "SCENARIO_KEYS",
    "SCENARIO_WEIGHTS",
    "PRICE_EPS",
    "DispatchInput",
    "DispatchResult",
    "DispatchInfeasible",
]