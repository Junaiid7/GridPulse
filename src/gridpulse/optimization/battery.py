"""Battery storage model (Phase 4D-A).

A *generic hypothetical battery* used by the dispatch optimizer. This is a
**research/simulation model** — it does not, and is not meant to, control a
real battery, grid or market.

Convention (documented here once, used everywhere downstream):

- ``capacity_mwh``, powers in ``MW`` over a one-hour period (so
  ``power MW · 1 h = energy MWh``).
- SOC is stored as an **energy** quantity in MWh internally and exposed as a
  **fraction of capacity** on the config.
- ``round_trip_efficiency`` is the end-to-end efficiency of a full
  charge→discharge cycle. By default it is split **evenly** between the two
  directions: ``charge_efficiency = discharge_efficiency = sqrt(eta_rt)``,
  so ``charge_efficiency * discharge_efficiency == round_trip_efficiency``.
  Callers may instead supply explicit per-direction efficiencies; the product
  must then equal the configured round-trip efficiency (validated).
- Sign of an action: ``charge > 0`` increases SOC, ``discharge > 0`` decreases
  SOC. The two may be split per direction because real losses differ by
  direction; the LP/hueristics keep them as separate variables but the optimal
  LP solution never charges and discharges in the same hour (see
  :mod:`gridpulse.optimization.strategies` for the statement/verification).

SOC transition (exact, one linear equation per hour):

    SOC[t+1] = SOC[t] + charge[t] * charge_efficiency
                     - discharge[t] / discharge_efficiency

The ``discharge[t] / discharge_efficiency`` term is the *withdrawal from the
battery* required to deliver ``discharge[t]`` MW to the grid: because
:math:`eta_d < 1`, one grid-MWh of discharge consumes more than one
battery-MWh.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

#: Tolerance for validating efficiency products.
ETA_TOL = 1e-9


def _as_float(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return value


@dataclass(frozen=True)
class BatteryConfig:
    """Validated, immutable battery parameters (generic hypothetical battery).

    SOC bounds and the initial SOC are stored as **fractions** of
    ``capacity_mwh`` in ``[0, 1]``. Defaults match the Phase 4D-A brief.

    ``charge_efficiency`` / ``discharge_efficiency`` may be supplied
    explicitly; otherwise both default to ``sqrt(round_trip_efficiency)`` (an
    even split of the round-trip loss). Whatever values are used,
    ``charge_efficiency * discharge_efficiency`` must equal
    ``round_trip_efficiency`` within :data:`ETA_TOL`.
    """

    capacity_mwh: float = 50.0
    max_charge_power_mw: float = 10.0
    max_discharge_power_mw: float = 10.0
    round_trip_efficiency: float = 0.90
    soc_min: float = 0.10
    soc_max: float = 0.90
    initial_soc: float = 0.50
    charge_efficiency: Optional[float] = None
    discharge_efficiency: Optional[float] = None

    def __post_init__(self):
        capacity = _as_float("capacity_mwh", self.capacity_mwh)
        if capacity <= 0:
            raise ValueError(f"capacity_mwh must be > 0; got {self.capacity_mwh!r}")
        p_charge = _as_float("max_charge_power_mw", self.max_charge_power_mw)
        if p_charge <= 0:
            raise ValueError(f"max_charge_power_mw must be > 0; got {self.max_charge_power_mw!r}")
        p_discharge = _as_float("max_discharge_power_mw", self.max_discharge_power_mw)
        if p_discharge <= 0:
            raise ValueError(
                f"max_discharge_power_mw must be > 0; got {self.max_discharge_power_mw!r}"
            )
        eta_rt = _as_float("round_trip_efficiency", self.round_trip_efficiency)
        if not 0.0 < eta_rt <= 1.0:
            raise ValueError(f"round_trip_efficiency must be in (0, 1]; got {self.round_trip_efficiency!r}")

        for name, frac in (("soc_min", self.soc_min), ("soc_max", self.soc_max),
                           ("initial_soc", self.initial_soc)):
            _as_float(name, frac)
        if not 0.0 <= self.soc_min <= 1.0:
            raise ValueError(f"soc_min must be in [0, 1]; got {self.soc_min!r}")
        if not 0.0 <= self.soc_max <= 1.0:
            raise ValueError(f"soc_max must be in [0, 1]; got {self.soc_max!r}")
        if self.soc_min > self.soc_max:
            raise ValueError(f"soc_min ({self.soc_min}) must be <= soc_max ({self.soc_max})")
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError(
                f"initial_soc ({self.initial_soc}) must lie in [soc_min, soc_max] "
                f"= [{self.soc_min}, {self.soc_max}]"
            )

        eta_c = (
            math.sqrt(eta_rt) if self.charge_efficiency is None
            else _as_float("charge_efficiency", self.charge_efficiency)
        )
        eta_d = (
            math.sqrt(eta_rt) if self.discharge_efficiency is None
            else _as_float("discharge_efficiency", self.discharge_efficiency)
        )
        if eta_c <= 0 or eta_c > 1:
            raise ValueError(f"charge_efficiency must be in (0, 1]; got {eta_c!r}")
        if eta_d <= 0 or eta_d > 1:
            raise ValueError(f"discharge_efficiency must be in (0, 1]; got {eta_d!r}")
        if abs(eta_c * eta_d - eta_rt) > ETA_TOL:
            raise ValueError(
                "charge_efficiency * discharge_efficiency must equal "
                f"round_trip_efficiency; got {eta_c * eta_d:.9f} != {eta_rt:.9f}"
            )
        # Normalise validated floats back onto the frozen dataclass.
        object.__setattr__(self, "capacity_mwh", capacity)
        object.__setattr__(self, "max_charge_power_mw", p_charge)
        object.__setattr__(self, "max_discharge_power_mw", p_discharge)
        object.__setattr__(self, "round_trip_efficiency", eta_rt)
        object.__setattr__(self, "soc_min", float(self.soc_min))
        object.__setattr__(self, "soc_max", float(self.soc_max))
        object.__setattr__(self, "initial_soc", float(self.initial_soc))
        object.__setattr__(self, "charge_efficiency", eta_c)
        object.__setattr__(self, "discharge_efficiency", eta_d)

    # -- energy (MWh) view of the fractional SOC values ---------------------
    @property
    def soc_min_mwh(self) -> float:
        return self.capacity_mwh * self.soc_min

    @property
    def soc_max_mwh(self) -> float:
        return self.capacity_mwh * self.soc_max

    @property
    def initial_soc_mwh(self) -> float:
        return self.capacity_mwh * self.initial_soc

    def to_dict(self) -> dict:
        return {
            "capacity_mwh": self.capacity_mwh,
            "max_charge_power_mw": self.max_charge_power_mw,
            "max_discharge_power_mw": self.max_discharge_power_mw,
            "round_trip_efficiency": self.round_trip_efficiency,
            "charge_efficiency": self.charge_efficiency,
            "discharge_efficiency": self.discharge_efficiency,
            "soc_min": self.soc_min,
            "soc_max": self.soc_max,
            "initial_soc": self.initial_soc,
        }


class BatteryModel:
    """Deterministic state machinery over a :class:`BatteryConfig`.

    Stateless: every method is a pure function of its inputs, so reuse across
    strategies and tests is safe and reproducible.
    """

    def __init__(self, battery: BatteryConfig):
        self.battery = battery

    @property
    def charge_efficiency(self) -> float:
        return self.battery.charge_efficiency

    @property
    def discharge_efficiency(self) -> float:
        return self.battery.discharge_efficiency

    def state_vector(
        self,
        charge_mw: Sequence[float],
        discharge_mw: Sequence[float],
        *,
        start_soc_mwh: Optional[float] = None,
    ) -> list[float]:
        """SOC (MWh) at the *end* of each period, in order.

        Uses the exact transition

            SOC[t+1] = SOC[t] + charge[t]*eta_c - discharge[t]/eta_d

        with ``SOC[0]`` seeded from ``start_soc_mwh`` (default the config's
        initial SOC). Raises ``ValueError`` on length mismatch.
        """
        if len(charge_mw) != len(discharge_mw):
            raise ValueError("charge and discharge vectors must be the same length")
        soc = start_soc_mwh if start_soc_mwh is not None else self.battery.initial_soc_mwh
        trajectory = []
        for ch, dis in zip(charge_mw, discharge_mw):
            soc = (
                soc
                + float(ch) * self.charge_efficiency
                - float(dis) / self.discharge_efficiency
            )
            trajectory.append(soc)
        return trajectory


__all__ = ["ETA_TOL", "BatteryConfig", "BatteryModel"]