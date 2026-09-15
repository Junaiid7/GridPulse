"""Probabilistic (quantile) forecast contract and utilities (Phase 4C).

Phase 4B produced *point* forecasts. Phase 4C extends the same leakage-safe
24-hour-ahead issue-time contract to the P10 / P50 / P90 predictive quantiles:

- ``p10`` = lower predictive quantile (alpha = 0.10);
- ``p50`` = median predictive quantile (alpha = 0.50);
- ``p90`` = upper predictive quantile (alpha = 0.90).

Contract guarantees
-------------------

- **Quantile ordering is enforced at the boundary.** A ``QuantileForecast``
  with ``p10 > p50`` or ``p50 > p90`` cannot be constructed — the dataclass
  ``__post_init__`` rejects it with ``ValueError``. Invalid triples are
  **detected and refused, never silently reordered**. The quantile model is
  responsible for producing an ordered triple *before* construction (otherwise
  its prediction would fail loudly, which is exactly what we want).
- **Missing predictions stay ``None``.** If the model cannot produce a triple
  (missing features / not fitted), ``p10 = p50 = p90 = None`` for that row —
  never a fabricated fill.
- **The P10/P50/P90 triple keeps the 24-hour-ahead issue semantics.**
  ``issue_time`` / ``target_time`` are populated from the forecasting dataset
  unchanged (see :mod:`gridpulse.forecast.contract`).

Quantile crossing
-----------------

Tree ensembles fit independently per quantile can *cross* (e.g. a fitted P10
exceeding the fitted P50 for some row). Crossing is detected with
:func:`detect_quantile_crossing` and corrected with :func:`correct_quantiles`
(the deterministic ascending re-ordering described there). Correction never
hides crossing silently — the model records how many raw triples crossed.

Risk score
----------

:func:`risk_score` returns ``(P90 - P10) / |P50|`` — a **relative uncertainty
indicator** (relative spread of the predictive distribution), **not** a
financial risk measure. The absolute value of ``P50`` is used so the score is
signed-positive; when ``P50`` is missing or too close to zero the score is
``None`` (division-by-zero / unstable values never silently propagate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

#: Nominal quantiles for the P10/P50/P90 triple.
QUANTILES: tuple[float, float, float] = (0.10, 0.50, 0.90)
NOMINAL_P10 = 0.10
NOMINAL_P50 = 0.50
NOMINAL_P90 = 0.90
#: Nominal coverage of the central 80% interval ``[P10, P90]``.
NOMINAL_INTERVAL_COVERAGE = NOMINAL_P90 - NOMINAL_P10  # 0.80

#: Near-zero guard for the risk-score denominator.
P50_EPS = 1e-9


def risk_score(p10: Optional[float], p50: Optional[float], p90: Optional[float]) -> Optional[float]:
    """Relative uncertainty indicator ``(P90 - P10) / |P50|``.

    Returns ``None`` when any quantile is missing or when ``|P50|`` is at or
    below :data:`P50_EPS` (avoids division-by-zero / unstable amplification).
    Documented as a *relative spread*, not a financial risk measure.
    """
    if p10 is None or p50 is None or p90 is None:
        return None
    if abs(p50) <= P50_EPS:
        return None
    return (p90 - p10) / abs(p50)


def detect_quantile_crossing(
    p10: Optional[float], p50: Optional[float], p90: Optional[float]
) -> bool:
    """True when a triple violates the ordering ``p10 <= p50 <= p90``.

    A triple containing a ``None`` quantile is *undetermined*, not a crossing:
    ``False`` is returned (the model treats it as "no prediction").
    """
    if p10 is None or p50 is None or p90 is None:
        return False
    return not (p10 <= p50 <= p90)


def correct_quantiles(p10, p50, p90):
    """Deterministic monotonic correction for a crossing triple.

    Re-orders the three present quantile values ascending and relabels them
    smallest -> ``p10``, middle -> ``p50``, largest -> ``p90``, guaranteeing
    ``p10 <= p50 <= p90``. Returns ``(p10, p50, p90, corrected)``.

    Justification: sorting is deterministic, order-preserving of the *values*,
    and always yields a valid interval. It is a knowingly crude correction —
    the extreme quantiles may be permuted — so it is applied only after the
    crossing has been counted and reported (see ``crossing_stats`` on the
    model). An undefined triple (any ``None``) is returned unchanged with
    ``corrected=False``.
    """
    vals = (p10, p50, p90)
    if any(v is None for v in vals):
        return p10, p50, p90, False
    lo, mid, hi = sorted(vals)
    corrected = (lo, mid, hi) != vals
    return lo, mid, hi, corrected


@dataclass(frozen=True)
class QuantileForecast:
    """One probabilistic 24h-ahead prediction.

    ``p10``/``p50``/``p90`` are the predicted quantiles of the residual load of
    the target hour. Construction **rejects** unordered triples (``ValueError``)
    — an invalid triple is never silently reordered here.
    """

    issue_time: datetime
    target_time: datetime
    p10: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None
    alpha_low: float = NOMINAL_P10
    alpha_high: float = NOMINAL_P90

    def __post_init__(self):
        if self.alpha_low >= self.alpha_high:
            raise ValueError(
                f"alpha_low ({self.alpha_low}) must be < alpha_high ({self.alpha_high})"
            )
        vals = (self.p10, self.p50, self.p90)
        if all(v is not None for v in vals):
            if not (self.p10 <= self.p50 <= self.p90):
                raise ValueError(
                    "quantile ordering violated: "
                    f"p10={self.p10!r} p50={self.p50!r} p90={self.p90!r} "
                    "(P10<=P50<=P90 required); invalid triples are rejected, "
                    "never silently reordered"
                )

    @property
    def has_quantiles(self) -> bool:
        return all(v is not None for v in (self.p10, self.p50, self.p90))

    @property
    def risk_score(self) -> Optional[float]:
        return risk_score(self.p10, self.p50, self.p90)

    @property
    def width(self) -> Optional[float]:
        """Interval width ``P90 - P10`` (``None`` when undefined)."""
        if not self.has_quantiles:
            return None
        return self.p90 - self.p10


@dataclass(frozen=True)
class QuantileForecasts:
    """Ordered set of probabilistic forecasts (one per issue row).

    Mirrors the :class:`~gridpulse.forecast.contract.ForecastingDataset`
    pattern: rows are sorted by issue time (predictions must be aligned with
    the dataset's issue rows by the caller, typically the benchmark).
    """

    rows: Sequence[QuantileForecast] = field(default_factory=list)
    quantiles: tuple[float, float, float] = QUANTILES

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.rows)


__all__ = [
    "QUANTILES",
    "NOMINAL_P10",
    "NOMINAL_P50",
    "NOMINAL_P90",
    "NOMINAL_INTERVAL_COVERAGE",
    "P50_EPS",
    "risk_score",
    "detect_quantile_crossing",
    "correct_quantiles",
    "QuantileForecast",
    "QuantileForecasts",
]