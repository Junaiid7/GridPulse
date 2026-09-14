"""Backward-looking rolling features over realised history.

All windows are computed from :func:`history_before` (strictly-before) and
cover ``[as_of - window, as_of)`` — they never include the cutoff row, so no
future information can enter.
"""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta
from typing import Optional

from .asof import History, history_before


def rolling_values(history: History, as_of: datetime, window_hours: int) -> list[float]:
    """Realised values within ``[as_of - window_hours, as_of)``."""
    if window_hours <= 0:
        raise ValueError("window must be positive")
    lo = as_of - timedelta(hours=window_hours)
    return [p.value for p in history_before(history, as_of) if p.timestamp_utc >= lo]


def rolling_mean(history: History, as_of: datetime, window_hours: int) -> Optional[float]:
    """Mean of the trailing ``window_hours`` values, or ``None`` if empty."""
    values = rolling_values(history, as_of, window_hours)
    return sum(values) / len(values) if values else None


def rolling_std(history: History, as_of: datetime, window_hours: int) -> Optional[float]:
    """Sample standard deviation of the trailing window (ddof=1), or ``None``."""
    values = rolling_values(history, as_of, window_hours)
    if len(values) < 2:
        return None
    return statistics.stdev(values)


__all__ = ["rolling_values", "rolling_mean", "rolling_std"]