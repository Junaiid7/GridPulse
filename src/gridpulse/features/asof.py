"""The single information-cut guard used by every feature builder.

Leakage rule (see ``docs/transform-model.md``): for a forecast issued at
``as_of``, predictive features may only use observations with
``timestamp_utc < as_of``. Everything below funnels history through
:func:`history_before`, so a future row — including one at the exact cutoff —
can never leak into a lag or rolling feature.

``as_of`` is the *issue time* of the 24 h-ahead forecast. ``target_utc`` (the
forecast hour) is always after ``as_of``; calendar features of the target hour
are derived from the calendar alone and use no data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Union

from ..ingestion.common.models import ensure_utc


@dataclass(frozen=True)
class HistoryPoint:
    """One realised observation on the UTC timeline."""

    timestamp_utc: datetime
    value: float


History = Iterable[Union[HistoryPoint, tuple]]  # accepts (dt, value) pairs too


def history_before(history: History, as_of: datetime) -> list[HistoryPoint]:
    """Return realised observations strictly before ``as_of`` (UTC).

    Raises ``ValueError`` when ``as_of`` is naive — a leaky cutoff must never
    be silently misinterpreted.
    """
    cutoff = _aware(as_of, "as_of")
    out: list[HistoryPoint] = []
    for item in history:
        if isinstance(item, tuple):
            item = HistoryPoint(item[0], float(item[1]))
        if item.timestamp_utc.tzinfo is None:
            raise ValueError("history timestamps must be timezone-aware (UTC)")
        if ensure_utc(item.timestamp_utc) < cutoff:
            out.append(item)
    return out


def _aware(dt: datetime, name: str) -> datetime:
    if dt.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware (UTC)")
    return ensure_utc(dt)


__all__ = ["HistoryPoint", "history_before"]