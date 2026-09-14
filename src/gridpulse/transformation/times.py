"""Timezone / DST policy for the transformation and feature layers.

Convention (documented in ``docs/transform-model.md``):

- **Canonical timestamps are aware UTC.** Every stored timestamp is UTC.
- **Local market time** is ``Europe/Amsterdam`` (CET/UTC+1 in winter,
  CEST/UTC+2 in summer). Local values are *derived* from UTC, never stored as
  the canonical clock.
- Because the mapping is always UTC → local, each UTC instant maps to exactly
  one local time — no ambiguous local timestamps can be created. DST
  transitions manifest as short/long local days (23 h on the spring-forward
  day, 25 h on the fall-back day) and as a per-timestamp offset that the
  ISO-8601 ``+01:00`` / ``+02:00`` suffix records explicitly.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from ..ingestion.common.models import ensure_utc

#: IANA name for the Dutch market / TenneT zone.
AMSTERDAM_TZ = ZoneInfo("Europe/Amsterdam")

#: Offsets (minutes) applicable on either side of a transition.
CET_OFFSET_MIN = 60  # UTC+1, winter
CEST_OFFSET_MIN = 120  # UTC+2, summer


def local_time(dt: datetime) -> datetime:
    """Return ``dt`` expressed in Europe/Amsterdam local (aware) time."""
    return ensure_utc(dt).astimezone(AMSTERDAM_TZ)


def local_date(dt: datetime) -> datetime.date:
    """Return the Europe/Amsterdam calendar date for a UTC instant."""
    return local_time(dt).date()


def local_hour(dt: datetime) -> int:
    """Return the Europe/Amsterdam wall-clock hour for a UTC instant."""
    return local_time(dt).hour


def utc_offset_minutes(dt: datetime) -> int:
    """Offset in minutes (``+60`` CET or ``+120`` CEST) applicable to *dt*."""
    offset = local_time(dt).utcoffset()
    if offset is None:
        return 0
    return int(offset.total_seconds() // 60)


__all__ = [
    "AMSTERDAM_TZ",
    "local_time",
    "local_date",
    "local_hour",
    "utc_offset_minutes",
    "CET_OFFSET_MIN",
    "CEST_OFFSET_MIN",
]