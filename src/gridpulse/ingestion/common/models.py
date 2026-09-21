"""Shared data models and time-range helpers for ingestion."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any


def ensure_utc(dt: datetime) -> datetime:
    """Return *dt* converted to an aware UTC datetime.

    Naive datetimes are assumed to be UTC already (the convention for both
    the ENTSO-E and Open-Meteo payloads we ingest) and are simply tagged.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def chunk_range(start: datetime, end: datetime, max_span: timedelta) -> list[TimeRange]:
    """Split ``[start, end)`` into consecutive sub-ranges no longer than max_span."""
    if max_span <= timedelta(0):
        raise ValueError("max_span must be positive")
    out: list[TimeRange] = []
    cursor = ensure_utc(start)
    stop = ensure_utc(end)
    while cursor < stop:
        nxt = min(cursor + max_span, stop)
        out.append(TimeRange(cursor, nxt))
        cursor = nxt
    return out


@dataclass(frozen=True)
class TimeRange:
    """A half-open, timezone-aware UTC time range ``[start, end)``."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("TimeRange requires timezone-aware datetimes")
        if self.start.astimezone(UTC) >= self.end.astimezone(UTC):
            raise ValueError("TimeRange end must be strictly after start")
        object.__setattr__(self, "start", ensure_utc(self.start))
        object.__setattr__(self, "end", ensure_utc(self.end))

    def chunk(self, max_span: timedelta) -> list[TimeRange]:
        return chunk_range(self.start, self.end, max_span)

    def split(self) -> tuple[TimeRange, TimeRange]:
        """Split into two halves (the half-open windows drop any empty half)."""
        mid = self.start + (self.end - self.start) / 2
        if mid <= self.start or mid >= self.end:  # single instants cannot be halved
            return (self, self)
        return (TimeRange(self.start, mid), TimeRange(mid, self.end))

    def __str__(self) -> str:
        return f"[{self.start:%Y-%m-%d %H:%M}Z, {self.end:%Y-%m-%d %H:%M}Z)"


@dataclass(frozen=True)
class DataPoint:
    """A single timestamped observation."""

    timestamp: datetime
    value: float


@dataclass(frozen=True)
class TimeSeries:
    """A normalised time series at ingestion resolution (already UTC-aware)."""

    source: str
    entity: str
    unit: str
    points: tuple[DataPoint, ...]
    resolution_minutes: int | None = None
    tz: str = "UTC"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def timestamps(self) -> list[datetime]:
        return [p.timestamp for p in self.points]

    @property
    def values(self) -> list[float]:
        return [p.value for p in self.points]

    def __len__(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class FetchResult:
    """A raw fetch destined for the Bronze tier.

    ``payload`` is the faithful raw response bytes; everything else is context
    captured so the raw data can be replayed or audited later.
    """

    source: str
    entity: str
    start: datetime
    end: datetime
    retrieved_at: datetime
    payload: bytes
    content_type: str | None = None
    encoding: str | None = None
    url: str | None = None
    identifiers: tuple[str, ...] = ()
    timezone: str = "UTC"
    units: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()