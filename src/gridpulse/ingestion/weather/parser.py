"""Parsing of Open-Meteo JSON responses into per-variable UTC time series.

Verified response shape (live, 2026-09):

.. code-block:: json

    {
      "latitude": 52.2, "longitude": 5.3, "elevation": 11.0,
      "utc_offset_seconds": 0, "timezone": "GMT",
      "hourly_units": {"time": "iso8601", "temperature_2m": "°C", "...": "..."},
      "hourly": {"time": ["2024-01-01T00:00", "..."], "temperature_2m": [3.1, ...]}
    }

Because we always request ``timezone=UTC``, naive ``time`` strings are treated
as UTC. Any null values (rare) are dropped and counted in metadata.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from ..common.errors import EmptyResponseError, MalformedResponseError
from ..common.models import DataPoint, TimeSeries


def _to_utc(label: str) -> datetime:
    dt = datetime.fromisoformat(label)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_historical_json(payload: bytes, *, entity: str = "historical-weather") -> dict[str, TimeSeries]:
    """Return one :class:`TimeSeries` per requested hourly variable."""
    try:
        data = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalformedResponseError(f"open-meteo payload is not valid JSON: {exc}") from exc

    hourly = data.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise MalformedResponseError("open-meteo payload missing 'hourly.time'")

    raw_times = hourly["time"]
    if not isinstance(raw_times, list) or not raw_times:
        raise EmptyResponseError("open-meteo payload has no hourly time points")

    hourly_units = data.get("hourly_units") or {}

    timestamps = [_to_utc(t) for t in raw_times]
    series: dict[str, TimeSeries] = {}
    for variable, raw_values in hourly.items():
        if variable == "time":
            continue
        if not isinstance(raw_values, list) or len(raw_values) != len(raw_times):
            raise MalformedResponseError(
                f"open-meteo column '{variable}' length {len(raw_values)} != times {len(raw_times)}"
            )
        points: list[DataPoint] = []
        nulls = 0
        for ts, value in zip(timestamps, raw_values):
            if value is None:
                nulls += 1
                continue
            points.append(DataPoint(ts, float(value)))

        resolution = _infer_resolution(timestamps)
        series[variable] = TimeSeries(
            source="open-meteo",
            entity=f"{entity}:{variable}",
            unit=str(hourly_units.get(variable, "")),
            points=tuple(points),
            resolution_minutes=resolution,
            tz="UTC",
            metadata={
                "timezone": data.get("timezone"),
                "utc_offset_seconds": data.get("utc_offset_seconds", 0),
                "nulls": nulls,
                "latitude": data.get("latitude"),
                "longitude": data.get("longitude"),
            },
        )
    if not series:
        raise MalformedResponseError("open-meteo payload declared no usable hourly variables")
    return series


def _infer_resolution(timestamps: list[datetime]) -> int | None:
    if len(timestamps) < 2:
        return None
    deltas = sorted(
        int((b - a).total_seconds() / 60)
        for a, b in zip(timestamps, timestamps[1:])
        if b > a
    )
    return deltas[len(deltas) // 2] if deltas else None


__all__ = ["parse_historical_json"]