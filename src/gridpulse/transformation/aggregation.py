"""Aggregation helpers used by Silver and Gold.

Two deterministic, documented transformations:

- :func:`to_hourly` — downsample any UTC series to a grid of UTC hours by
  averaging the minutes present inside each hour. Only present minutes are
  averaged; an hour with *no* observations stays absent (missing is preserved,
  never invented). Partial hours are counted in metadata.
- :func:`aggregate_locations` — combine per-location series of the same
  variable into one equal-weight mean series. Explicitly NOT a spatial
  optimum; a documented, deterministic default (see ``docs/transform-model.md``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime

from ..ingestion.common.models import DataPoint, TimeSeries, ensure_utc


def to_hourly(series: TimeSeries) -> TimeSeries:
    """Return a new :class:`TimeSeries` on UTC hour boundaries (mean per hour).

    Duplicate timestamps within a source are collapsed by the mean of the
    duplicates; the caller's earlier validation normally excludes them.
    """
    buckets: dict[datetime, list[float]] = {}
    for point in series.points:
        hour_start = ensure_utc(point.timestamp).replace(minute=0, second=0, microsecond=0)
        buckets.setdefault(hour_start, []).append(point.value)

    points = tuple(
        DataPoint(hour_start, sum(vals) / len(vals))
        for hour_start, vals in sorted(buckets.items())
    )
    metadata = dict(series.metadata)
    metadata["downsampled_to_hourly"] = "mean_of_present_minutes"
    return TimeSeries(
        source=series.source,
        entity=series.entity,
        unit=series.unit,
        points=points,
        resolution_minutes=60,
        tz="UTC",
        metadata=metadata,
    )


def aggregate_locations(by_location: Mapping[str, TimeSeries]) -> TimeSeries:
    """Equal-weight mean of per-location series aligned on shared timestamps.

    For each timestamp present in at least one location the value is the mean
    over the locations that have an observation that hour; locations without an
    observation are skipped (their absence is reported in metadata, not
    invented as zero). Returns an empty-points series when no data exists.
    """
    by_location = {name: s for name, s in by_location.items() if s.points}
    if not by_location:
        return TimeSeries(
            source="open-meteo",
            entity="nl-aggregate",
            unit="",
            points=(),
            resolution_minutes=None,
            tz="UTC",
            metadata={"agg_strategy": "equal_weight_location_mean", "locations_used": 0},
        )

    units = {s.unit for s in by_location.values()}
    if len(units) > 1:
        raise ValueError(f"cannot aggregate locations with mixed units: {sorted(units)}")

    per_ts: dict[datetime, list[float]] = {}
    for loc, series in by_location.items():
        for point in series.points:
            ts = ensure_utc(point.timestamp)
            per_ts.setdefault(ts, []).append(point.value)

    n_locations = len(by_location)
    points = tuple(DataPoint(ts, sum(vals) / len(vals)) for ts, vals in sorted(per_ts.items()))
    first = next(iter(by_location.values()))
    return TimeSeries(
        source=first.source,
        entity="nl-aggregate",
        unit=first.unit,
        points=points,
        resolution_minutes=first.resolution_minutes,
        tz="UTC",
        metadata={
            "agg_strategy": "equal_weight_location_mean",
            "locations_total": n_locations,
            "locations_used": n_locations,
            "hours_with_partial_coverage": sum(1 for v in per_ts.values() if len(v) < n_locations),
        },
    )


def as_value_map(points: Iterable[DataPoint]) -> dict[datetime, float]:
    """Index a series' points by exact UTC timestamp."""
    return {ensure_utc(p.timestamp): p.value for p in points}


__all__ = ["to_hourly", "aggregate_locations", "as_value_map"]