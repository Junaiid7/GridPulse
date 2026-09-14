"""Unit tests for aggregation helpers: downsampling and location combining."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gridpulse.ingestion.common.models import DataPoint, TimeSeries
from gridpulse.transformation.aggregation import aggregate_locations, as_value_map, to_hourly

UTC = timezone.utc


def _series(points: list[tuple[str, float]], *, unit: str = "MW") -> TimeSeries:
    return TimeSeries(
        source="entsoe",
        entity="test",
        unit=unit,
        points=tuple(DataPoint(datetime.fromisoformat(t).astimezone(UTC), v) for t, v in points),
        resolution_minutes=15,
        tz="UTC",
    )


def test_to_hourly_averages_present_minutes() -> None:
    s = _series(
        [
            ("2024-01-01T12:00:00+00:00", 1.0),
            ("2024-01-01T12:15:00+00:00", 2.0),
            ("2024-01-01T12:30:00+00:00", 3.0),
            ("2024-01-01T12:45:00+00:00", 4.0),
            ("2024-01-01T13:00:00+00:00", 6.0),
        ]
    )
    hourly = to_hourly(s)
    assert [p.timestamp.isoformat() for p in hourly.points] == [
        "2024-01-01T12:00:00+00:00",
        "2024-01-01T13:00:00+00:00",
    ]
    assert [p.value for p in hourly.points] == [2.5, 6.0]
    assert hourly.resolution_minutes == 60
    assert hourly.metadata["downsampled_to_hourly"] == "mean_of_present_minutes"


def test_to_hourly_preserves_missing_hours() -> None:
    # An hour with no observations stays absent rather than being invented.
    s = _series(
        [
            ("2024-01-01T12:00:00+00:00", 1.0),
            ("2024-01-01T14:00:00+00:00", 9.0),
        ]
    )
    hourly = to_hourly(s)
    assert [p.timestamp.hour for p in hourly.points] == [12, 14]


def test_to_hourly_collapses_duplicate_timestamps_by_mean() -> None:
    s = _series(
        [
            ("2024-01-01T12:00:00+00:00", 1.0),
            ("2024-01-01T12:00:00+00:00", 3.0),
        ]
    )
    hourly = to_hourly(s)
    assert len(hourly.points) == 1
    assert hourly.points[0].value == 2.0


def test_aggregate_locations_equal_weight_mean() -> None:
    a = _series(
        [
            ("2024-01-01T12:00:00+00:00", 2.0),
            ("2024-01-01T13:00:00+00:00", 4.0),
        ]
    )
    b = _series(
        [
            ("2024-01-01T12:00:00+00:00", 6.0),
            ("2024-01-01T14:00:00+00:00", 8.0),
        ]
    )
    agg = aggregate_locations({"a": a, "b": b})
    assert agg.entity == "nl-aggregate"
    by_ts = as_value_map(agg.points)
    assert by_ts[datetime(2024, 1, 1, 12, tzinfo=UTC)] == 4.0  # (2+6)/2
    assert by_ts[datetime(2024, 1, 1, 13, tzinfo=UTC)] == 4.0  # only a
    assert by_ts[datetime(2024, 1, 1, 14, tzinfo=UTC)] == 8.0  # only b
    assert agg.metadata["locations_total"] == 2
    assert agg.metadata["hours_with_partial_coverage"] == 2


def test_aggregate_locations_empty() -> None:
    agg = aggregate_locations({})
    assert agg.entity == "nl-aggregate"
    assert len(agg.points) == 0
    assert agg.metadata["locations_used"] == 0


def test_aggregate_locations_ignores_empty_series() -> None:
    full = _series([("2024-01-01T12:00:00+00:00", 5.0)])
    empty = _series([])
    agg = aggregate_locations({"full": full, "empty": empty})
    assert agg.metadata["locations_total"] == 1
    assert agg.metadata["locations_used"] == 1


def test_aggregate_locations_rejects_mixed_units() -> None:
    a = _series([("2024-01-01T12:00:00+00:00", 1.0)], unit="MW")
    b = _series([("2024-01-01T12:00:00+00:00", 1.0)], unit="GW")
    with pytest.raises(ValueError):
        aggregate_locations({"a": a, "b": b})