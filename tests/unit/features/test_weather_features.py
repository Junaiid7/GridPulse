"""Unit tests for weather features: last-observation-before and NL aggregation."""

from __future__ import annotations

from datetime import datetime, timezone

from gridpulse.features.weather import hourly_nl, last_observation_before, weather_features
from gridpulse.ingestion.common.models import DataPoint, TimeSeries

UTC = timezone.utc


def _series(points: list[tuple[str, float]], *, unit: str = "%") -> TimeSeries:
    return TimeSeries(
        source="open-meteo",
        entity="temperature",
        unit=unit,
        points=tuple(DataPoint(datetime.fromisoformat(t).astimezone(UTC), v) for t, v in points),
        resolution_minutes=60,
        tz="UTC",
    )


def test_last_observation_before_returns_latest_value() -> None:
    s = _series([
        ("2024-01-01T10:00:00Z", 5.0),
        ("2024-01-01T11:00:00Z", 7.0),
        ("2024-01-01T13:00:00Z", 9.0),
    ])
    assert last_observation_before(s, datetime(2024, 1, 1, 12, 0, tzinfo=UTC)) == 7.0
    # 13:00 is exactly at the cutoff and must be excluded.
    assert last_observation_before(s, datetime(2024, 1, 1, 13, 0, tzinfo=UTC)) == 7.0
    assert last_observation_before(s, datetime(2024, 1, 1, 14, 0, tzinfo=UTC)) == 9.0


def test_last_observation_before_none_when_empty() -> None:
    assert last_observation_before(_series([]), datetime(2024, 1, 1, 12, 0, tzinfo=UTC)) is None


def test_last_observation_before_none_when_all_future() -> None:
    s = _series([("2024-01-01T13:00:00Z", 5.0)])
    assert last_observation_before(s, datetime(2024, 1, 1, 12, 0, tzinfo=UTC)) is None


def test_hourly_nl_equal_weight_mean() -> None:
    a = _series([("2024-01-01T12:00:00Z", 2.0), ("2024-01-01T13:00:00Z", 4.0)])
    b = _series([("2024-01-01T12:00:00Z", 6.0)])
    agg = hourly_nl({"a": a, "b": b})
    assert agg.entity == "nl-aggregate"
    vals = {p.timestamp.hour: p.value for p in agg.points}
    assert vals[12] == 4.0  # (2+6)/2
    assert vals[13] == 4.0  # only a


def test_hourly_nl_15min_downsampled_first() -> None:
    # Per-location 15-min data should be downsampled to hourly before combining.
    a = _series([
        ("2024-01-01T12:00:00Z", 1.0),
        ("2024-01-01T12:15:00Z", 3.0),
        ("2024-01-01T12:30:00Z", 5.0),
        ("2024-01-01T12:45:00Z", 7.0),
    ])
    agg = hourly_nl({"a": a})
    assert len(agg.points) == 1
    assert agg.points[0].value == 4.0


def test_weather_features_per_variable() -> None:
    by_var = {
        "temperature_2m": _series([
            ("2024-01-01T11:00:00Z", 5.0),
            ("2024-01-01T12:00:00Z", 7.0),
        ]),
        "wind_speed_10m": _series([
            ("2024-01-01T11:00:00Z", 8.0),
            ("2024-01-01T12:00:00Z", 6.0),
        ]),
    }
    feats = weather_features(by_var, datetime(2024, 1, 1, 12, 0, tzinfo=UTC))
    # The 12:00 observations are at the cutoff and must be excluded.
    assert feats["temperature_2m"] == 5.0  # last strictly before 12:00
    assert feats["wind_speed_10m"] == 8.0