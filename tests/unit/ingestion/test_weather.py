"""Tests for Open-Meteo ingestion: request parameter construction, JSON
parsing, unit/timezone metadata, null handling, and chunked windows.
No live network.
"""

from __future__ import annotations

import json
from datetime import UTC, date

import pytest

from gridpulse.ingestion.common.errors import EmptyResponseError, MalformedResponseError
from gridpulse.ingestion.common.http import HttpResponse
from gridpulse.ingestion.weather.open_meteo import OpenMeteoClient
from gridpulse.ingestion.weather.parser import parse_historical_json
from gridpulse.ingestion.weather.variables import Location

UTC = UTC

SAMPLE = {
    "latitude": 52.21,
    "longitude": 5.29,
    "elevation": 5.0,
    "utc_offset_seconds": 0,
    "timezone": "GMT",
    "hourly_units": {
        "time": "iso8601",
        "temperature_2m": "°C",
        "wind_speed_100m": "m/s",
        "shortwave_radiation": "W/m²",
    },
    "hourly": {
        "time": ["2024-01-01T00:00", "2024-01-01T01:00", "2024-01-01T02:00"],
        "temperature_2m": [3.1, 2.9, None],
        "wind_speed_100m": [8.9, 9.4, 9.9],
        "shortwave_radiation": [0.0, 0.0, 12.5],
    },
}

CENTRAL = Location("nl-central", 52.21, 5.29)


class FakeWeatherHttp:
    def __init__(self, body: bytes | str):
        self.calls: list[dict] = []
        self.body = body.encode() if isinstance(body, str) else body

    def get(self, url, *, params=None, headers=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        return HttpResponse(
            200, {"content-type": "application/json"}, self.body, "http://fake"
        )


def test_fetch_historical_builds_expected_params() -> None:
    fake = FakeWeatherHttp(json.dumps(SAMPLE))
    client = OpenMeteoClient(http=fake)
    results = client.fetch_historical(
        CENTRAL,
        date(2024, 1, 1),
        date(2024, 1, 1),
        variables=("temperature_2m", "wind_speed_100m"),
    )

    params = fake.calls[0]["params"]
    assert params["latitude"] == CENTRAL.lat
    assert params["longitude"] == CENTRAL.lon
    assert params["start_date"] == "2024-01-01"
    assert params["end_date"] == "2024-01-01"
    assert params["hourly"] == "temperature_2m,wind_speed_100m"
    assert params["timezone"] == "UTC"
    assert params["wind_speed_unit"] == "ms"

    result = results[0]
    assert result.source == "open-meteo"
    assert result.entity == "historical-weather"
    assert result.start.isoformat() == "2024-01-01T00:00:00+00:00"
    assert result.end.isoformat() == "2024-01-02T00:00:00+00:00"
    assert "location:nl-central" in result.identifiers
    assert result.metadata["variables"].startswith("temperature_2m")


def test_fetch_historical_chunk_days_splits_windows() -> None:
    fake = FakeWeatherHttp(json.dumps(SAMPLE))
    client = OpenMeteoClient(http=fake)
    client.fetch_historical(CENTRAL, date(2024, 1, 1), date(2024, 1, 3), chunk_days=2)
    assert len(fake.calls) == 2
    assert (
        fake.calls[0]["params"]["start_date"],
        fake.calls[0]["params"]["end_date"],
    ) == ("2024-01-01", "2024-01-02")
    assert (
        fake.calls[1]["params"]["start_date"],
        fake.calls[1]["params"]["end_date"],
    ) == ("2024-01-03", "2024-01-03")


def test_fetch_historical_rejects_inverted_range() -> None:
    client = OpenMeteoClient(http=FakeWeatherHttp(b"{}"))
    with pytest.raises(ValueError):
        client.fetch_historical(CENTRAL, date(2024, 1, 3), date(2024, 1, 1))


def test_parse_historical_json_builds_variable_series() -> None:
    series = parse_historical_json(json.dumps(SAMPLE).encode())
    assert set(series) == {"temperature_2m", "wind_speed_100m", "shortwave_radiation"}
    assert series["wind_speed_100m"].unit == "m/s"
    assert series["temperature_2m"].unit == "°C"
    assert all(p.timestamp.tzinfo is UTC for p in series["temperature_2m"].points)
    assert series["temperature_2m"].resolution_minutes == 60
    # null value dropped and counted
    assert series["temperature_2m"].metadata["nulls"] == 1
    assert len(series["temperature_2m"].points) == 2


def test_parse_historical_json_missing_hourly_raises() -> None:
    with pytest.raises(MalformedResponseError):
        parse_historical_json(b'{"hourly":{}}')


def test_parse_historical_json_not_json_raises() -> None:
    with pytest.raises(MalformedResponseError):
        parse_historical_json(b"<html>oops</html>")


def test_parse_historical_json_empty_time_raises() -> None:
    with pytest.raises(EmptyResponseError):
        parse_historical_json(b'{"hourly":{"time":[],"temperature_2m":[]}}')


def test_parse_historical_json_mismatched_column_raises() -> None:
    bad = json.dumps(
        {"hourly": {"time": ["2024-01-01T00:00"], "temperature_2m": [1.0, 2.0]}}
    ).encode()
    with pytest.raises(MalformedResponseError):
        parse_historical_json(bad)
