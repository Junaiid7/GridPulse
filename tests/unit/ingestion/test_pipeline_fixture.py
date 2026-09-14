"""Offline integration-style test: full ENTSO-E and Open-Meteo fetches driven
by stored synthetic fixtures (not the live internet), running
request -> parse -> validate -> Bronze write end to end.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

from gridpulse.ingestion.common.http import HttpResponse
from gridpulse.ingestion.common.validation import aggregate, validate_required_fields, validate_timestamps
from gridpulse.ingestion.entsoe.client import EntsoeClient
from gridpulse.ingestion.entsoe.domains import NL
from gridpulse.ingestion.entsoe.parser import parse_load, parse_day_ahead_prices
from gridpulse.ingestion.common.storage import write_bronze
from gridpulse.ingestion.weather.open_meteo import OpenMeteoClient
from gridpulse.ingestion.weather.parser import parse_historical_json
from gridpulse.ingestion.weather.variables import Location

UTC = timezone.utc

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


class FixtureHttp:
    """Returns a canned payload from a lookup table, recording calls."""

    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.calls: list[dict] = []

    def get(self, url, *, params=None, headers=None):
        self.calls.append(dict(params or {}))
        content_type = "application/xml" if "tp.entsoe" in url else "application/json"
        body = self.payloads[url]
        return HttpResponse(200, {"content-type": content_type}, body, "http://fake")


def test_entsoe_load_fixture_full_pipeline(tmp_path: Path) -> None:
    xml_payload = (FIXTURES / "entsoe_actual_load.xml").read_bytes()
    http = FixtureHttp({"https://web-api.tp.entsoe.eu/api": xml_payload})
    client = EntsoeClient(api_key="fixture-key", http=http)

    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, 2, 0, tzinfo=UTC)
    results = client.fetch_load(NL, start, end)
    assert len(results) == 1

    series = parse_load(results[0].payload)
    assert len(series.points) == 8  # 2h at 15-min resolution
    assert series.resolution_minutes == 15
    assert series.unit == "MW"

    report = aggregate([validate_timestamps(series), validate_required_fields(series.metadata, ["outBiddingZone_Domain.mRID"])])
    assert report.ok

    write = write_bronze(tmp_path / "bronze", results[0], validation=report)
    assert write.payload_path.exists()
    manifest = json.loads(write.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "entsoe"
    assert manifest["validation"] == {"errors": 0, "warnings": 0}
    assert manifest["payload"]["sha256"] == results[0].sha256


def test_entsoe_prices_fixture_pipeline(tmp_path: Path) -> None:
    xml_payload = (FIXTURES / "entsoe_dayahead_prices.xml").read_bytes()
    http = FixtureHttp({"https://web-api.tp.entsoe.eu/api": xml_payload})
    client = EntsoeClient(api_key="fixture-key", http=http)

    results = client.fetch_day_ahead_prices(NL, datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 1, 4, 0, tzinfo=UTC))
    series = parse_day_ahead_prices(results[0].payload)
    assert series.unit == "EUR/MWh"
    assert [p.value for p in series.points] == [45.20, 48.70, 55.05, 39.80]


def test_open_meteo_fixture_full_pipeline(tmp_path: Path) -> None:
    json_payload = (FIXTURES / "open_meteo_hourly.json").read_bytes()
    http = FixtureHttp({"https://archive-api.open-meteo.com/v1/archive": json_payload})
    client = OpenMeteoClient(http=http)

    centre = Location("nl-central", 52.21, 5.29)
    results = client.fetch_historical(centre, date(2024, 1, 1), date(2024, 1, 1))
    assert len(results) == 1

    series = parse_historical_json(results[0].payload)
    assert "wind_speed_100m" in series
    assert series["wind_speed_100m"].unit == "m/s"

    report = aggregate([validate_timestamps(series["wind_speed_100m"])])
    assert report.ok

    write = write_bronze(tmp_path / "bronze", results[0], validation=report)
    assert write.payload_path.suffix == ".json"
    manifest = json.loads(write.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "open-meteo"
    assert manifest["requested_window"]["start_utc"] == "2024-01-01T00:00:00+00:00"
    assert manifest["payload"]["size_bytes"] == len(json_payload)