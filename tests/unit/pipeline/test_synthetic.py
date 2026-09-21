"""Unit tests for synthetic pipeline and fixture generation."""

from __future__ import annotations

import pytest

from gridpulse.ingestion.common.errors import NoDataError
from gridpulse.ingestion.entsoe.domains import NL
from gridpulse.ingestion.weather.variables import Location
from gridpulse.pipeline.synthetic import (
    DEFAULT_END,
    DEFAULT_START,
    SyntheticEntsoeClient,
    SyntheticImbalanceSource,
    SyntheticOpenMeteoClient,
    build_synthetic_window,
)


def test_build_synthetic_window() -> None:
    fixtures = build_synthetic_window(DEFAULT_START, DEFAULT_END)
    assert fixtures.start == DEFAULT_START
    assert fixtures.end == DEFAULT_END
    assert "load" in fixtures.payloads
    assert "generation" in fixtures.payloads
    assert "prices" in fixtures.payloads
    assert "weather" in fixtures.payloads
    assert len(fixtures.timestamps) > 0
    assert len(fixtures.per_hour) == len(fixtures.timestamps)

    sig = fixtures.signals_at(fixtures.timestamps[0])
    assert "load_mw" in sig
    assert "day_ahead_price_eur_mwh" in sig


def test_synthetic_entsoe_client() -> None:
    fixtures = build_synthetic_window(DEFAULT_START, DEFAULT_END)
    client = SyntheticEntsoeClient(fixtures)
    load_res = client.fetch_load(NL, DEFAULT_START, DEFAULT_END)
    assert len(load_res) == 1
    assert load_res[0].source == "entsoe"

    gen_res = client.fetch_generation(NL, DEFAULT_START, DEFAULT_END)
    assert len(gen_res) == 1

    price_res = client.fetch_day_ahead_prices(NL, DEFAULT_START, DEFAULT_END)
    assert len(price_res) == 1

    with pytest.raises(NoDataError):
        client.fetch_imbalance_prices(NL, DEFAULT_START, DEFAULT_END)


def test_synthetic_open_meteo_client() -> None:
    fixtures = build_synthetic_window(DEFAULT_START, DEFAULT_END)
    client = SyntheticOpenMeteoClient(fixtures)
    loc = Location("nl-central", 52.21, 5.29)
    res = client.fetch_historical(loc, DEFAULT_START.date(), DEFAULT_START.date())
    assert len(res) == 1
    assert res[0].source == "open-meteo"


def test_synthetic_imbalance_source() -> None:
    source = SyntheticImbalanceSource()
    assert source.availability["available"] is False
    with pytest.raises(NoDataError):
        source.fetch(NL, DEFAULT_START, DEFAULT_END)
