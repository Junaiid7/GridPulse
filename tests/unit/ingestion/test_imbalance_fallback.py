"""Tests enforcing that the NL imbalance-price gap is never silently filled.

Two invariants: (1) the real ENTSO-E imbalance source records availability as
unverified and always goes through the A85 endpoint; (2) the clearly
separated synthetic fallback is deliberately not implemented and raises.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from gridpulse.ingestion.common.errors import ConfigurationError
from gridpulse.ingestion.common.http import HttpResponse
from gridpulse.ingestion.entsoe.client import EntsoeClient
from gridpulse.ingestion.entsoe.domains import NL
from gridpulse.ingestion.entsoe.imbalance import EntsoeImbalancePrices, ImbalancePricesSource
from gridpulse.ingestion.fallback.imbalance import ImbalancePenaltyProvider, SyntheticImbalancePenalty

UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 1, 2, tzinfo=UTC)


class FakeHttp:
    def get(self, url, *, params=None, headers=None):
        return HttpResponse(200, {"content-type": "application/zip"}, b"PK", "http://fake")


def test_synthetic_fallback_is_explicitly_not_implemented() -> None:
    penalty = SyntheticImbalancePenalty()
    assert penalty.available is False
    with pytest.raises(NotImplementedError, match="intentionally not implemented"):
        penalty.fetch(NL, START, END)


def test_synthetic_fallback_satisfies_provider_protocol() -> None:
    penalty = SyntheticImbalancePenalty()
    assert isinstance(penalty, ImbalancePenaltyProvider)


def test_entsoe_imbalance_declares_nl_unverified() -> None:
    client = EntsoeClient(api_key="test-key", http=FakeHttp())
    source = EntsoeImbalancePrices(client)
    assert isinstance(source, ImbalancePricesSource)
    assert "UNVERIFIED" in source.availability["nl_coverage"]


def test_entsoe_imbalance_queries_a85() -> None:
    client = EntsoeClient(api_key="test-key", http=FakeHttp())
    results = client.fetch_imbalance_prices(NL, START, END)
    assert results[0].units == "EUR/MWh"
    assert results[0].metadata["document_type"] == "A85"


def test_no_key_is_configuration_error() -> None:
    with pytest.raises(ConfigurationError):
        EntsoeClient(api_key="")