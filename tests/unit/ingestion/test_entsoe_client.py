"""Tests for the ENTSO-E client: parameter construction, chunking, the
element-limit halving fallback, empty-result detection, and key hygiene.
All HTTP is mocked; the fixture payloads are synthetic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import urlencode

import pytest

from gridpulse.ingestion.common.errors import (
    EmptyResponseError,
    HttpError,
    NoDataError,
)
from gridpulse.ingestion.common.http import HttpResponse
from gridpulse.ingestion.entsoe.client import EntsoeClient
from gridpulse.ingestion.entsoe.domains import DE_LU, NL

UTC = UTC


def _dt(s: str):
    return datetime.fromisoformat(s).astimezone(UTC)


class FakeEntsoeHttp:
    """Injectable stand-in for HttpClient that records every call."""

    def __init__(self, responder):
        self.calls: list[dict] = []
        self.responder = responder

    def get(self, url, *, params=None, headers=None):
        full_url = url + ("?" + urlencode(params) if params else "")
        self.calls.append({"url": full_url, "params": dict(params or {})})
        result = self.responder(self.calls[-1]["params"])
        # Echo the real request URL (with query string) so token masking is testable.
        return HttpResponse(result.status, result.headers, result.body, full_url)


def _ok(status=200, body: bytes = b"<ok/>"):
    return HttpResponse(
        status, {"content-type": "application/xml"}, body, "http://fake"
    )


def _client(fake_http) -> EntsoeClient:
    return EntsoeClient(api_key="TEST-SUPERSECRET", http=fake_http)


def test_fetch_load_builds_correct_params() -> None:
    fake = FakeEntsoeHttp(lambda params: _ok(body=b"<ok/>"))
    client = _client(fake)
    results = client.fetch_load(
        NL, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-01T12:00:00+00:00")
    )

    assert len(results) == 1
    params = fake.calls[0]["params"]
    assert params["securityToken"] == "TEST-SUPERSECRET"
    assert params["documentType"] == "A65"
    assert params["processType"] == "A16"
    assert params["out_Domain"] == NL.code
    assert params["outBiddingZone_Domain"] == NL.code
    assert params["periodStart"] == "202401010000"
    assert params["periodEnd"] == "202401011200"
    assert results[0].units == "MW"
    assert results[0].metadata["area"] == NL.code


def test_fetch_load_chunks_long_ranges() -> None:
    fake = FakeEntsoeHttp(lambda params: _ok(body=b"<ok/>"))
    client = _client(fake)
    # 2024 is a leap year: 01-01 .. 03-05 is 64 days -> 3 monthly chunks.
    results = client.fetch_load(
        NL, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-03-05T00:00:00+00:00")
    )

    assert len(results) == 3
    assert len(fake.calls) == 3
    starts = [c["params"]["periodStart"] for c in fake.calls]
    ends = [c["params"]["periodEnd"] for c in fake.calls]
    assert starts[0] == "202401010000"
    assert ends[-1] == "202403050000"
    # consecutive windows
    assert starts[1] == ends[0]
    assert starts[2] == ends[1]


def test_fetch_empty_reports_no_data_error() -> None:
    fake = FakeEntsoeHttp(lambda params: _ok(body=b"<ack>No matching data found</ack>"))
    client = _client(fake)
    with pytest.raises(NoDataError):
        client.fetch_load(
            NL, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-02T00:00:00+00:00")
        )


def test_fetch_halves_when_element_limit_reached() -> None:
    sent = {"over_limit": True}

    def responder(params):
        if sent["over_limit"]:
            sent["over_limit"] = False
            raise HttpError(
                "400",
                status=400,
                body=b"amount of requested data exceeds allowed limit",
            )
        return _ok(body=b"<ok/>")

    fake = FakeEntsoeHttp(responder)
    client = _client(fake)
    results = client.fetch_load(
        NL, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-02T00:00:00+00:00")
    )

    # original attempt + two halves
    assert len(fake.calls) == 3
    assert len(results) == 2
    halves = [c["params"]["periodStart"] for c in fake.calls[1:]]
    assert halves[0] == "202401010000"
    assert halves[1] == "202401011200"  # midpoint


def test_fetch_propagates_transient_http_errors() -> None:
    fake = FakeEntsoeHttp(
        lambda params: _raise(HttpError("500", status=500, body=b"boom"))
    )
    client = _client(fake)
    with pytest.raises(HttpError):
        client.fetch_load(
            NL, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-02T00:00:00+00:00")
        )


def test_fetch_stored_url_never_contains_api_key() -> None:
    fake = FakeEntsoeHttp(lambda params: _ok(body=b"<ok/>"))
    client = _client(fake)
    results = client.fetch_load(
        NL, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-01T01:00:00+00:00")
    )
    assert "TEST-SUPERSECRET" not in results[0].url
    assert "redacted" in results[0].url


def test_day_ahead_prices_params_and_sequence() -> None:
    nl_fake = FakeEntsoeHttp(lambda params: _ok(body=b"<ok/>"))
    nl_client = _client(nl_fake)
    nl_client.fetch_day_ahead_prices(
        NL, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-02T00:00:00+00:00")
    )
    assert nl_fake.calls[0]["params"]["documentType"] == "A44"
    assert nl_fake.calls[0]["params"]["in_Domain"] == NL.code
    assert nl_fake.calls[0]["params"]["contract_MarketAgreement.type"] == "A01"
    assert (
        "classificationSequence_AttributeInstanceComponent.position"
        not in nl_fake.calls[0]["params"]
    )

    de_fake = FakeEntsoeHttp(lambda params: _ok(body=b"<ok/>"))
    _client(de_fake).fetch_day_ahead_prices(
        DE_LU, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-02T00:00:00+00:00")
    )
    seq = de_fake.calls[0]["params"][
        "classificationSequence_AttributeInstanceComponent.position"
    ]
    assert seq == 1


def test_crossborder_flows_direction_encoding() -> None:
    fake = FakeEntsoeHttp(lambda params: _ok(body=b"<ok/>"))
    client = _client(fake)
    results = client.fetch_crossborder_flows(
        NL, DE_LU, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-02T00:00:00+00:00")
    )

    params = fake.calls[0]["params"]
    # Positive flow = from out (NL) into in (DE_LU).
    assert params["documentType"] == "A11"
    assert params["out_Domain"] == NL.code
    assert params["in_Domain"] == DE_LU.code
    assert results[0].entity == "cross-border-physical-flows:nl-de_lu"
    assert results[0].units == "MW"


def test_generation_psr_filter() -> None:
    fake = FakeEntsoeHttp(lambda params: _ok(body=b"<ok/>"))
    client = _client(fake)
    client.fetch_generation(
        NL,
        _dt("2024-01-01T00:00:00+00:00"),
        _dt("2024-01-01T01:00:00+00:00"),
        psr_type="B16",
    )
    params = fake.calls[0]["params"]
    assert params["documentType"] == "A75"
    assert params["psrType"] == "B16"


def test_generation_rejects_unknown_psr() -> None:
    fake = FakeEntsoeHttp(lambda params: _ok(body=b"<ok/>"))
    client = _client(fake)
    with pytest.raises(ValueError):
        client.fetch_generation(
            NL,
            _dt("2024-01-01T00:00:00+00:00"),
            _dt("2024-01-01T01:00:00+00:00"),
            psr_type="Z99",
        )


def test_constructor_requires_key() -> None:
    from gridpulse.ingestion.common.errors import ConfigurationError

    with pytest.raises(ConfigurationError):
        EntsoeClient(api_key="   ")


def test_empty_body_is_empty_response_error() -> None:
    fake = FakeEntsoeHttp(
        lambda params: HttpResponse(
            200, {"content-type": "application/xml"}, b"", "http://fake"
        )
    )
    client = _client(fake)
    with pytest.raises(EmptyResponseError):
        client.fetch_load(
            NL, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-01T01:00:00+00:00")
        )


def test_fetch_load_forecast_uses_dayahead_process() -> None:
    fake = FakeEntsoeHttp(lambda params: _ok(body=b"<ok/>"))
    _client(fake).fetch_load_forecast(
        NL, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-01T01:00:00+00:00")
    )
    assert fake.calls[0]["params"]["processType"] == "A01"


def test_fetch_imbalance_prices_sends_a85() -> None:
    fake = FakeEntsoeHttp(lambda params: _ok(body=b"<ok/>"))
    client = _client(fake)
    results = client.fetch_imbalance_prices(
        NL, _dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-01T01:00:00+00:00")
    )
    assert fake.calls[0]["params"]["documentType"] == "A85"
    assert fake.calls[0]["params"]["controlArea_Domain"] == NL.code
    assert results[0].units == "EUR/MWh"


def _raise(exc):
    raise exc
