"""Unit tests for shared ingestion machinery: time ranges, chunking, HTTP
behaviour (retries, auth/rate-limit mapping, log sanitisation), validation,
and deterministic Bronze storage. No live network.
"""

from __future__ import annotations

import io
import json
import logging
import time
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gridpulse.ingestion.common import errors as err
from gridpulse.ingestion.common.http import HttpClient, mask_url
from gridpulse.ingestion.common.models import (
    DataPoint,
    FetchResult,
    TimeRange,
    TimeSeries,
    chunk_range,
    ensure_utc,
)
from gridpulse.ingestion.common.storage import write_bronze
from gridpulse.ingestion.common.validation import (
    Issue,
    ValidationReport,
    aggregate,
    validate_non_negative,
    validate_required_fields,
    validate_timestamps,
    validate_units,
)

UTC = timezone.utc


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)


class _FakeResponse:
    def __init__(self, status: int = 200, body: bytes = b"", headers: dict | None = None):
        self.status = status
        self.headers = headers or {}
        self._body = body

    def read(self) -> bytes:
        return self._body


class _RecordingOpener:
    """Captures requests and delegates each to a callable script."""

    def __init__(self, script):
        self.calls: list[dict] = []
        self.script = script

    def open(self, request, timeout=None):
        self.calls.append({"url": request.full_url, "headers": dict(request.header_items()), "timeout": timeout})
        return self.script(self.calls[-1])


def _http_error(code: int, body: bytes = b"", headers: dict | None = None):
    return urllib.error.HTTPError("http://fake", code, "err", headers or {}, io.BytesIO(body))


# ---------------------------------------------------------------- time ranges

def test_chunk_range_splits_to_max_span() -> None:
    start = _dt("2024-01-01T00:00:00+00:00")
    chunks = chunk_range(start, start + timedelta(hours=6), timedelta(hours=2))
    assert [c.start for c in chunks] == [
        _dt("2024-01-01T00:00:00+00:00"),
        _dt("2024-01-01T02:00:00+00:00"),
        _dt("2024-01-01T04:00:00+00:00"),
    ]
    assert chunks[-1].end == start + timedelta(hours=6)
    assert all((c.end - c.start) <= timedelta(hours=2) for c in chunks)


def test_chunk_range_exact_multiple_has_no_trailing_empty() -> None:
    start = _dt("2024-01-01T00:00:00+00:00")
    chunks = chunk_range(start, start + timedelta(hours=4), timedelta(hours=2))
    assert len(chunks) == 2
    assert chunks[-1].end == start + timedelta(hours=4)


def test_time_range_requires_aware_datetimes() -> None:
    with pytest.raises(ValueError):
        TimeRange(datetime(2024, 1, 1), _dt("2024-01-02T00:00:00+00:00"))


def test_time_range_rejects_backwards() -> None:
    with pytest.raises(ValueError):
        TimeRange(_dt("2024-01-02T00:00:00+00:00"), _dt("2024-01-01T00:00:00+00:00"))


def test_ensure_utc_tags_naive_and_converts() -> None:
    assert ensure_utc(datetime(2024, 1, 1)).tzinfo is UTC
    tagged = ensure_utc(_dt("2024-01-01T02:00:00+02:00"))
    assert tagged.hour == 0  # converted back to UTC


def test_time_range_split_halves() -> None:
    left, right = TimeRange(_dt("2024-01-01T00:00:00+00:00"), _dt("2024-01-02T00:00:00+00:00")).split()
    assert left.end == right.start
    assert (left.end - left.start) == timedelta(hours=12)


# ---------------------------------------------------------------- http client

def test_http_get_success() -> None:
    opener = _RecordingOpener(
        lambda call: _FakeResponse(200, b'{"ok": true}', {"Content-Type": "application/json; charset=utf-8"})
    )
    client = HttpClient(timeout=10, retries=2, backoff=0.1, opener=opener)
    resp = client.get("https://example.com/api", params={"a": 1})
    assert resp.status == 200
    assert resp.body == b'{"ok": true}'
    assert resp.text == '{"ok": true}'
    assert "a=1" in opener.calls[0]["url"]


def test_http_get_404_is_http_error() -> None:
    opener = _RecordingOpener(lambda call: _raise(_http_error(404, b"missing")))
    client = HttpClient(retries=2, backoff=0.1, opener=opener)
    with pytest.raises(err.HttpError) as excinfo:
        client.get("https://example.com/api")
    assert excinfo.value.status == 404
    assert len(opener.calls) == 1  # 404 is never retried


def test_http_get_401_is_auth_error_and_no_retry() -> None:
    opener = _RecordingOpener(lambda call: _raise(_http_error(401, b"Auth failed")))
    client = HttpClient(retries=2, backoff=0.1, opener=opener)
    with pytest.raises(err.AuthError) as excinfo:
        client.get("https://example.com/api")
    assert excinfo.value.status == 401
    assert len(opener.calls) == 1


def test_http_get_429_raises_rate_limit_with_retry_after_and_retries() -> None:
    opener = _RecordingOpener(lambda call: _raise(_http_error(429, b"slow down", {"Retry-After": "1"})))
    client = HttpClient(retries=3, backoff=0.1, opener=opener)
    with pytest.raises(err.RateLimitError) as excinfo:
        client.get("https://example.com/api")
    assert excinfo.value.status == 429
    assert excinfo.value.retry_after == 1
    assert len(opener.calls) == 3


def test_http_retries_transient_500_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    state = {"n": 0}

    def script(call):
        state["n"] += 1
        if state["n"] == 1:
            return _raise(_http_error(500, b"boom"))
        return _FakeResponse(200, b"ok")

    client = HttpClient(retries=3, backoff=0.1, opener=_RecordingOpener(script))
    resp = client.get("https://example.com/api")
    assert resp.status == 200
    assert state["n"] == 2


def test_http_retries_connection_error_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)
    state = {"n": 0}

    def script(call):
        state["n"] += 1
        if state["n"] == 1:
            raise ConnectionError("reset")
        return _FakeResponse(200, b"ok")

    client = HttpClient(retries=3, backoff=0.1, opener=_RecordingOpener(script))
    assert client.get("https://example.com/api").status == 200
    assert state["n"] == 2


def test_http_exhausted_connection_raises(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _: None)

    def script(call):
        raise ConnectionError("always down")

    client = HttpClient(retries=2, backoff=0.1, opener=_RecordingOpener(script))
    with pytest.raises(err.HttpError):
        client.get("https://example.com/api")


def test_http_logs_never_contain_api_key(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="gridpulse.ingestion.http")
    opener = _RecordingOpener(lambda call: _FakeResponse(200, b"ok"))
    client = HttpClient(retries=1, backoff=0.1, opener=opener)
    client.get("https://example.com/api", params={"securityToken": "SUPERSECRETKEY", "a": "1"})
    assert "SUPERSECRETKEY" not in caplog.text
    assert "redacted" in caplog.text


def test_mask_url_redacts_token_but_not_other_params() -> None:
    masked = mask_url("https://example.com/api?securityToken=SECRET&documentType=A65")
    assert "SECRET" not in masked
    assert "redacted" in masked
    assert "documentType=A65" in masked


# ---------------------------------------------------------------- validation

def _series_from(values, freq_min=60, unit="MW"):
    start = _dt("2024-01-01T00:00:00+00:00")
    points = tuple(DataPoint(start + timedelta(minutes=freq_min * i), v) for i, v in enumerate(values))
    return TimeSeries(source="test", entity="x", unit=unit, points=points, resolution_minutes=freq_min)


def test_validate_timestamps_catches_duplicates_and_order() -> None:
    # Craft series with actual duplicate *timestamps* (same ts, different values).
    base = _dt("2024-01-01T00:00:00+00:00")
    dup_points = (
        DataPoint(base, 1.0),
        DataPoint(base, 2.0),  # same timestamp as above → duplicate
        DataPoint(base + timedelta(hours=1), 3.0),
    )
    dup = TimeSeries(source="test", entity="x", unit="MW", points=dup_points, resolution_minutes=60)
    report = validate_timestamps(dup)
    assert any(i.code == "timestamp_duplicate" for i in report.issues)
    assert not report.ok

    # Craft series with non-monotonic timestamps.
    order_points = (
        DataPoint(base + timedelta(hours=2), 1.0),
        DataPoint(base, 2.0),  # earlier than previous
        DataPoint(base + timedelta(hours=1), 3.0),
    )
    unordered = TimeSeries(source="test", entity="x", unit="MW", points=order_points, resolution_minutes=15)
    assert any(i.code == "timestamp_order" for i in validate_timestamps(unordered).issues)


def test_validate_timestamps_warns_on_gaps() -> None:
    start = _dt("2024-01-01T00:00:00+00:00")
    points = (DataPoint(start, 1.0), DataPoint(start + timedelta(hours=3), 2.0))  # 3h gap vs 1h resolution
    series = TimeSeries(source="test", entity="x", unit="MW", points=points, resolution_minutes=60)
    report = validate_timestamps(series)
    assert any(i.code == "timestamp_gap" for i in report.warnings)
    assert report.ok  # gaps are warnings, not errors


def test_validate_non_negative_flags_impossible_negatives() -> None:
    series = _series_from([10.0, -1.0, 200.0], unit="W/m²")
    report = validate_non_negative(series, reason="radiation cannot be negative")
    assert any(i.code == "negative_value" for i in report.errors)
    assert not report.ok


def test_validate_required_fields_and_units() -> None:
    assert validate_required_fields({"a": 1, "b": "x"}, ["a", "b"]).ok
    missing = validate_required_fields({"a": None}, ["a", "b"])
    assert any(i.code == "missing_field" for i in missing.errors)
    assert validate_units(
        TimeSeries(source="t", entity="prices", unit="EUR/MWh", points=(), resolution_minutes=60), "EUR/MWh"
    ).ok
    bad_units = validate_units(_series_from([1.0], unit="GW"), "MW")
    assert any(i.code == "unit_mismatch" for i in bad_units.errors)


def test_aggregate_combines_reports() -> None:
    # Series with duplicate timestamps.
    base = _dt("2024-01-01T00:00:00+00:00")
    dup_points = (DataPoint(base, 1.0), DataPoint(base, 2.0))
    dup_series = TimeSeries(source="test", entity="x", unit="MW", points=dup_points, resolution_minutes=60)
    combined = aggregate([validate_timestamps(dup_series), validate_required_fields({}, ["x"])])
    assert combined.errors
    assert "timestamp_duplicate" in {i.code for i in combined.issues}
    assert "missing_field" in {i.code for i in combined.issues}


# ---------------------------------------------------------------- bronze storage

def _fetch_result(**overrides) -> FetchResult:
    base = dict(
        source="entsoe",
        entity="actual-total-load",
        start=_dt("2024-01-01T00:00:00+00:00"),
        end=_dt("2024-01-02T00:00:00+00:00"),
        retrieved_at=_dt("2026-09-14T08:00:00+00:00"),
        payload=b"<xml/>",
        content_type="application/xml",
        units="MW",
        identifiers=("10YNL----------L",),
        timezone="UTC",
    )
    base.update(overrides)
    return FetchResult(**base)


def test_bronze_paths_are_deterministic(tmp_path: Path) -> None:
    a = _fetch_result()
    b = _fetch_result()
    wa = write_bronze(tmp_path, a)
    wb = write_bronze(tmp_path, b)
    assert wa.payload_path == wb.payload_path
    assert wa.manifest_path == wb.manifest_path


def test_bronze_layout_source_entity_date(tmp_path: Path) -> None:
    result = _fetch_result(retrieved_at=_dt("2026-09-14T08:00:00+00:00"))
    write = write_bronze(tmp_path, result)
    assert write.payload_path.parent == tmp_path / "entsoe" / "actual-total-load" / "20240101__20240102"
    assert write.payload_path.suffix == ".xml"
    assert write.manifest_path.exists()
    assert (write.payload_path.read_bytes()) == result.payload


def test_bronze_manifest_metadata(tmp_path: Path) -> None:
    result = _fetch_result(
        retrieved_at=_dt("2026-09-14T08:00:00+00:00"),
        url="https://web-api.tp.entsoe.eu/api",
    )
    write = write_bronze(
        tmp_path,
        result,
        validation=ValidationReport((Issue("warning", "timestamp_gap", "x"),)),
    )
    manifest = json.loads(write.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "entsoe"
    assert manifest["entity"] == "actual-total-load"
    assert manifest["requested_window"]["start_utc"] == "2024-01-01T00:00:00+00:00"
    assert manifest["timezone"] == "UTC"
    assert manifest["units"] == "MW"
    assert manifest["identifiers"] == ["10YNL----------L"]
    assert manifest["payload"]["sha256"] == result.sha256
    assert manifest["validation"] == {"errors": 0, "warnings": 1}


def test_bronze_slug_for_flow_entity(tmp_path: Path) -> None:
    result = _fetch_result(entity="cross-border-physical-flows:nl-de_lu")
    write = write_bronze(tmp_path, result)
    assert "nl-de-lu" in str(write.payload_path.parent)


def test_bronze_extension_by_content_type(tmp_path: Path) -> None:
    json_result = _fetch_result(payload=b"{}", content_type="application/json")
    assert write_bronze(tmp_path, json_result).payload_path.suffix == ".json"
    zip_result = _fetch_result(payload=b"PK", content_type="application/zip")
    assert write_bronze(tmp_path, zip_result).payload_path.suffix == ".zip"


def _raise(exc):
    raise exc