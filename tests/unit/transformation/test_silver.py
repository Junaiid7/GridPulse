"""Unit tests for Silver transformation: cleaning, deduplication and write."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from gridpulse.ingestion.common.models import DataPoint, TimeSeries
from gridpulse.transformation.silver import SilverRecord, clean_timeseries, read_silver, write_silver

UTC = timezone.utc


def _series(
    points: list[tuple[str, float]],
    *,
    tz: str = "UTC",
    resolution_minutes: int = 15,
) -> TimeSeries:
    return TimeSeries(
        source="entsoe",
        entity="actual-total-load",
        unit="MW",
        points=tuple(DataPoint(datetime.fromisoformat(t).astimezone(UTC), v) for t, v in points),
        resolution_minutes=resolution_minutes,
        tz=tz,
    )


def test_clean_normal_series_sorted_and_timelocal() -> None:
    s = _series([
        ("2024-01-01T11:45:00+00:00", 2.0),
        ("2024-01-01T12:00:00+00:00", 3.0),
        ("2024-01-01T12:15:00+00:00", 4.0),
    ])
    records, report = clean_timeseries(s)
    assert [r.timestamp_utc.isoformat() for r in records] == [
        "2024-01-01T11:45:00+00:00",
        "2024-01-01T12:00:00+00:00",
        "2024-01-01T12:15:00+00:00",
    ]
    # Local Amsterdam time for 12:00 UTC in January = 13:00 CET.
    assert records[1].timestamp_local.hour == 13
    assert records[1].timestamp_local.date().month == 1
    assert report.ok  # no errors; no gaps (3 points at 15-min intervals)


def test_clean_non_monotonic_input_reported_as_error() -> None:
    # Cleaning must not silently reinterpret an unsorted input series.
    s = _series([
        ("2024-01-01T12:15:00+00:00", 4.0),
        ("2024-01-01T12:00:00+00:00", 3.0),
    ])
    records, report = clean_timeseries(s)
    assert [r.timestamp_utc.isoformat() for r in records] == [
        "2024-01-01T12:00:00+00:00",
        "2024-01-01T12:15:00+00:00",
    ]
    assert any(i.code == "timestamp_order" for i in report.issues)
    assert not report.ok


def test_clean_duplicate_timestamps_dropped() -> None:
    s = _series([
        ("2024-01-01T12:00:00+00:00", 10.0),
        ("2024-01-01T12:00:00+00:00", 5.0),
        ("2024-01-01T12:30:00+00:00", 7.0),
    ])
    records, report = clean_timeseries(s)
    # First duplicate kept, second flagged in report; 2 Silver rows produced.
    assert len(records) == 2
    assert records[0].value == 10.0
    dup_warnings = [i for i in report.issues if i.code == "duplicate_dropped"]
    assert len(dup_warnings) == 1


def test_clean_gap_detected_as_warning() -> None:
    s = _series(
        [
            ("2024-01-01T12:00:00+00:00", 10.0),
            ("2024-01-01T13:00:00+00:00", 20.0),
        ],
        resolution_minutes=15,
    )
    _, report = clean_timeseries(s)
    gap_warnings = [i for i in report.issues if i.code == "timestamp_gap"]
    assert len(gap_warnings) == 1
    assert "3" in gap_warnings[0].message  # 3 missing 15-min intervals


def test_clean_rejects_non_utc_tz() -> None:
    s = _series([("2024-01-01T12:00:00+00:00", 1.0)], tz="CET")
    with pytest.raises(ValueError, match="UTC-aware"):
        clean_timeseries(s)


def test_clean_rejects_naive_timestamps() -> None:
    s = TimeSeries(
        source="entsoe",
        entity="test",
        unit="MW",
        points=(DataPoint(datetime(2024, 1, 1, 12, 0), 1.0),),  # naive!
        resolution_minutes=15,
        tz="UTC",
    )
    with pytest.raises(ValueError, match="UTC-aware"):
        clean_timeseries(s)


def test_clean_empty_series() -> None:
    s = _series([])
    records, report = clean_timeseries(s)
    assert records == []
    assert report.ok


def test_write_silver_creates_csv_and_meta(tmp_path: Path) -> None:
    _, report = clean_timeseries(_series([
        ("2024-01-01T12:00:00+00:00", 10.0),
        ("2024-01-01T12:15:00+00:00", 20.0),
    ]))
    records = [
        SilverRecord(
            timestamp_utc=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            timestamp_local=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
            value=10.0,
            unit="MW",
            source="entsoe",
            entity="actual-total-load",
        ),
    ]
    csv_path, meta_path = write_silver(tmp_path, records, "entsoe", "actual-total-load", "MW", report=report)
    assert csv_path.name == "actual-total-load.csv"
    assert meta_path.name == "actual-total-load.meta.json"
    assert csv_path.exists()
    assert meta_path.exists()


def test_read_silver_roundtrip(tmp_path: Path) -> None:
    records = [
        SilverRecord(
            timestamp_utc=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            timestamp_local=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
            value=10.5,
            unit="MW",
            source="entsoe",
            entity="actual-total-load",
        ),
    ]
    csv_path, _ = write_silver(tmp_path, records, "entsoe", "actual-total-load", "MW")
    rows = read_silver(csv_path)
    assert len(rows) == 1
    assert rows[0]["value"] == "10.5"
    assert rows[0]["source"] == "entsoe"


def test_silver_entity_under_source(tmp_path: Path) -> None:
    records = [
        SilverRecord(
            timestamp_utc=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
            timestamp_local=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
            value=10.0,
            unit="MW",
            source="ENTSO-E",
            entity="Actual-Total-Load",
        ),
    ]
    csv_path, _ = write_silver(tmp_path, records, "ENTSO-E", "Actual-Total-Load", "MW")
    # Path contains slugified lower-cased parts.
    assert "entso-e" in csv_path.parts
    assert "actual-total-load" in csv_path.parts