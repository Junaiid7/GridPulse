"""Unit tests for the transformation-layer timezone/DST policy helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from gridpulse.transformation.times import (
    AMSTERDAM_TZ,
    CEST_OFFSET_MIN,
    CET_OFFSET_MIN,
    local_date,
    local_hour,
    local_time,
    utc_offset_minutes,
)

UTC = UTC


def test_local_time_winter_cet() -> None:
    # UTC 23:30 in January -> next day 00:30 Amsterdam (UTC+1, CET).
    utc = datetime(2024, 1, 15, 23, 30, tzinfo=UTC)
    local = local_time(utc)
    assert local.tzinfo == AMSTERDAM_TZ
    assert local.hour == 0
    assert local.date().day == 16
    assert local.utcoffset().total_seconds() == CET_OFFSET_MIN * 60


def test_local_time_summer_cest() -> None:
    # UTC 22:00 in July -> next day 00:00 Amsterdam (UTC+2, CEST).
    utc = datetime(2024, 7, 15, 22, 0, tzinfo=UTC)
    local = local_time(utc)
    assert local.hour == 0
    assert local.date().day == 16
    assert local.utcoffset().total_seconds() == CEST_OFFSET_MIN * 60


def test_utc_offset_minutes_winter_and_summer() -> None:
    assert utc_offset_minutes(datetime(2024, 1, 15, 12, tzinfo=UTC)) == CET_OFFSET_MIN
    assert utc_offset_minutes(datetime(2024, 7, 15, 12, tzinfo=UTC)) == CEST_OFFSET_MIN


def test_dst_spring_forward() -> None:
    # 2024-03-31: at 01:00 UTC Amsterdam jumps CET -> CEST.
    assert (
        utc_offset_minutes(datetime(2024, 3, 31, 0, 30, tzinfo=UTC)) == CET_OFFSET_MIN
    )
    assert (
        utc_offset_minutes(datetime(2024, 3, 31, 1, 30, tzinfo=UTC)) == CEST_OFFSET_MIN
    )


def test_dst_fall_back() -> None:
    # 2024-10-27: at 01:00 UTC Amsterdam drops CEST -> CET.
    assert (
        utc_offset_minutes(datetime(2024, 10, 27, 0, 30, tzinfo=UTC)) == CEST_OFFSET_MIN
    )
    assert (
        utc_offset_minutes(datetime(2024, 10, 27, 1, 30, tzinfo=UTC)) == CET_OFFSET_MIN
    )


def test_local_date_across_midnight() -> None:
    assert (
        local_date(datetime(2024, 1, 1, 23, 0, tzinfo=UTC))
        == datetime(2024, 1, 2).date()
    )


def test_local_hour_boundary() -> None:
    assert (
        local_hour(datetime(2024, 7, 1, 0, 0, tzinfo=UTC)) == 2
    )  # UTC midnight -> 02:00 CEST
    assert (
        local_hour(datetime(2024, 1, 1, 23, 0, tzinfo=UTC)) == 0
    )  # UTC 23:00 -> 00:00 CET next day
