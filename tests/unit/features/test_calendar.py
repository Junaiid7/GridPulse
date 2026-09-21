"""Unit tests for calendar features derived from target hour."""

from __future__ import annotations

from datetime import UTC, datetime

from gridpulse.features.calendar import calendar_features
from gridpulse.features.holiday import NetherlandsHolidayCalendar

UTC = UTC


def test_winter_hour_and_day_of_week() -> None:
    # 2024-01-15 23:30 UTC = 2024-01-16 00:30 CET (Tuesday).
    feats = calendar_features(datetime(2024, 1, 15, 23, 30, tzinfo=UTC))
    assert feats["hour"] == 0
    assert feats["day_of_week"] == 1  # Tuesday
    assert feats["day_of_year"] == 16
    assert feats["month"] == 1
    assert feats["is_weekend"] is False


def test_weekend_true_on_saturday() -> None:
    # 2024-01-13 is a Saturday. UTC 12:00 = 13:00 CET.
    feats = calendar_features(datetime(2024, 1, 13, 12, 0, tzinfo=UTC))
    assert feats["is_weekend"] is True


def test_summer_hour_cest() -> None:
    # 2024-07-01 00:00 UTC = 02:00 CEST.
    feats = calendar_features(datetime(2024, 7, 1, 0, 0, tzinfo=UTC))
    assert feats["hour"] == 2


def test_holiday_calendar_detected() -> None:
    # 2024-12-25 12:00 UTC = 13:00 CET, Christmas Day.
    feats = calendar_features(
        datetime(2024, 12, 25, 12, 0, tzinfo=UTC),
        holiday_calendar=NetherlandsHolidayCalendar(),
    )
    assert feats["is_holiday"] is True


def test_no_holiday_calendar_no_key() -> None:
    feats = calendar_features(datetime(2024, 12, 25, 12, 0, tzinfo=UTC))
    assert "is_holiday" not in feats


def test_day_of_year_in_leap_year() -> None:
    # 2024 is a leap year; Feb 29 is day 60.
    feats = calendar_features(datetime(2024, 2, 29, 12, 0, tzinfo=UTC))
    assert feats["day_of_year"] == 60
    assert feats["month"] == 2
