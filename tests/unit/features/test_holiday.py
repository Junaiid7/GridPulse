"""Unit tests for the HolidayCalendar implementations."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from gridpulse.features.holiday import (
    CsvHolidayCalendar,
    NetherlandsHolidayCalendar,
    easter_sunday,
)


# --- Gregorian Easter (Meeus–Jones–Butcher) ---

def test_easter_sunday_known_years() -> None:
    assert easter_sunday(2024) == date(2024, 3, 31)
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2026) == date(2026, 4, 5)
    assert easter_sunday(2012) == date(2012, 4, 8)


# --- Netherlands public holidays ---

_cal = NetherlandsHolidayCalendar()


def test_new_year() -> None:
    assert _cal.is_holiday(date(2024, 1, 1))


def test_christmas_and_boxing_day() -> None:
    assert _cal.is_holiday(date(2024, 12, 25))
    assert _cal.is_holiday(date(2024, 12, 26))


def test_liberation_day_approximation() -> None:
    assert _cal.is_holiday(date(2025, 5, 5))


def test_good_friday_not_included() -> None:
    assert not _cal.is_holiday(date(2024, 3, 29))


def test_kings_day_sunday_observed_saturday() -> None:
    # 2014-04-27 is a Sunday → King's Day observed on 2014-04-26.
    assert _cal.is_holiday(date(2014, 4, 26))


def test_kings_day_normal_monday() -> None:
    # 2015-04-27 is a Monday → King's Day stays on the 27th.
    assert _cal.is_holiday(date(2015, 4, 27))


def test_holidays_between_inclusive() -> None:
    # Easter Sunday 2024 = Mar 31; Easter Monday 2024 = Apr 1.
    result = _cal.holidays_between(date(2024, 3, 30), date(2024, 4, 1))
    assert result == {date(2024, 3, 31), date(2024, 4, 1)}


def test_as_set_is_non_empty() -> None:
    s = _cal.as_set()
    assert len(s) > 100
    assert date(2024, 1, 1) in s


# --- CsvHolidayCalendar ---

def test_csv_calendar_loads_and_filters(tmp_path) -> None:
    csv_path = tmp_path / "holidays.csv"
    csv_path.write_text(
        "# Dutch holidays 2024\n"
        "\n"
        "2024-01-01\n"
        "2024-12-25, Christmas Day\n"
        "2024-12-26\n"
        "bad-date-line\n",
        encoding="utf-8",
    )
    cal = CsvHolidayCalendar(csv_path)
    assert cal.is_holiday(date(2024, 1, 1)) is True
    assert cal.is_holiday(date(2024, 12, 25)) is True
    assert cal.is_holiday(date(2024, 12, 27)) is False
    assert cal.holidays_between(date(2024, 12, 20), date(2024, 12, 26)) == {
        date(2024, 12, 25),
        date(2024, 12, 26),
    }
    assert cal.as_set() == {date(2024, 1, 1), date(2024, 12, 25), date(2024, 12, 26)}


def test_csv_calendar_blank_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "holidays.csv"
    csv_path.write_text("", encoding="utf-8")
    cal = CsvHolidayCalendar(csv_path)
    assert cal.as_set() == set()
    assert cal.is_holiday(date(2024, 1, 1)) is False