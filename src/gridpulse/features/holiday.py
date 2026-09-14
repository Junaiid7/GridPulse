"""Holiday features: a clean interface plus data-driven implementations.

The forecast model must never hardcode a random list of Dutch holidays. The
interface is :class:`HolidayCalendar` (``is_holiday`` / ``holidays_between``);
exactly two implementations ship:

- :class:`CsvHolidayCalendar` — the *data-driven* path: reads ISO dates from a
  CSV file (one date per line), so the holiday set lives in configuration,
  not model code;
- :class:`NetherlandsHolidayCalendar` — a minimal, deterministic built-in for
  the standard Dutch public holidays, computed from the Gregorian Computus.
  It is an approximation: it assumes Liberation Day (5 May) every year (Dutch
  law marks it every five years), and does not include Good Friday (not an
  official Dutch public holiday). Avoid it for production — point
  ``GRIDPULSE_HOLIDAYS_CSV`` at authoritative data instead.

All calendars are frozen and cheap to construct; date comparisons are
timezone-free (calendar dates).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Protocol, Set


class HolidayCalendar(Protocol):
    def is_holiday(self, day: date) -> bool: ...
    def holidays_between(self, start: date, end: date) -> Set[date]: ...
    def as_set(self) -> Set[date]: ...


def easter_sunday(year: int) -> date:
    """Gregorian Easter (Meeus–Jones–Butcher / Gauss algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@dataclass(frozen=True)
class CsvHolidayCalendar:
    """Reading holidays from a data file (one ISO date per line, or CSV).

    Lines are split on ``,``/whitespace and the first field of each being a
    valid ``YYYY-MM-DD`` date is used. Blank lines and ``#`` comments ignored.
    """

    path: Path

    def _load(self) -> Set[date]:
        days: Set[date] = set()
        with self.path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                token = line.split(",")[0].strip().split()[0]
                try:
                    days.add(date.fromisoformat(token))
                except ValueError:
                    continue
        return days

    def is_holiday(self, day: date) -> bool:
        return day in self._load()

    def holidays_between(self, start: date, end: date) -> Set[date]:
        return {d for d in self._load() if start <= d <= end}

    def as_set(self) -> Set[date]:
        return self._load()


@dataclass(frozen=True)
class NetherlandsHolidayCalendar:
    """Minimal built-in Dutch public-holiday calendar (documented approximation).

    Days: New Year; Easter Sunday/Monday; King's Day (27 Apr, moved to 26 Apr
    when the 27th is a Sunday); Liberation Day (5 May); Ascension;
    Whit Sunday/Monday; Christmas; Boxing Day.
    """

    start_year: int = 2000
    end_year: int = 2050
    _cached_years: Set[int] = field(default_factory=set, init=False, repr=False)
    _by_year: dict[int, Set[date]] = field(default_factory=dict, init=False, repr=False)

    def _holidays_for(self, year: int) -> Set[date]:
        """Compute the public holidays of one year (cached per year)."""
        if year < self.start_year or year > self.end_year:
            return set()
        if year in self._cached_years:
            return self._by_year[year]
        easter = easter_sunday(year)
        kings = date(year, 4, 27)
        if kings.weekday() == 6:  # King's Day on a Sunday -> observed 26 Apr
            kings = date(year, 4, 26)
        days = {
            date(year, 1, 1),  # New Year
            easter,  # Easter Sunday
            easter + timedelta(days=1),  # Easter Monday
            kings,  # King's Day
            date(year, 5, 5),  # Liberation Day (see docstring for the every-5-years caveat)
            easter + timedelta(days=39),  # Ascension
            easter + timedelta(days=49),  # Whit Sunday
            easter + timedelta(days=50),  # Whit Monday
            date(year, 12, 25),  # Christmas
            date(year, 12, 26),  # Boxing Day
        }
        merged = dict(self._by_year)
        merged[year] = days
        object.__setattr__(self, "_by_year", merged)
        object.__setattr__(self, "_cached_years", self._cached_years | {year})
        return days

    def as_set(self) -> Set[date]:
        return {d for y in range(self.start_year, self.end_year + 1) for d in self._holidays_for(y)}

    def is_holiday(self, day: date) -> bool:
        return day in self._holidays_for(day.year)

    def holidays_between(self, start: date, end: date) -> Set[date]:
        return {d for y in range(start.year, end.year + 1) for d in self._holidays_for(y) if start <= d <= end}


__all__ = [
    "HolidayCalendar",
    "easter_sunday",
    "CsvHolidayCalendar",
    "NetherlandsHolidayCalendar",
]