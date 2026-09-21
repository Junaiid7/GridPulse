"""Calendar features (computed from the calendar, never from data).

For a forecast of hour ``target_utc``, these features are fully known at issue
time regardless of data availability: hour / day-of-week / day-of-year / month
are Europe/Amsterdam local; ``is_weekend`` and ``is_holiday`` follow.
"""

from __future__ import annotations

from datetime import datetime

from ..transformation.times import AMSTERDAM_TZ, local_time
from .holiday import HolidayCalendar


def calendar_features(
    target_utc: datetime,
    *,
    holiday_calendar: HolidayCalendar | None = None,
    tz=AMSTERDAM_TZ,
) -> dict[str, object]:
    """Return the feature dict for one target hour (local Amsterdam time)."""
    local = local_time(target_utc)
    local_d = local.date()

    features: dict[str, object] = {
        "hour": local.hour,
        "day_of_week": local_d.weekday(),  # Monday=0 … Sunday=6
        "day_of_year": local_d.timetuple().tm_yday,
        "month": local_d.month,
        "is_weekend": local_d.weekday() >= 5,
    }
    if holiday_calendar is not None:
        features["is_holiday"] = bool(holiday_calendar.is_holiday(local_d))
    return features


__all__ = ["calendar_features"]