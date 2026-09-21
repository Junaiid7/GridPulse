"""Open-Meteo Historical Archive API client.

Endpoint ``https://archive-api.open-meteo.com/v1/archive`` (verified live,
no API key required for non-commercial use). The archive serves long ranges
in a single call (verified: a 3-year hourly request returns ~1 MB), so no
chunking is applied by default; ``chunk_days`` is available for callers who
want bounded memory or polite request sizes.

Requested with ``timezone=UTC`` and ``wind_speed_unit=ms`` so payload
timestamps are UTC and wind speeds are SI.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta

from ..common.errors import EmptyResponseError
from ..common.http import HttpClient
from ..common.models import FetchResult, ensure_utc
from .variables import HOURLY_VARIABLES, WIND_SPEED_UNIT, Location

LOGGER = logging.getLogger("gridpulse.ingestion.weather")

#: Historical archive endpoint (verified live, 2026-09).
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


class OpenMeteoClient:
    """Client for the Open-Meteo historical archive."""

    def __init__(self, *, http: HttpClient | None = None, logger: logging.Logger | None = None) -> None:
        self._http = http or HttpClient()
        self._log = logger or LOGGER

    def fetch_historical(
        self,
        location: Location,
        start_date: date,
        end_date: date,
        *,
        variables: Iterable[str] = HOURLY_VARIABLES,
        tz: str = "UTC",
        chunk_days: int | None = None,
        model: str | None = None,
    ) -> list[FetchResult]:
        """Fetch a historical hourly block for one location.

        ``end_date`` is inclusive on Open-Meteo's side; the returned
        ``FetchResult`` records the requested window as the half-open UTC
        range ``[start_date 00:00, end_date+1d 00:00)``.
        """
        if start_date > end_date:
            raise ValueError("start_date must not be after end_date")

        variable_csv = ",".join(variables)
        windows = self._windows(start_date, end_date, chunk_days)
        results: list[FetchResult] = []
        for w_start, w_end in windows:
            params = {
                "latitude": location.lat,
                "longitude": location.lon,
                "start_date": w_start.isoformat(),
                "end_date": w_end.isoformat(),
                "hourly": variable_csv,
                "timezone": tz,
                "wind_speed_unit": WIND_SPEED_UNIT,
            }
            if model:
                params["model"] = model

            response = self._http.get(ARCHIVE_URL, params=params)
            if not response.body:
                raise EmptyResponseError(f"open-meteo returned empty body for {location.name}")

            window_start = ensure_utc(datetime.combine(w_start, time.min))
            window_end = ensure_utc(datetime.combine(w_end, time.min)) + timedelta(days=1)
            results.append(
                FetchResult(
                    source="open-meteo",
                    entity="historical-weather",
                    start=window_start,
                    end=window_end,
                    retrieved_at=datetime.now(UTC),
                    payload=response.body,
                    content_type=response.headers.get("content-type") or "application/json",
                    encoding="utf-8",
                    url=response.url,
                    identifiers=(f"location:{location.name}", f"lat:{location.lat:.6f}", f"lon:{location.lon:.6f}"),
                    timezone=tz,
                    metadata={
                        "location": location.name,
                        "latitude": location.lat,
                        "longitude": location.lon,
                        "variables": variable_csv,
                        "model": model,
                        "start_date": w_start.isoformat(),
                        "end_date": w_end.isoformat(),
                    },
                )
            )
            self._log.info("Fetched open-meteo %s %s..%s (%d bytes)", location.name, w_start, w_end, len(response.body))
        return results

    @staticmethod
    def _windows(start_date: date, end_date: date, chunk_days: int | None) -> list[tuple[date, date]]:
        if chunk_days is None or chunk_days <= 0:
            return [(start_date, end_date)]
        out: list[tuple[date, date]] = []
        cursor = start_date
        while cursor <= end_date:
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), end_date)
            out.append((cursor, chunk_end))
            cursor = chunk_end + timedelta(days=1)
        return out


__all__ = ["OpenMeteoClient", "ARCHIVE_URL"]