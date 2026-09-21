"""ENTSO-E Transparency Platform REST client.

Fetches raw XML/ZIP payloads for the configured datasets and returns
:class:`~gridpulse.ingestion.common.models.FetchResult` objects ready for the
Bronze tier. The client:

- injects the ``securityToken`` query parameter (from a key passed at
  construction — never hardcoded, never logged);
- chunks long ranges into conservative windows and automatically halves any
  request the API rejects for exceeding the element limit;
- treats a ``200`` body containing ``No matching data found`` as an empty
  result, not an error;
- stores a sanitised (token-free) URL in provenance metadata.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from ..common.errors import (
    ConfigurationError,
    HttpError,
    QueryTooLargeError,
    RateLimitError,
)
from ..common.http import HttpClient, mask_url
from ..common.models import FetchResult, TimeRange, chunk_range, ensure_utc
from .domains import Area
from .mappings import (
    CHUNK_SPANS,
    ENTSOE_API_URL,
    actual_generation_params,
    actual_load_params,
    crossborder_physical_flows_params,
    day_ahead_load_forecast_params,
    day_ahead_prices_params,
    imbalance_prices_params,
)

LOGGER = logging.getLogger("gridpulse.ingestion.entsoe")

#: Deepest recursion for the automatic halving fallback.
_MAX_HALVING_DEPTH = 10


def _format_period(dt: datetime) -> str:
    """ENTSO-E ``periodStart``/``periodEnd``: UTC, ``YYYYMMDDHH00``."""
    utc = ensure_utc(dt).replace(minute=0, second=0, microsecond=0)
    return utc.strftime("%Y%m%d%H00")


class EntsoeClient:
    """Low-level client for the ENTSO-E Transparency Platform REST API."""

    def __init__(
        self,
        api_key: str,
        *,
        http: HttpClient | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not api_key or not str(api_key).strip():
            raise ConfigurationError(
                "ENTSОЕ API key is required. Set ENTSOE_API_KEY (never hardcode it)."
            )
        self._key = str(api_key).strip()
        self._http = http or HttpClient()
        self._log = logger or LOGGER

    # -- public fetchers ---------------------------------------------------

    def fetch_load(self, area: Area, start: datetime, end: datetime) -> list[FetchResult]:
        """Realised total load (documentType A65 / processType A16), MW."""
        return self._fetch(
            params=actual_load_params(area),
            area=area,
            entity="actual-total-load",
            start=start,
            end=end,
            span=CHUNK_SPANS["actual-total-load"],
            units="MW",
        )

    def fetch_load_forecast(self, area: Area, start: datetime, end: datetime) -> list[FetchResult]:
        """Day-ahead total load forecast (documentType A65 / processType A01), MW."""
        return self._fetch(
            params=day_ahead_load_forecast_params(area),
            area=area,
            entity="dayahead-total-load-forecast",
            start=start,
            end=end,
            span=CHUNK_SPANS["dayahead-total-load-forecast"],
            units="MW",
        )

    def fetch_generation(
        self,
        area: Area,
        start: datetime,
        end: datetime,
        *,
        psr_type: str | None = None,
    ) -> list[FetchResult]:
        """Realised generation by production type (documentType A75), MW."""
        identifiers = (f"psr:{psr_type}",) if psr_type else ()
        return self._fetch(
            params=actual_generation_params(area, psr_type),
            area=area,
            entity="actual-generation-by-type",
            start=start,
            end=end,
            span=CHUNK_SPANS["actual-generation-by-type"],
            units="MW",
            identifiers=identifiers,
        )

    def fetch_day_ahead_prices(self, area: Area, start: datetime, end: datetime) -> list[FetchResult]:
        """Day-ahead (SDAC) prices (documentType A44), EUR/MWh."""
        # DE_LU/AT publish multiple sequences; sequence 1 is the single market
        # clearing price (mirrors entsoe-py behaviour).
        sequence = 1 if area.name in ("DE_LU", "AT") else None
        return self._fetch(
            params=day_ahead_prices_params(area, classification_sequence=sequence),
            area=area,
            entity="dayahead-prices",
            start=start,
            end=end,
            span=CHUNK_SPANS["dayahead-prices"],
            units="EUR/MWh",
        )

    def fetch_crossborder_flows(
        self,
        area_from: Area,
        area_to: Area,
        start: datetime,
        end: datetime,
    ) -> list[FetchResult]:
        """Cross-border physical flows (documentType A11), MW.

        Positive values flow from ``area_from`` into ``area_to``.
        """
        entity = f"cross-border-physical-flows:{area_from.name.lower()}-{area_to.name.lower()}"
        return self._fetch(
            params=crossborder_physical_flows_params(area_from, area_to),
            area=area_from,
            entity=entity,
            start=start,
            end=end,
            span=CHUNK_SPANS["cross-border-physical-flows"],
            units="MW",
            identifiers=(area_from.code, area_to.code),
        )

    def fetch_imbalance_prices(self, area: Area, start: datetime, end: datetime) -> list[FetchResult]:
        """Imbalance prices (documentType A85).

        NL availability is UNVERIFIED without a live key; the endpoint may
        return ``No matching data found``. See ``docs/data-sources.md`` and the
        separate fallback interface in :mod:`gridpulse.ingestion.fallback`.
        """
        return self._fetch(
            params=imbalance_prices_params(area),
            area=area,
            entity="imbalance-prices",
            start=start,
            end=end,
            span=CHUNK_SPANS["imbalance-prices"],
            units="EUR/MWh",
        )

    # -- internals ---------------------------------------------------------

    def _fetch(
        self,
        *,
        params: dict[str, str | int],
        area: Area,
        entity: str,
        start: datetime,
        end: datetime,
        span,
        units: str | None,
        identifiers: tuple[str, ...] = (),
    ) -> list[FetchResult]:
        rng = TimeRange(ensure_utc(start), ensure_utc(end))
        window = chunk_range(rng.start, rng.end, span)
        results: list[FetchResult] = []
        for piece in window:
            results.extend(
                self._fetch_window(params, area, entity, piece, units, identifiers, depth=0)
            )
        return results

    def _fetch_window(
        self,
        params,
        area: Area,
        entity: str,
        rng: TimeRange,
        units: str | None,
        identifiers: tuple[str, ...],
        depth: int,
    ) -> list[FetchResult]:
        try:
            return [self._single_request(params, area, entity, rng, units, identifiers)]
        except QueryTooLargeError:
            if depth >= _MAX_HALVING_DEPTH:
                raise
            left, right = rng.split()
            self._log.info("Chunk too large for %s, halving %s", entity, rng)
            return self._fetch_window(params, area, entity, left, units, identifiers, depth + 1) + self._fetch_window(
                params, area, entity, right, units, identifiers, depth + 1
            )

    def _single_request(
        self,
        params,
        area: Area,
        entity: str,
        rng: TimeRange,
        units: str | None,
        identifiers: tuple[str, ...],
    ) -> FetchResult:
        query = dict(params)
        query["securityToken"] = self._key
        query["periodStart"] = _format_period(rng.start)
        query["periodEnd"] = _format_period(rng.end)

        try:
            response = self._http.get(ENTSOE_API_URL, params=query)
        except HttpError as exc:
            if exc.body and b"exceeds allowed limit" in exc.body:
                raise QueryTooLargeError(
                    "request exceeds ENTSO-E element limit", status=exc.status, body=exc.body
                ) from exc
            if isinstance(exc, RateLimitError):
                raise
            raise

        text = response.text
        if "No matching data found" in text:
            from ..common.errors import NoDataError

            raise NoDataError(f"{entity}: no data for {rng}")
        if not response.body:
            from ..common.errors import EmptyResponseError

            raise EmptyResponseError(f"{entity}: empty response for {rng}")

        self._log.info("Fetched %s %s (%d bytes)", entity, rng, len(response.body))
        return FetchResult(
            source="entsoe",
            entity=entity,
            start=rng.start,
            end=rng.end,
            retrieved_at=datetime.now(UTC),
            payload=response.body,
            content_type=response.headers.get("content-type"),
            encoding="utf-8",
            url=mask_url(response.url),
            identifiers=identifiers,
            timezone="UTC",
            units=units,
            metadata={
                "area": area.code,
                "area_name": area.name,
                "area_tz": area.tz,
                "document_type": str(query.get("documentType")),
                "process_type": str(query.get("processType")) if query.get("processType") else None,
                "period_start": query["periodStart"],
                "period_end": query["periodEnd"],
            },
        )


__all__ = ["EntsoeClient", "ENTSOE_API_URL", "_format_period"]