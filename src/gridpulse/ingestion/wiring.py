"""Construction of ingestion clients from the centralised configuration.

Keeps secrets out of code paths: the ENTSO-E key is read from the environment
(via :func:`gridpulse.config.get_settings`) and never hardcoded or logged.
"""

from __future__ import annotations

import logging

from ..config import get_settings, require_entsoe_api_key
from .common.http import HttpClient
from .entsoe.client import EntsoeClient
from .weather.open_meteo import OpenMeteoClient

LOGGER = logging.getLogger("gridpulse.ingestion.wiring")


def get_http_client(settings=None) -> HttpClient:
    settings = settings or get_settings()
    return HttpClient(
        timeout=settings.http_timeout_s,
        retries=settings.http_retries,
        logger=logging.getLogger("gridpulse.ingestion.http"),
    )


def get_entsoe_client(settings=None) -> EntsoeClient:
    """Build the ENTSO-E client, requiring an API key.

    Raises ``ConfigurationError`` when ``ENTSOE_API_KEY`` is not configured.
    """
    settings = settings or get_settings()
    api_key = require_entsoe_api_key(settings)
    return EntsoeClient(api_key=api_key, http=get_http_client(settings))


def get_open_meteo_client(settings=None) -> OpenMeteoClient:
    settings = settings or get_settings()
    return OpenMeteoClient(http=get_http_client(settings))


__all__ = ["get_http_client", "get_entsoe_client", "get_open_meteo_client"]