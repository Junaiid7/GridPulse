"""Open-Meteo historical weather ingestion.

Requests the historical archive API (no API key for non-commercial use) and
stores the raw JSON faithfully in Bronze. Variables are chosen for Dutch
electricity demand and renewable generation (temperature, wind at hub height,
solar radiation); forecast ingestion is a reserved namespace and not
implemented yet.
"""

from .open_meteo import OpenMeteoClient
from .variables import HOURLY_VARIABLES, NL_POINTS, Location

__all__ = ["OpenMeteoClient", "HOURLY_VARIABLES", "Location", "NL_POINTS"]