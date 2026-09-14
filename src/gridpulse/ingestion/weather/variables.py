"""Weather variables and representative Dutch locations.

The coordinate list is a starting point for residual-load modelling; a denser
grid can replace it later without code changes (it is read per-fetch via
:class:`~gridpulse.ingestion.weather.open_meteo.OpenMeteoClient`). Units are
whatever Open-Meteo returns for the requested unit preferences; the response
carries ``hourly_units`` so Silver never has to guess.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Location:
    name: str
    lat: float
    lon: float


#: Representative NL points — central, coast, south, east and two offshore
#: (wind) locations. Keep it small and explicit for Phase 2.
NL_POINTS: tuple[Location, ...] = (
    Location("nl-central", 52.21, 5.29),
    Location("nl-coast-west", 52.00, 4.50),
    Location("nl-south", 51.50, 5.50),
    Location("nl-east", 52.30, 6.50),
    Location("nl-offshore-north", 54.00, 5.50),
    Location("nl-offshore-west", 52.50, 3.50),
)

#: Hourly variables relevant to Dutch demand and renewable generation.
HOURLY_VARIABLES: tuple[str, ...] = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
    "precipitation",
)

#: Request wind speeds in m/s (SI) rather than Open-Meteo's km/h default.
WIND_SPEED_UNIT = "ms"

#: Variables whose source physics forbid negative values (validation contract).
NON_NEGATIVE_CONTRACT = (
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
    "cloud_cover",
    "precipitation",
    "wind_speed_10m",
    "wind_speed_100m",
)

__all__ = ["Location", "NL_POINTS", "HOURLY_VARIABLES", "WIND_SPEED_UNIT", "NON_NEGATIVE_CONTRACT"]