"""Data ingestion: pull raw market and weather data into the Bronze tier.

Phase 2 implements the ingestion foundation:

- ``common``  — shared HTTP client, models, time-range handling, validation and
  deterministic Bronze storage.
- ``entsoe``  — ENTSO-E Transparency Platform REST client, XML parsing, and a
  clearly separated fallback interface for the (unverified / future synthetic)
  NL imbalance penalty.
- ``weather`` — Open-Meteo historical archive client and JSON parsing; the
  ``forecast`` namespace is reserved and intentionally not implemented yet.

Raw responses are stored faithfully in ``data/bronze``; aggressive
transformation is deliberately out of scope at this tier.
"""

from .wiring import get_entsoe_client, get_open_meteo_client, get_settings

__all__ = ["get_entsoe_client", "get_open_meteo_client", "get_settings"]