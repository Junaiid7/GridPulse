"""ENTSO-E REST API parameter definitions.

Document types, process/business types, resolution strings, production type
(PSR) codes and default chunk windows. Values follow the ENTSO-E web API and
the widely used entsoe-py client; see ``docs/data-sources.md``.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from .domains import Area

#: REST base URL (verified live: returns HTTP 401 with an XML acknowledgement
#: when the security token is missing or invalid).
ENTSOE_API_URL = "https://web-api.tp.entsoe.eu/api"

# Document types (``documentType``).
DOC_ACTUAL_LOAD = "A65"  # processType A16 = "Realised"
DOC_LOAD_FORECAST = "A65"  # processType A01 = "Day ahead"
DOC_ACTUAL_GENERATION = "A75"
DOC_DAYAHEAD_PRICES = "A44"
DOC_CROSSBORDER_PHYSICAL_FLOWS = "A11"
DOC_IMBALANCE_PRICES = "A85"
DOC_IMBALANCE_VOLUMES = "A86"

# Process types.
PROCESS_REALISED = "A16"
PROCESS_DAY_AHEAD = "A01"

#: Day-ahead market agreement for SDAC prices.
CONTRACT_DAY_AHEAD_AUCTION = "A01"

#: PSR (production type) codes — the subset relevant to NL residual load.
#: Full list (B01–B25) is documented on the ENTSO-E guide; we keep the symbols
#: we actually query so no constant drifts out of sync.
PRODUCTION_TYPES: dict[str, str] = {
    "B01": "biomass",
    "B04": "fossil-gas",
    "B10": "hydro-pumped-storage",
    "B11": "hydro-run-of-river",
    "B12": "hydro-water-reservoir",
    "B14": "nuclear",
    "B16": "solar",
    "B17": "waste",
    "B18": "wind-offshore",
    "B19": "wind-onshore",
    "B20": "other",
}

#: Production types that drive residual load volatility in NL (solar + wind).
RESIDUAL_DRIVEN_PSR = ("B16", "B18", "B19")

#: ENTSO-E resolution strings → minutes.
RESOLUTION_MINUTES: dict[str, int] = {
    "PT15M": 15,
    "PT30M": 30,
    "PT60M": 60,
    "PT1M": 1,
    "P1D": 1440,
}

#: Quantity unit name used in the XML for average MW values.
UNIT_QUANTITY_RAW_MAW = "MAW"
UNIT_PRICE_RAW_MWH = "MWH"

# Query builders -----------------------------------------------------------
# Each builder returns the API parameters for one document type. The client
# adds ``securityToken``, ``periodStart`` and ``periodEnd``.


def actual_load_params(area: Area) -> dict[str, str]:
    return {
        "documentType": DOC_ACTUAL_LOAD,
        "processType": PROCESS_REALISED,
        "outBiddingZone_Domain": area.code,
        "out_Domain": area.code,
    }


def day_ahead_load_forecast_params(area: Area) -> dict[str, str]:
    return {
        "documentType": DOC_LOAD_FORECAST,
        "processType": PROCESS_DAY_AHEAD,
        "outBiddingZone_Domain": area.code,
    }


def actual_generation_params(area: Area, psr_type: Optional[str] = None) -> dict[str, str]:
    params = {
        "documentType": DOC_ACTUAL_GENERATION,
        "processType": PROCESS_REALISED,
        "in_Domain": area.code,
    }
    if psr_type:
        if psr_type not in PRODUCTION_TYPES:
            raise ValueError(f"unknown psrType {psr_type!r}; expected one of {sorted(PRODUCTION_TYPES)}")
        params["psrType"] = psr_type
    return params


def day_ahead_prices_params(area: Area, *, classification_sequence: Optional[int] = None) -> dict[str, str | int]:
    params: dict[str, str | int] = {
        "documentType": DOC_DAYAHEAD_PRICES,
        "in_Domain": area.code,
        "out_Domain": area.code,
        "contract_MarketAgreement.type": CONTRACT_DAY_AHEAD_AUCTION,
    }
    if classification_sequence is not None:
        # Some zones publish multiple sequences; DE_LU and AT need sequence 1.
        params["classificationSequence_AttributeInstanceComponent.position"] = classification_sequence
    return params


def crossborder_physical_flows_params(area_from: Area, area_to: Area) -> dict[str, str]:
    """Physical flows from ``area_from`` (out) into ``area_to`` (in).

    Positive quantities flow from ``area_from`` to ``area_to``.
    """
    return {
        "documentType": DOC_CROSSBORDER_PHYSICAL_FLOWS,
        "in_Domain": area_to.code,
        "out_Domain": area_from.code,
    }


def imbalance_prices_params(area: Area) -> dict[str, str]:
    """Imbalance prices (documentType A85) — availability is source-dependent.

    NL coverage is UNVERIFIED (needs a live API key); see ``imbalance.py`` and
    ``docs/data-sources.md``. Responses may be a ZIP of XML files.
    """
    return {
        "documentType": DOC_IMBALANCE_PRICES,
        "controlArea_Domain": area.code,
    }


#: Conservative default chunk windows per dataset (ISO durations). The API
#: enforces element-based limits; the client halves automatically if the
#: ``exceeds allowed limit`` error is returned.
CHUNK_SPANS: dict[str, timedelta] = {
    "actual-total-load": timedelta(days=31),
    "dayahead-total-load-forecast": timedelta(days=31),
    "actual-generation-by-type": timedelta(days=92),
    "dayahead-prices": timedelta(days=366),
    "cross-border-physical-flows": timedelta(days=92),
    "imbalance-prices": timedelta(days=366),
}


__all__ = [
    "ENTSOE_API_URL",
    "PRODUCTION_TYPES",
    "RESIDUAL_DRIVEN_PSR",
    "RESOLUTION_MINUTES",
    "actual_load_params",
    "day_ahead_load_forecast_params",
    "actual_generation_params",
    "day_ahead_prices_params",
    "crossborder_physical_flows_params",
    "imbalance_prices_params",
    "CHUNK_SPANS",
]