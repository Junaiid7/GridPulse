"""ENTSO-E Transparency Platform ingestion.

Fetches raw market data (load, generation by production type, day-ahead
prices, cross-border flows, imbalance prices) as XML/ZIP payloads and stores
them faithfully in the Bronze tier.

Endpoint identifiers and document types were verified against the public
ENTSO-E REST API and the widely used `entsoe-py` client (September 2026); see
``docs/data-sources.md`` for the verified-vs-uncertain register.
"""

from .client import ENTSOE_API_URL, EntsoeClient
from .domains import BE, DE_LU, NL, Area
from .imbalance import EntsoeImbalancePrices, ImbalancePricesSource

__all__ = ["Area", "NL", "DE_LU", "BE", "EntsoeClient", "ENTSOE_API_URL", "EntsoeImbalancePrices", "ImbalancePricesSource"]