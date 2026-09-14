"""ENTSO-E Transparency Platform ingestion.

Fetches raw market data (load, generation by production type, day-ahead
prices, cross-border flows, imbalance prices) as XML/ZIP payloads and stores
them faithfully in the Bronze tier.

Endpoint identifiers and document types were verified against the public
ENTSO-E REST API and the widely used `entsoe-py` client (September 2026); see
``docs/data-sources.md`` for the verified-vs-uncertain register.
"""

from .domains import Area, NL, DE_LU, BE
from .client import EntsoeClient, ENTSOE_API_URL
from .imbalance import EntsoeImbalancePrices, ImbalancePricesSource

__all__ = ["Area", "NL", "DE_LU", "BE", "EntsoeClient", "ENTSOE_API_URL", "EntsoeImbalancePrices", "ImbalancePricesSource"]