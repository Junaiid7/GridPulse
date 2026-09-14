"""Dutch imbalance-price availability and the interface for its sources.

Status (verified September 2026, no live ENTSO-E key available to us):

- The ENTSO-E REST API exposes ``documentType=A85`` (imbalance prices) and the
  endpoint reachability is verified, but **NL imbalance-price coverage is
  UNVERIFIED**. TSOs publish imbalance data voluntarily on the Transparency
  Platform; TenneT primarily publishes Dutch regulation/imbalance prices from
  its own channels. Since we cannot prove a real NL A85 series exists, we do
  NOT assume one and never fabricate a source.
- For the future synthetic imbalance penalty there is a clearly separated
  fallback interface in :mod:`gridpulse.ingestion.fallback` — intentionally
  not implemented yet.

Rule: no code outside these modules may silently treat ``None``/empty NL
imbalance data as a real signal.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ..common.models import FetchResult
from .client import EntsoeClient
from .domains import Area


@runtime_checkable
class ImbalancePricesSource(Protocol):
    """Contract for any provider of imbalance-price/penalty signals."""

    availability: dict[str, str]

    def fetch(self, area: Area, start: datetime, end: datetime) -> list[FetchResult]: ...


class EntsoeImbalancePrices:
    """ENTSO-E imbalance prices (A85) with an explicit availability caveat.

    Because NL availability could not be verified, callers are expected to
    handle ``NoDataError`` and fall back to the (future) synthetic penalty via
    :class:`gridpulse.ingestion.fallback.imbalance.SyntheticImbalancePenalty`.
    """

    availability: dict[str, str] = {
        "verified": "endpoint reachable",
        "nl_coverage": "UNVERIFIED — requires a live ENTSO-E API key to confirm",
    }

    def __init__(self, client: EntsoeClient) -> None:
        self._client = client

    def fetch(self, area: Area, start: datetime, end: datetime) -> list[FetchResult]:
        return self._client.fetch_imbalance_prices(area, start, end)


__all__ = ["ImbalancePricesSource", "EntsoeImbalancePrices"]