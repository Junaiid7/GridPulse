"""Fallback interface for a future synthetic imbalance penalty.

Context: NL imbalance-price availability via ENTSO-E could not be verified
(Phase 2 research; see ``docs/data-sources.md``). Rather than invent a real
source, GridPulse routes the (hypothetical) absence of NL imbalance prices
through this interface. The synthetic penalty itself is intentionally NOT
implemented yet — creating one is a modelled, reviewed decision for a later
phase, not an ingestion shortcut.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from ..common.models import FetchResult
from ..entsoe.domains import Area


@runtime_checkable
class ImbalancePenaltyProvider(Protocol):
    """Contract for any future provider of NL imbalance penalty signals."""

    available: bool

    def fetch(self, area: Area, start: datetime, end: datetime) -> list[FetchResult]: ...


class SyntheticImbalancePenalty:
    """Placeholder for a future synthetic NL imbalance penalty.

    Calling ``fetch`` raises :class:`NotImplementedError` on purpose: pretend
    data is worse than documented absence. Models that need an imbalance
    penalty must check :attr:`available` and handle ``False`` explicitly.
    """

    available: bool = False
    reason: str = (
        "NL imbalance price availability via ENTSO-E is unverified; "
        "a synthetic imbalance penalty is planned for a later phase and has "
        "not been implemented."
    )

    def fetch(self, area: Area, start: datetime, end: datetime) -> list[FetchResult]:
        raise NotImplementedError(
            f"Synthetic NL imbalance penalty is intentionally not implemented "
            f"({self.reason})"
        )


__all__ = ["ImbalancePenaltyProvider", "SyntheticImbalancePenalty"]