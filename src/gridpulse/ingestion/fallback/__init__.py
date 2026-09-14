"""Clearly separated fallback providers for data that is not (yet) available.

Kept separate from the real-source packages so no fallback can be mistaken
for market data. Currently the only member is the future synthetic NL
imbalance penalty, which is deliberately NOT implemented in Phase 2.
"""

from .imbalance import ImbalancePenaltyProvider, SyntheticImbalancePenalty

__all__ = ["ImbalancePenaltyProvider", "SyntheticImbalancePenalty"]