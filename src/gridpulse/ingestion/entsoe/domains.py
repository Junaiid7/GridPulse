"""Bidding-zone / control-area domains (EIC codes) used by the ENTSO-E API.

Codes verified from the ENTSO-E domain reference (as mirrored by entsoe-py's
``DOMAIN_MAPPINGS``, September 2026). The timezone column is informational:
ENTSO-E payloads are UTC; the timezone is used only to describe local market
conventions and future display.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Area:
    """A bidding zone or control area as identified by its EIC code."""

    name: str
    code: str
    tz: str  # IANA timezone of the market area (informational)

    def __str__(self) -> str:
        return f"{self.name} ({self.code})"


#: Netherlands — TenneT TSO, single bidding zone.
NL = Area(name="NL", code="10YNL----------L", tz="Europe/Amsterdam")

#: Germany/Luxembourg combined bidding zone (valid since 2018).
DE_LU = Area(name="DE_LU", code="10Y1001A1001A82H", tz="Europe/Berlin")

#: Belgium — Elia TSO, single bidding zone.
BE = Area(name="BE", code="10YBE----------2", tz="Europe/Brussels")


__all__ = ["Area", "NL", "DE_LU", "BE"]