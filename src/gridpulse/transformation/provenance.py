"""Provenance records attached to Silver/Gold outputs.

Provenance is written into the sidecar ``.meta.json`` so any downstream table
records: its tier, source entity, unit, the transformation policy in force
(timezone/DST convention), its version, and when it was generated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class Provenance:
    tier: str  # "silver" | "gold"
    source: str
    entity: str
    unit: str | None = None
    #: Central timezone/DST policy applied throughout the transformation layer.
    timezone_policy: str = "canonical_utc; local derived as Europe/Amsterdam"
    #: Retained as (…) when a source table was built from an older schema.
    version: str = "1.0"
    generated_at: datetime | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.generated_at is None:
            object.__setattr__(self, "generated_at", datetime.now(UTC))

    def to_dict(self) -> Mapping[str, object]:
        return {
            "tier": self.tier,
            "source": self.source,
            "entity": self.entity,
            "unit": self.unit,
            "timezone_policy": self.timezone_policy,
            "version": self.version,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "notes": list(self.notes),
        }


__all__ = ["Provenance"]