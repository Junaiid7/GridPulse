"""Deterministic Bronze-tier storage.

Writes each raw fetch as an immutable payload file plus a sidecar JSON
manifest carrying all provenance metadata (source, entity, requested window,
retrieval time, timezone, units, identifiers, checksum, validation summary).
Paths are deterministic: ``data/bronze/<source>/<entity>/<start>__<end>/``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .models import FetchResult
from .validation import ValidationReport


def _entity_slug(entity: str) -> str:
    """Stable filesystem-safe entity name (alphanumeric segments separated by dashes)."""
    slug = re.sub(r"[^a-z0-9]+", "-", entity.lower()).strip("-")
    if not slug:
        raise ValueError(f"entity '{entity}' produces an empty filesystem slug")
    return slug


def _extension(content_type: str | None) -> str:
    ct = (content_type or "").lower()
    if "xml" in ct:
        return "xml"
    if "json" in ct:
        return "json"
    if "zip" in ct or "gzip" in ct:
        return "zip"
    if "csv" in ct:
        return "csv"
    return "bin"


@dataclass(frozen=True)
class BronzeWrite:
    """Result of a Bronze write: the created files and their directory."""

    base_dir: Path
    payload_path: Path
    manifest_path: Path


def write_bronze(
    bronze_root: Path,
    result: FetchResult,
    *,
    validation: ValidationReport | None = None,
) -> BronzeWrite:
    """Persist ``result`` under ``bronze_root`` and return created paths."""
    base_dir = (
        bronze_root
        / result.source
        / _entity_slug(result.entity)
        / f"{result.start:%Y%m%d}__{result.end:%Y%m%d}"
    )
    stamp = f"{result.retrieved_at:%Y%m%dT%H%M%S}Z"
    payload_path = base_dir / f"payload_{stamp}.{_extension(result.content_type)}"
    manifest_path = base_dir / f"manifest_{stamp}.json"

    base_dir.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(result.payload)

    manifest = {
        "schema_version": "1.0",
        "source": result.source,
        "entity": result.entity,
        "requested_window": {
            "start_utc": result.start.isoformat(),
            "end_utc": result.end.isoformat(),
        },
        "retrieved_at_utc": result.retrieved_at.isoformat(),
        "timezone": result.timezone,
        "units": result.units,
        "content_type": result.content_type,
        "encoding": result.encoding,
        "source_url": result.url,
        "identifiers": sorted(result.identifiers),
        "payload": {
            "file": payload_path.name,
            "size_bytes": len(result.payload),
            "sha256": result.sha256,
        },
        "metadata": dict(result.metadata),
        "validation": validation.summary() if validation is not None else None,
    }
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True, default=str)
    manifest_path.write_text(manifest_text, encoding="utf-8")

    return BronzeWrite(base_dir=base_dir, payload_path=payload_path, manifest_path=manifest_path)