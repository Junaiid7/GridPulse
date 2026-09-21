"""Deterministic storage for Silver and Gold tables.

Standard-library CSV was chosen over pandas/pyarrow/Parquet deliberately:

- the hourly NL dataset is small (~9k rows/year per entity), so CSV is trivial
  to read, diff, and debug;
- the project dependency policy (Phase 3) forbids heavy tabular dependencies
  "merely because they are common";
- Parquet/pyarrow can replace this writer in the future warehouse phase
  without changing the public API.

Paths are deterministic (fixed basenames — no run-timestamp in filenames), and
a sidecar ``.meta.json`` records provenance, schema version and row counts.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .provenance import Provenance
from .schema import SCHEMA_VERSION


def write_table(
    csv_path: Path,
    *,
    header: Sequence[str],
    rows: Iterable[Sequence[Any]],
    provenance: Provenance,
    meta_path: Path | None = None,
    meta_extras: Mapping[str, Any] | None = None,
) -> None:
    """Write rows to ``csv_path`` (LF endings, CSV-quoted) plus optional sidecar."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([_serialize(cell) for cell in row])

    if meta_path is not None:
        meta = {
            "schema_version": SCHEMA_VERSION,
            "columns": list(header),
            "row_count": _count_rows(csv_path) - 1,
            "provenance": provenance.to_dict(),
        }
        if meta_extras:
            meta.update(dict(meta_extras))
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True, default=str, ensure_ascii=False),
            encoding="utf-8",
        )


def read_table(csv_path: Path) -> list[dict[str, Any]]:
    """Read a table written by :func:`write_table` back into dicts."""
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _count_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in fh)


def _serialize(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, datetime):
        # Aware ISO with explicit offset — never ambiguous, always parseable.
        if cell.tzinfo is None:
            cell = cell.replace(tzinfo=UTC)
        return cell.isoformat()
    if isinstance(cell, bool):
        return "true" if cell else "false"
    if isinstance(cell, float):
        return repr(cell)
    if isinstance(cell, (list, tuple)):  # flags / sources as comma-joined
        return ",".join(str(c) for c in cell)
    return str(cell)


__all__ = ["write_table", "read_table"]