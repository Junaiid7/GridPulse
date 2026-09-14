"""Silver transformation: UTC-normalised, conformed CSV tables.

Input is the *parsed* UTC series from the ingestion layer
(:class:`~gridpulse.ingestion.common.models.TimeSeries`), already carrying
``source`` / ``entity`` / ``unit`` / per-point UTC timestamps.

Cleaning performed here (and only here):

- timestamps forced to aware UTC and sorted;
- duplicate timestamps resolved (keep first, rest flagged ``duplicate_dropped``);
- missing timestamps are NOT filled — they stay absent and are reported as a
  warning (``timestamp_gap``) with a count;
- unit, source and entity are carried through verbatim;
- each record gets its deterministic Europe/Amsterdam ``timestamp_local``.

Output: deterministic CSV ``data/silver/<source>/<entity>/<entity>.csv`` plus
a ``<entity>.meta.json`` sidecar carrying provenance, schema columns and a
cleaning summary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ...ingestion.common.models import TimeSeries, ensure_utc
from ...ingestion.common.validation import Issue, ValidationReport, validate_timestamps
from ..csvio import read_table, write_table
from ..provenance import Provenance
from ..schema import SILVER_COLUMNS, column_names
from ..times import local_time

_SILVER_HEADER = column_names(SILVER_COLUMNS)


@dataclass(frozen=True)
class SilverRecord:
    """One cleaned Silver row (already UTC-normalised)."""

    timestamp_utc: datetime
    timestamp_local: datetime
    value: float
    unit: str
    source: str
    entity: str
    flags: tuple[str, ...] = ()


def clean_timeseries(series: TimeSeries) -> tuple[list[SilverRecord], ValidationReport]:
    """Normalise and clean one parsed series into Silver records.

    Returns ``(records, report)``. The report reuses the ingestion validation
    machinery: duplicate timestamps become an error, gaps a warning. Rows are
    sorted by UTC.
    """
    if series.tz != "UTC" or any(p.timestamp.tzinfo is None for p in series.points):
        raise ValueError("Silver requires UTC-aware, pre-normalised input series")

    report = validate_timestamps(series)

    seen: set[datetime] = set()
    records: list[SilverRecord] = []
    for point in sorted(series.points, key=lambda p: p.timestamp):
        ts = ensure_utc(point.timestamp)
        if ts in seen:
            report = ValidationReport(
                report.issues
                + (Issue("warning", "duplicate_dropped", f"dropped duplicate {ts.isoformat()}"),)
            )
            continue
        seen.add(ts)
        records.append(
            SilverRecord(
                timestamp_utc=ts,
                timestamp_local=local_time(ts),
                value=float(point.value),
                unit=series.unit,
                source=series.source,
                entity=series.entity,
            )
        )
    return records, report


def write_silver(
    dest_root: Path,
    records: list[SilverRecord],
    source: str,
    entity: str,
    unit: str,
    *,
    report: ValidationReport | None = None,
    generated_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Write Silver records deterministically; returns ``(csv_path, meta_path)``."""
    base = dest_root / _slug(source) / _slug(entity)
    csv_path = base / f"{_slug(entity)}.csv"
    meta_path = base / f"{_slug(entity)}.meta.json"

    rows = [
        [
            r.timestamp_utc,
            r.timestamp_local,
            r.value,
            r.unit,
            r.source,
            r.entity,
            ",".join(r.flags),
        ]
        for r in records
    ]
    summary = report.summary() if report is not None else {"errors": 0, "warnings": 0}
    write_table(
        csv_path,
        header=_SILVER_HEADER,
        rows=rows,
        provenance=Provenance(
            tier="silver",
            source=source,
            entity=entity,
            unit=unit,
            generated_at=generated_at,
        ),
        meta_path=meta_path,
        meta_extras={"cleaning": summary},
    )
    return csv_path, meta_path


def read_silver(csv_path: Path) -> list[dict[str, Any]]:
    """Read back a Silver CSV into plain dicts (for tests / inspection)."""
    return read_table(csv_path)


def _slug(text: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "entity"


__all__ = ["SilverRecord", "clean_timeseries", "write_silver", "read_silver"]