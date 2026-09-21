"""Gold: assemble the hourly Dutch (NL) analytical dataset.

Inputs are hourly, UTC-anchored value maps (timestamp → value) produced by the
Silver layer for each quantity. The builder aligns them on the union of their
UTC hour timestamps, derives calendar fields in Europe/Amsterdam time, computes
residual load under the documented missing-data policy, and attaches per-row
provenance (``flags`` + ``sources``).

Columns are defined in :mod:`..schema` (``GOLD_SCHEMA``). Missing values are
represented as empty CSV cells and flagged — never fabricated. The builder does
not invent a grid: it reports every union hour present in any input.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from ...ingestion.common.models import ensure_utc
from ..csvio import read_table, write_table
from ..provenance import Provenance
from ..schema import GOLD_COLUMNS, column_names
from .residual_load import (
    PSR_LABELS,
    SOLAR_PSR,
    WIND_PSR,
    category_hourly,
    compute_residual,
)

_GOLD_HEADER = column_names(GOLD_COLUMNS)
_NL_TZ = ZoneInfo("Europe/Amsterdam")


class HolidayLike(Protocol):
    def is_holiday(self, d: object) -> bool: ...


@dataclass(frozen=True)
class GoldHourlyRow:
    """One hour of the Gold NL dataset (typed; CSV cells derive from this)."""

    timestamp_utc: datetime
    timestamp_local: datetime
    load_mw: float | None
    wind_mw: float | None
    solar_mw: float | None
    residual_load_mw: float | None
    day_ahead_price_eur_mwh: float | None
    hour: int
    day_of_week: int
    day_of_year: int
    month: int
    is_weekend: bool
    is_holiday: bool
    flags: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoldInputs:
    """Named inputs to :func:`build_gold_hourly`.

    ``generation_by_psr`` maps a PSR code (e.g. ``"B18"``) to its per-timestamp
    value map; ``wind``/``solar`` are derived from it via the documented PSR
    aggregation in :mod:`residual_load`. ``sources`` maps input keys to
    human-readable provenance labels recorded on each row.
    """

    load: Mapping[datetime, float] = field(default_factory=dict)
    generation_by_psr: Mapping[str, Mapping[datetime, float]] = field(default_factory=dict)
    day_ahead_price: Mapping[datetime, float] = field(default_factory=dict)
    holiday_calendar: HolidayLike | None = None
    #: ``key → label``; keys "load", "price" and "psr:<code>" are recognised.
    sources: Mapping[str, str] = field(default_factory=dict)


def build_gold_hourly(inputs: GoldInputs) -> list[GoldHourlyRow]:
    """Assemble the Gold hourly NL rows for the union of supplied timestamps."""
    wind, _ = category_hourly(dict(inputs.generation_by_psr), WIND_PSR)
    solar, _ = category_hourly(dict(inputs.generation_by_psr), SOLAR_PSR)

    grid: set[datetime] = set()
    grid.update(ensure_utc(ts) for ts in inputs.load)
    grid.update(ensure_utc(ts) for ts in wind)
    grid.update(ensure_utc(ts) for ts in solar)
    grid.update(ensure_utc(ts) for ts in inputs.day_ahead_price)
    if not grid:
        return []

    rows: list[GoldHourlyRow] = []
    for ts in sorted(grid):
        ts = ensure_utc(ts)
        local = ts.astimezone(_NL_TZ)

        load = inputs.load.get(ts)
        wind_at = wind.get(ts)
        solar_at = solar.get(ts)
        residual = compute_residual(load, wind_at, solar_at)

        d = local.date()
        sources = [
            label
            for key, label in inputs.sources.items()
            if (key == "load" and load is not None)
            or (key == "price" and ts in inputs.day_ahead_price)
            or (key.startswith("psr:") and ts in inputs.generation_by_psr.get(key[len("psr:"):], {}))
        ]

        rows.append(
            GoldHourlyRow(
                timestamp_utc=ts,
                timestamp_local=local,
                load_mw=load,
                wind_mw=wind_at,
                solar_mw=solar_at,
                residual_load_mw=residual.residual_mw,
                day_ahead_price_eur_mwh=inputs.day_ahead_price.get(ts),
                hour=local.hour,
                day_of_week=d.weekday(),
                day_of_year=d.timetuple().tm_yday,
                month=d.month,
                is_weekend=d.weekday() >= 5,
                is_holiday=bool(inputs.holiday_calendar and inputs.holiday_calendar.is_holiday(d)),
                flags=tuple(sorted(set(residual.flags))) if residual.flags else (),
                sources=tuple(sorted(set(sources))) if sources else (),
            )
        )
    return rows


def write_gold(
    dest_root: Path,
    rows: list[GoldHourlyRow],
    *,
    area: str = "nl",
    generated_at: datetime | None = None,
    residual_definition: str = "load − wind (B18+B19) − solar (B16)",
) -> tuple[Path, Path]:
    """Write Gold rows deterministically; returns ``(csv_path, meta_path)``."""
    base = dest_root / _slug(area)
    csv_path = base / "hourly.csv"
    meta_path = base / "hourly.meta.json"

    matrix: list[list[Any]] = [
        [
            r.timestamp_utc,
            r.timestamp_local,
            r.load_mw,
            r.wind_mw,
            r.solar_mw,
            r.residual_load_mw,
            r.day_ahead_price_eur_mwh,
            r.hour,
            r.day_of_week,
            r.day_of_year,
            r.month,
            r.is_weekend,
            r.is_holiday,
            ",".join(r.flags),
            ",".join(r.sources),
        ]
        for r in rows
    ]

    missing = {
        "load_mw": sum(1 for r in rows if r.load_mw is None),
        "wind_mw": sum(1 for r in rows if r.wind_mw is None),
        "solar_mw": sum(1 for r in rows if r.solar_mw is None),
        "residual_load_mw": sum(1 for r in rows if r.residual_load_mw is None),
        "day_ahead_price_eur_mwh": sum(1 for r in rows if r.day_ahead_price_eur_mwh is None),
    }

    write_table(
        csv_path,
        header=_GOLD_HEADER,
        rows=matrix,
        provenance=Provenance(
            tier="gold",
            source="entsoe+open-meteo",
            entity=f"hourly-{area}-dataset",
            unit="MW / EUR/MWh",
            generated_at=generated_at,
            notes=(residual_definition, f"wind_psr={','.join(WIND_PSR)}", f"solar_psr={','.join(SOLAR_PSR)}"),
        ),
        meta_path=meta_path,
        meta_extras={
            "residual_definition": residual_definition,
            "psr_mapping": dict(PSR_LABELS),
            "wind_psr": list(WIND_PSR),
            "solar_psr": list(SOLAR_PSR),
            "missing_counts": missing,
        },
    )
    return csv_path, meta_path


def read_gold(csv_path: Path) -> list[dict[str, Any]]:
    """Read a Gold CSV back into plain dicts (for tests / inspection)."""
    return read_table(csv_path)


def _slug(text: str) -> str:
    import re

    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "area"


__all__ = ["GoldHourlyRow", "GoldInputs", "build_gold_hourly", "write_gold", "read_gold"]