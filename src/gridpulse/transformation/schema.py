"""Explicit, typed schemas for the Silver and Gold layers.

Every column is documented (name, meaning, unit, timezone, nullability,
source). The lists live in one place so the CSV writers, the sidecar metadata
and ``docs/transform-model.md`` cannot drift apart.

Timezone convention: ``timestamp_utc`` is the canonical clock; local columns
use Europe/Amsterdam. ``unit`` is the SI/market unit as annotated at ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    """One column of a Silver or Gold table."""

    name: str
    meaning: str
    unit: str | None
    timezone: str | None
    nullable: bool
    source: str
    #: Python-ish storage type hint for documentation ("float", "int", "bool", "str").
    dtype: str


SCHEMA_VERSION = "1.0"


# Silver: one row per cleaned minute/hour observation, one table per entity.
SILVER_COLUMNS: tuple[Column, ...] = (
    Column("timestamp_utc", "Canonical observation time (UTC)", None, "UTC", False, "all", "str"),
    Column("timestamp_local", "Europe/Amsterdam wall-clock time (with UTC offset)", None, "Europe/Amsterdam", False, "all", "str"),
    Column("value", "Cleaned numeric value in the entity's unit", None, None, False, "all", "float"),
    Column("unit", "Unit of ``value`` as annotated by the source", None, None, False, "all", "str"),
    Column("source", "Source provider (``entsoe`` / ``open-meteo``)", None, None, False, "all", "str"),
    Column("entity", "Source entity key (e.g. ``actual-total-load``)", None, None, False, "all", "str"),
    Column("flags", "''-joined data-quality flags (e.g. ``duplicate_dropped``)", None, None, True, "all", "str"),
)

# Gold: the forecast-ready hourly Dutch dataset (schema_version 1.0).
GOLD_COLUMNS: tuple[Column, ...] = (
    Column("timestamp_utc", "Canonical start of the UTC hour", None, "UTC", False, "derived", "str"),
    Column("timestamp_local", "Europe/Amsterdam start of the local hour (with offset)", None, "Europe/Amsterdam", False, "derived", "str"),
    Column("load_mw", "Actual total load, hour mean", "MW", "UTC", True, "entsoe actual-total-load (A65/A16)", "float"),
    Column("wind_mw", "Wind generation = wind-offshore + wind-onshore (B18+B19), hour mean", "MW", "UTC", True, "entsoe actual-generation-by-type (A75)", "float"),
    Column("solar_mw", "Solar generation (B16), hour mean", "MW", "UTC", True, "entsoe actual-generation-by-type (A75)", "float"),
    Column("residual_load_mw", "load_mw − wind_mw − solar_mw (negative kept; NaN when load missing)", "MW", "UTC", True, "derived", "float"),
    Column("day_ahead_price_eur_mwh", "Day-ahead SDAC price, hour value", "EUR/MWh", "UTC", True, "entsoe dayahead-prices (A44)", "float"),
    Column("hour", "Europe/Amsterdam local hour (0–23)", None, "Europe/Amsterdam", False, "derived", "int"),
    Column("day_of_week", "Europe/Amsterdam weekday, Monday=0 … Sunday=6", None, "Europe/Amsterdam", False, "derived", "int"),
    Column("day_of_year", "Europe/Amsterdam day of year (1–366)", None, "Europe/Amsterdam", False, "derived", "int"),
    Column("month", "Europe/Amsterdam month (1–12)", None, "Europe/Amsterdam", False, "derived", "int"),
    Column("is_weekend", "True on local Saturday/Sunday", None, "Europe/Amsterdam", False, "derived", "bool"),
    Column("is_holiday", "True on a configured NL public holiday", None, "Europe/Amsterdam", False, "holiday calendar", "bool"),
    Column("flags", "Comma-joined presence flags (e.g. ``wind_missing``)", None, None, True, "derived", "str"),
    Column("sources", "Comma-joined sources that contributed values to this row", None, None, True, "derived", "str"),
)

#: Schema mapping name → :class:`Column` for programmatic checks.
GOLD_SCHEMA: dict[str, Column] = {c.name: c for c in GOLD_COLUMNS}
SILVER_SCHEMA: dict[str, Column] = {c.name: c for c in SILVER_COLUMNS}


def column_names(columns: tuple[Column, ...]) -> list[str]:
    return [c.name for c in columns]


__all__ = [
    "Column",
    "SCHEMA_VERSION",
    "SILVER_COLUMNS",
    "GOLD_COLUMNS",
    "SILVER_SCHEMA",
    "GOLD_SCHEMA",
    "column_names",
]