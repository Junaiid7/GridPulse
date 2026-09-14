"""Unit tests for the explicit Silver/Gold schemas in transform-model.md."""

from __future__ import annotations

from gridpulse.transformation.schema import (
    GOLD_COLUMNS,
    GOLD_SCHEMA,
    SILVER_COLUMNS,
    SILVER_SCHEMA,
    SCHEMA_VERSION,
    Column,
    column_names,
)


def test_schema_version() -> None:
    assert SCHEMA_VERSION == "1.0"


def test_silver_column_names_and_count() -> None:
    assert column_names(SILVER_COLUMNS) == [
        "timestamp_utc",
        "timestamp_local",
        "value",
        "unit",
        "source",
        "entity",
        "flags",
    ]


def test_gold_column_names() -> None:
    assert column_names(GOLD_COLUMNS) == [
        "timestamp_utc",
        "timestamp_local",
        "load_mw",
        "wind_mw",
        "solar_mw",
        "residual_load_mw",
        "day_ahead_price_eur_mwh",
        "hour",
        "day_of_week",
        "day_of_year",
        "month",
        "is_weekend",
        "is_holiday",
        "flags",
        "sources",
    ]


def test_every_column_is_documented() -> None:
    for col in (*SILVER_COLUMNS, *GOLD_COLUMNS):
        assert col.meaning.strip(), f"{col.name} missing meaning"
        assert col.dtype in {"float", "int", "bool", "str"}, f"{col.name} bad dtype"
        assert col.source.strip(), f"{col.name} missing source"


def test_gold_nullability_is_explicit() -> None:
    # Currency/market columns may be absent; derived calendar fields never are.
    assert GOLD_SCHEMA["load_mw"].nullable is True
    assert GOLD_SCHEMA["wind_mw"].nullable is True
    assert GOLD_SCHEMA["solar_mw"].nullable is True
    assert GOLD_SCHEMA["residual_load_mw"].nullable is True
    assert GOLD_SCHEMA["day_ahead_price_eur_mwh"].nullable is True
    assert GOLD_SCHEMA["hour"].nullable is False
    assert GOLD_SCHEMA["is_weekend"].nullable is False
    assert GOLD_SCHEMA["is_holiday"].nullable is False
    for name in ("flags", "sources"):
        assert GOLD_SCHEMA[name].nullable is True


def test_gold_units_annotated() -> None:
    assert GOLD_SCHEMA["load_mw"].unit == "MW"
    assert GOLD_SCHEMA["wind_mw"].unit == "MW"
    assert GOLD_SCHEMA["solar_mw"].unit == "MW"
    assert GOLD_SCHEMA["residual_load_mw"].unit == "MW"
    assert GOLD_SCHEMA["day_ahead_price_eur_mwh"].unit == "EUR/MWh"


def test_timezone_columns_are_annotated() -> None:
    assert GOLD_SCHEMA["timestamp_utc"].timezone == "UTC"
    assert GOLD_SCHEMA["timestamp_local"].timezone == "Europe/Amsterdam"
    assert GOLD_SCHEMA["hour"].timezone == "Europe/Amsterdam"
    assert GOLD_SCHEMA["day_of_week"].timezone == "Europe/Amsterdam"


def test_schema_and_lookup_dont_drift() -> None:
    assert set(GOLD_SCHEMA) == {c.name for c in GOLD_COLUMNS}
    assert set(SILVER_SCHEMA) == {c.name for c in SILVER_COLUMNS}
    for col in GOLD_COLUMNS:
        assert GOLD_SCHEMA[col.name] == col


def test_no_duplicate_column_names() -> None:
    assert len({c.name for c in GOLD_COLUMNS}) == len(GOLD_COLUMNS)
    assert len({c.name for c in SILVER_COLUMNS}) == len(SILVER_COLUMNS)


def test_column_is_frozen_dataclass() -> None:
    col = GOLD_COLUMNS[0]
    assert isinstance(col, Column)
    assert col.name == "timestamp_utc"