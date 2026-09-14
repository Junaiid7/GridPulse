"""Unit tests for the Gold hourly NL dataset builder and writer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from gridpulse.features.holiday import NetherlandsHolidayCalendar
from gridpulse.transformation.gold import GoldInputs, build_gold_hourly, read_gold, write_gold

UTC = timezone.utc

# Christmas Day 2024 is a Wednesday in the local calendar.
H_12 = datetime(2024, 12, 25, 12, 0, tzinfo=UTC)
H_13 = datetime(2024, 12, 25, 13, 0, tzinfo=UTC)


def _inputs() -> GoldInputs:
    return GoldInputs(
        load={H_12: 100.0, H_13: 200.0},
        generation_by_psr={
            "B18": {H_12: 10.0, H_13: 20.0},
            "B19": {H_12: 5.0},
            "B16": {H_12: 2.0},
        },
        day_ahead_price={H_12: 50.0},
        holiday_calendar=NetherlandsHolidayCalendar(),
        sources={
            "load": "entsoe load",
            "price": "entsoe price",
            "psr:B18": "entsoe wind-offshore",
            "psr:B19": "entsoe wind-onshore",
            "psr:B16": "entsoe solar",
        },
    )


def test_build_gold_hourly_union_grid() -> None:
    rows = build_gold_hourly(_inputs())
    assert [r.timestamp_utc for r in rows] == [H_12, H_13]


def test_build_gold_hourly_values_and_holiday() -> None:
    rows = rows_by_ts = {r.timestamp_utc: r for r in build_gold_hourly(_inputs())}

    r12 = rows_by_ts[H_12]
    assert r12.load_mw == 100.0
    assert r12.wind_mw == 15.0  # B18 10 + B19 5
    assert r12.solar_mw == 2.0
    assert r12.residual_load_mw == 83.0
    assert r12.day_ahead_price_eur_mwh == 50.0
    assert r12.is_holiday is True  # Christmas Day
    assert r12.is_weekend is False  # Wednesday
    assert set(r12.sources) == {
        "entsoe load",
        "entsoe price",
        "entsoe wind-offshore",
        "entsoe wind-onshore",
        "entsoe solar",
    }
    assert r12.flags == ()


def test_build_gold_hourly_flags_missing_generation() -> None:
    r13 = {r.timestamp_utc: r for r in build_gold_hourly(_inputs())}[H_13]
    # B19 and B16 have no data at H_13 -> wind present (B18), solar missing.
    assert r13.load_mw == 200.0
    assert r13.wind_mw == 20.0
    assert r13.solar_mw is None
    assert r13.residual_load_mw == 180.0  # solar treated as zero but flagged
    assert r13.flags == ("solar_missing",)
    assert r13.day_ahead_price_eur_mwh is None
    assert set(r13.sources) == {"entsoe load", "entsoe wind-offshore"}


def test_build_gold_hourly_calendar_fields_local() -> None:
    r12 = {r.timestamp_utc: r for r in build_gold_hourly(_inputs())}[H_12]
    # 12:00 UTC on 2024-12-25 -> 13:00 local CET.
    assert r12.hour == 13
    assert r12.day_of_year == 360
    assert r12.month == 12


def test_build_gold_hourly_empty_input() -> None:
    assert build_gold_hourly(GoldInputs()) == []


def test_write_gold_and_read_roundtrip(tmp_path: Path) -> None:
    rows = build_gold_hourly(_inputs())
    csv_path, meta_path = write_gold(tmp_path, rows)
    assert csv_path.name == "hourly.csv"
    assert meta_path.name == "hourly.meta.json"
    assert csv_path.exists()
    assert meta_path.exists()

    data = read_gold(csv_path)
    assert len(data) == 2
    first = data[0]
    assert first["timestamp_utc"] == "2024-12-25T12:00:00+00:00"
    assert first["load_mw"] == "100.0"
    assert first["wind_mw"] == "15.0"
    assert first["residual_load_mw"] == "83.0"
    assert first["is_holiday"] == "true"
    # Missing values serialize to empty cells (H_13 has no price).
    assert data[0]["day_ahead_price_eur_mwh"] == "50.0"
    assert data[1]["day_ahead_price_eur_mwh"] == ""

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["row_count"] == 2
    assert meta["solar_psr"] == ["B16"]
    assert meta["wind_psr"] == ["B18", "B19"]


def test_write_gold_missing_counts(tmp_path: Path) -> None:
    rows = build_gold_hourly(_inputs())
    _, meta_path = write_gold(tmp_path, rows)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    missing = meta["missing_counts"]
    assert missing["load_mw"] == 0
    assert missing["wind_mw"] == 0
    assert missing["solar_mw"] == 1  # H_13 lacks B16/B19
    assert missing["residual_load_mw"] == 0
    assert missing["day_ahead_price_eur_mwh"] == 1  # H_13 lacks price