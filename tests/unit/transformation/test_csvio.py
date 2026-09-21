"""Unit tests for deterministic CSV storage and sidecar metadata."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from gridpulse.transformation.csvio import read_table, write_table
from gridpulse.transformation.provenance import Provenance

UTC = UTC


def test_write_table_creates_file_with_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "silver" / "src" / "ent" / "ent.csv"
    write_table(
        csv_path,
        header=["timestamp_utc", "value"],
        rows=[[datetime(2024, 1, 1, 12, 0, tzinfo=UTC), 10.5]],
        provenance=Provenance("silver", "src", "ent"),
    )
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "timestamp_utc,value"
    assert lines[1] == "2024-01-01T12:00:00+00:00,10.5"


def test_write_table_sidecar_metadata(tmp_path: Path) -> None:
    csv_path = tmp_path / "t.csv"
    meta_path = tmp_path / "t.meta.json"
    write_table(
        csv_path,
        header=["value"],
        rows=[[1], [2], [3]],
        provenance=Provenance("gold", "entsoe+open-meteo", "hourly-nl-dataset"),
        meta_path=meta_path,
        meta_extras={"missing_counts": {"load_mw": 0}},
    )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["schema_version"] == "1.0"
    assert meta["columns"] == ["value"]
    assert meta["row_count"] == 3
    assert meta["provenance"]["tier"] == "gold"
    assert meta["missing_counts"] == {"load_mw": 0}


def test_read_table_roundtrips(tmp_path: Path) -> None:
    csv_path = tmp_path / "t.csv"
    write_table(
        csv_path,
        header=["value", "flag"],
        rows=[[1.5, "dup"], [2.0, ""]],
        provenance=Provenance("silver", "s", "e"),
    )
    rows = read_table(csv_path)
    assert rows == [
        {"value": "1.5", "flag": "dup"},
        {"value": "2.0", "flag": ""},
    ]


def test_datetime_and_bool_serialization(tmp_path: Path) -> None:
    csv_path = tmp_path / "t.csv"
    write_table(
        csv_path,
        header=["ts", "flag", "n", "f"],
        rows=[[datetime(2024, 6, 1, 0, 0, tzinfo=UTC), True, None, 2.0]],
        provenance=Provenance("gold", "s", "e", unit="MW"),
    )
    rows = read_table(csv_path)
    assert rows[0]["ts"] == "2024-06-01T00:00:00+00:00"
    assert rows[0]["flag"] == "true"
    assert rows[0]["n"] == ""
    assert rows[0]["f"] == "2.0"


def test_naive_datetime_serialised_as_utc(tmp_path: Path) -> None:
    csv_path = tmp_path / "t.csv"
    write_table(
        csv_path,
        header=["ts"],
        rows=[[datetime(2024, 1, 1)]],
        provenance=Provenance("silver", "s", "e"),
    )
    rows = read_table(csv_path)
    assert rows[0]["ts"] == "2024-01-01T00:00:00+00:00"
