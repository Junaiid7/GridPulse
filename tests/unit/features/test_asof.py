"""Unit tests for the information-cut guard: history_before()."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gridpulse.features.asof import HistoryPoint, history_before

UTC = UTC


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)


def test_only_strictly_before_as_of() -> None:
    history = [
        HistoryPoint(_dt("2024-01-01T11:59:00Z"), 1.0),
        HistoryPoint(_dt("2024-01-01T12:00:00Z"), 2.0),
        HistoryPoint(_dt("2024-01-01T12:01:00Z"), 3.0),
    ]
    result = history_before(history, _dt("2024-01-01T12:00:00Z"))
    assert len(result) == 1
    assert result[0].value == 1.0


def test_includes_tuple_pairs() -> None:
    history = [(_dt("2024-01-01T11:00:00Z"), 10.0), (_dt("2024-01-01T11:30:00Z"), 20.0)]
    result = history_before(history, _dt("2024-01-01T12:00:00Z"))
    assert len(result) == 2
    assert result[1].value == 20.0


def test_naive_as_of_raises() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        history_before([], datetime(2024, 1, 1))


def test_naive_history_timestamp_raises() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        history_before([(datetime(2024, 1, 1), 1.0)], _dt("2024-01-01T12:00:00Z"))


def test_non_utc_history_converted_to_utc() -> None:
    # A point at 02:00+01:00 == 01:00 UTC; as_of = 00:30 UTC → point is excluded.
    p = HistoryPoint(_dt("2024-01-01T01:30:00+01:00"), 5.0)  # 2024-01-01 00:30Z
    result = history_before([p], _dt("2024-01-01T00:30:00Z"))
    assert result == []
    result2 = history_before([p], _dt("2024-01-01T00:31:00Z"))
    assert len(result2) == 1


def test_empty_history_returns_empty_list() -> None:
    assert history_before([], _dt("2024-01-01T12:00:00Z")) == []


def test_order_preserved() -> None:
    history = [
        HistoryPoint(_dt("2024-01-01T10:00:00Z"), 3.0),
        HistoryPoint(_dt("2024-01-01T08:00:00Z"), 1.0),
        HistoryPoint(_dt("2024-01-01T09:00:00Z"), 2.0),
    ]
    result = history_before(history, _dt("2024-01-01T12:00:00Z"))
    assert [r.value for r in result] == [3.0, 1.0, 2.0]
