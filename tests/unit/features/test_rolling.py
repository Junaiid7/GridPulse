"""Unit tests for backward-looking rolling features."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gridpulse.features.asof import HistoryPoint
from gridpulse.features.rolling import rolling_mean, rolling_std, rolling_values

UTC = UTC


def _pt(s: str, v: float) -> HistoryPoint:
    return HistoryPoint(datetime.fromisoformat(s).astimezone(UTC), v)


def _hourly_history(start: str, count: int) -> list[HistoryPoint]:
    from datetime import timedelta

    t0 = datetime.fromisoformat(start).astimezone(UTC)
    return [_pt((t0 + timedelta(hours=i)).isoformat(), float(i)) for i in range(count)]


def test_rolling_values_window_6() -> None:
    # Hourly history 00:00..05:00; as_of 06:00, window 6h → [00:00,01:00..05:00].
    history = _hourly_history("2024-01-01T00:00:00Z", 6)
    values = rolling_values(history, _dt("2024-01-01T06:00:00Z"), window_hours=6)
    assert values == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_rolling_values_excludes_cutoff_row() -> None:
    # A point exactly at as_of must never be included.
    history = [
        _pt("2024-01-01T11:00:00Z", 10.0),
        _pt("2024-01-01T12:00:00Z", 20.0),  # at cutoff
        _pt("2024-01-01T13:00:00Z", 30.0),
    ]
    values = rolling_values(history, _dt("2024-01-01T12:00:00Z"), window_hours=24)
    assert 20.0 not in values
    assert values == [10.0]


def test_rolling_mean_empty() -> None:
    assert rolling_mean([], _dt("2024-01-01T12:00:00Z"), 24) is None


def test_rolling_mean_computed() -> None:
    history = [_pt("2024-01-01T11:00:00Z", 10.0), _pt("2024-01-01T11:30:00Z", 20.0)]
    assert rolling_mean(history, _dt("2024-01-01T12:00:00Z"), 24) == 15.0


def test_rolling_std_none_for_one_value() -> None:
    history = [_pt("2024-01-01T11:00:00Z", 10.0)]
    assert rolling_std(history, _dt("2024-01-01T12:00:00Z"), 24) is None


def test_rolling_std_computed() -> None:
    import statistics

    history = [_pt("2024-01-01T11:00:00Z", 10.0), _pt("2024-01-01T11:30:00Z", 20.0)]
    expected = statistics.stdev([10.0, 20.0])
    assert rolling_std(history, _dt("2024-01-01T12:00:00Z"), 24) == pytest.approx(
        expected
    )


def test_window_zero_or_negative_raises() -> None:
    history = [_pt("2024-01-01T11:00:00Z", 1.0)]
    with pytest.raises(ValueError, match="positive"):
        rolling_values(history, _dt("2024-01-01T12:00:00Z"), 0)
    with pytest.raises(ValueError, match="positive"):
        rolling_values(history, _dt("2024-01-01T12:00:00Z"), -1)


def test_rolling_values_narrow_window() -> None:
    history = _hourly_history("2024-01-01T00:00:00Z", 24)
    values = rolling_values(history, _dt("2024-01-02T00:00:00Z"), window_hours=3)
    assert len(values) == 3
    assert values == [21.0, 22.0, 23.0]


# Helper needed for other tests
def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s).astimezone(UTC)
