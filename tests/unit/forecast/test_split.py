"""Chronological train/validation/test splitter: no random shuffle, reject
invalid (overlapping / non-contiguous / naive / empty) boundaries."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gridpulse.forecast.split import (
    ChronologicalSplit,
    chronological_split_by_fraction,
    partition_timestamps,
)

UTC = UTC


def _daily_issues(
    n: int, start: datetime = datetime(2024, 1, 1, tzinfo=UTC)
) -> list[datetime]:
    return [start + timedelta(hours=24 * i) for i in range(n)]


def test_fraction_split_is_deterministic():
    ts = _daily_issues(58)
    a = chronological_split_by_fraction(
        ts, train_fraction=0.7, validation_fraction=0.15
    )
    b = chronological_split_by_fraction(
        list(reversed(ts)), train_fraction=0.7, validation_fraction=0.15
    )
    assert a == b  # order-insensitive, deterministic
    assert a.to_dict() == b.to_dict()


def test_partition_preserves_order_and_covers_all():
    ts = _daily_issues(58)
    split = chronological_split_by_fraction(ts)
    parts = partition_timestamps(ts, split)
    for name in ("train", "validation", "test"):
        times = parts[name]
        assert times == sorted(times)
        assert len(times) > 0
    union = parts["train"] + parts["validation"] + parts["test"]
    assert sorted(times for times in union) == sorted(ts)


def test_test_window_is_strictly_newest_and_non_empty():
    ts = _daily_issues(58)
    split = chronological_split_by_fraction(ts)
    parts = partition_timestamps(ts, split)
    assert max(parts["train"]) < min(parts["validation"])
    assert max(parts["validation"]) < min(parts["test"])
    # Half-open windows: a timestamp equal to a boundary belongs to the NEXT window.
    boundary = parts["train"][-1]
    assert boundary < split.train_end
    assert partition_timestamps([split.train_end], split)["validation"] == [
        split.train_end
    ]


def test_tiny_inputs_rejected():
    with pytest.raises(ValueError):
        chronological_split_by_fraction(_daily_issues(2))
    with pytest.raises(ValueError):
        chronological_split_by_fraction([])


def test_contiguity_enforced():
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        ChronologicalSplit(
            train_start=t0,
            train_end=t0 + timedelta(hours=24),
            validation_start=t0 + timedelta(hours=25),  # gap
            validation_end=t0 + timedelta(hours=49),
            test_start=t0 + timedelta(hours=49),
            test_end=t0 + timedelta(hours=73),
        )


def test_overlap_rejected():
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        ChronologicalSplit(
            train_start=t0,
            train_end=t0 + timedelta(hours=24),
            validation_start=t0 + timedelta(hours=12),  # overlapping start
            validation_end=t0 + timedelta(hours=48),
            test_start=t0 + timedelta(hours=48),
            test_end=t0 + timedelta(hours=72),
        )


def test_naive_boundaries_rejected():
    t0 = datetime(2024, 1, 1)  # naive
    with pytest.raises(ValueError):
        ChronologicalSplit(
            train_start=t0,
            train_end=t0,
            validation_start=t0,
            validation_end=t0,
            test_start=t0,
            test_end=t0,
        )


def test_empty_window_rejected():
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        ChronologicalSplit(
            train_start=t0,
            train_end=t0 + timedelta(hours=24),
            validation_start=t0 + timedelta(hours=24),
            validation_end=t0 + timedelta(hours=24),  # zero-length validation
            test_start=t0 + timedelta(hours=24),
            test_end=t0 + timedelta(hours=48),
        )
