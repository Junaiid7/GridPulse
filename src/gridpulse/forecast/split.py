"""Leakage-safe chronological train/validation/test splitting (Phase 4B).

Time-series splitter. No random shuffling is ever applied: partition
boundaries are chosen purely from the *ordered* set of issue timestamps, so a
model trained on the train window can never observe data from the validation
or test windows, and vice versa.

Ranges are **half-open** ``[start, end)`` and timezone-aware. Validation and
test share their start boundary with the previous window's end, so the whole
timeline is partitioned without gaps and without overlaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

from ..ingestion.common import models as _models


@dataclass(frozen=True)
class ChronologicalSplit:
    """Three contiguous half-open windows over one UTC timeline."""

    train_start: datetime
    train_end: datetime
    validation_end: datetime
    test_end: datetime
    validation_start: datetime  # == train_end (kept explicit for readability)
    test_start: datetime  # == validation_end

    def __post_init__(self):
        ensure_utc = lambda dt: _models.ensure_utc(dt)
        ts = {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "validation_start": self.validation_start,
            "validation_end": self.validation_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
        }
        for name, dt in ts.items():
            if getattr(dt, "tzinfo", None) is None:
                raise ValueError(f"{name} must be timezone-aware (UTC)")
            object.__setattr__(self, name, ensure_utc(dt))
        # Half-open windows share adjacency boundaries; enforce it up front.
        if self.train_end != self.validation_start:
            raise ValueError("train_end must equal validation_start (contiguous windows)")
        if self.validation_end != self.test_start:
            raise ValueError("validation_end must equal test_start (contiguous windows)")
        # Strictly-increasing chain over the *distinct* boundaries: because
        # train_end == validation_start and validation_end == test_start, the
        # effective chain is train_start < train_end < validation_end < test_end.
        chain = (self.train_start, self.train_end, self.validation_end, self.test_end)
        if not all(a < b for a, b in zip(chain, chain[1:])):
            body = " ".join(f"{dt:%Y-%m-%d %H:%M}Z" for dt in chain)
            raise ValueError(f"split must be strictly chronological and non-empty: {body}")

    @property
    def train(self) -> tuple[datetime, datetime]:
        return (self.train_start, self.train_end)

    @property
    def validation(self) -> tuple[datetime, datetime]:
        return (self.validation_start, self.validation_end)

    @property
    def test(self) -> tuple[datetime, datetime]:
        return (self.test_start, self.test_end)

    def to_dict(self) -> dict:
        return {
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "validation_start": self.validation_start.isoformat(),
            "validation_end": self.validation_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "windows": {"train": "(train_start, train_end)", "validation": "(validation_start, validation_end)", "test": "(test_start, test_end)"},
        }


def chronological_split_by_fraction(
    timestamps: Sequence[datetime],
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> ChronologicalSplit:
    """Deterministic chronological split from an arbitrary ordered timeline.

    ``timestamps`` (issue times, any order — sorted internally) are assigned
    to windows in strict time order. The train window holds the earliest
    ``train_fraction``, validation the next ``validation_fraction``, and the
    remaining (freshest) points form the test window. No randomness.

    Boundary timestamps are the *last timestamps* of the preceding window;
    the split's end-exclusive semantics put each timestamp in exactly one
    window. ``test_end`` is ``max + 1h`` (hourly-grid convention).
    """
    if not (0.0 < train_fraction < 1.0 and 0.0 < validation_fraction < 1.0):
        raise ValueError("fractions must be in (0, 1)")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be < 1 (test window needed)")
    unique = sorted({_models.ensure_utc(ts) for ts in timestamps})
    if len(unique) < 3:
        raise ValueError("need at least 3 distinct timestamps to form 3 windows")

    n = len(unique)
    n_val = max(1, int(n * validation_fraction))
    n_train = max(1, int(n * train_fraction))
    n_test = n - n_train - n_val
    if n_test < 1:
        raise ValueError("not enough timestamps to leave a non-empty test window")

    train_start = unique[0]
    train_end = unique[n_train]  # exclusive; == validation start bound
    # Last timestamp assigned to validation:
    validation_end = unique[n_train + n_val]  # exclusive; == test start bound
    test_end = unique[-1] + timedelta(hours=1)  # hourly-grid convention, end-exclusive
    validation_start = train_end
    test_start = validation_end

    return ChronologicalSplit(
        train_start=train_start,
        train_end=train_end,
        validation_start=validation_start,
        validation_end=validation_end,
        test_start=test_start,
        test_end=test_end,
    )


def partition_timestamps(timestamps, split: ChronologicalSplit) -> dict[str, list[datetime]]:
    """Partition issue timestamps into train/validation/test list in time order."""
    out = {"train": [], "validation": [], "test": []}
    for ts in sorted(_models.ensure_utc(t) for t in timestamps):
        if ts < split.train_end:
            out["train"].append(ts)
        elif ts < split.validation_end:
            out["validation"].append(ts)
        else:
            out["test"].append(ts)
    return out


__all__ = ["ChronologicalSplit", "chronological_split_by_fraction", "partition_timestamps"]