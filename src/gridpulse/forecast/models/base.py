"""Minimal-fit forecasting models (Phase 4B, pure stdlib).

Model interface (heavily simplified on purpose — no LightGBM yet, per the
phase brief):

- ``fit(dataset, *, start, end)`` — learn anything from the train window only;
- ``predict(dataset, *, start, end)`` — for each issue row in the window,
  return ``float | None`` (``None`` when the model cannot decide, e.g.
  insufficient history or missing features — never a silent fill);
- ``feature_columns()`` — predictors the model consumes;
- ``metadata()`` — reproducible, JSON-able description of the fitted model;
- ``save(path)`` / ``load(path)`` — JSON (``.json``) round-trip.

The split object is *not* given to models; models only ever see half-open
``[start, end)`` UTC windows, and the caller (the benchmark) chooses those
windows to be the train/validation/test windows from
:mod:`gridpulse.forecast.split`. That keeps the boundary — and therefore the
leakage guarantee — explicit at the call site.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

from ...ingestion.common.models import ensure_utc
from ..contract import ForecastingDataset, ForecastRow

Vec = list[Optional[float]]


def select_rows(dataset: ForecastingDataset, start, end) -> list[ForecastRow]:
    """Rows whose issue time is in the half-open window ``[start, end)``."""
    lo, hi = ensure_utc(start), ensure_utc(end)
    if lo >= hi:
        raise ValueError("window must be non-empty and [start, end) with start < end")
    return [r for r in dataset.rows if lo <= r.issue_time < hi]


class ForecastModel:
    """Base class for all step-4 models. Subclasses override the four hooks."""

    target = "residual_load_mw"

    def fit(self, dataset: ForecastingDataset, *, start, end) -> None:
        raise NotImplementedError

    def predict(self, dataset: ForecastingDataset, *, start, end) -> Vec:
        raise NotImplementedError

    def feature_columns(self) -> list:
        return []

    def metadata(self) -> dict:
        return {"model": self.__class__.__name__}

    def save(self, path) -> None:
        payload = {"metadata": self.metadata()}
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=True, indent=2, sort_keys=True)
            fh.write("\n")

    def load(self, path) -> None:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self._restore(data)


__all__ = ["ForecastModel", "select_rows"]