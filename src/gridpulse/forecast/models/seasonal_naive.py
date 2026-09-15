"""Seasonal-naive baseline: prediction(T) = residual_load(T - 24h).

This is the "same-hour-yesterday" persistence forecast — the reference
baseline every GridPulse model must beat on MAE to be meaningful.

The vintage value comes from the forecasting dataset's ``naive_vintage_mw``
column (the issue row's own residual, i.e. ``residual_load(T - lag_hours)``
under the documented zero-observation-latency convention). The model is
stateless — a fitted run only records what was fitted; prediction never
touches the fitted state.
"""

from __future__ import annotations

from .base import ForecastModel, select_rows


class SeasonalNaiveModel(ForecastModel):
    name = "seasonal_naive_24h"

    def __init__(self, *, lag_hours: int = 24):
        self.lag_hours = lag_hours
        self._fitted = False

    def fit(self, dataset, *, start, end):
        self._fitted = True  # stateless — nothing to learn for persistence

    def predict(self, dataset, *, start, end):
        return [r.naive_vintage_mw for r in select_rows(dataset, start, end)]

    def feature_columns(self):
        return ["naive_vintage_mw"]  # the issue row's residual at T - 24h

    def metadata(self):
        return {
            "model": self.name,
            "target": self.target,
            "rule": "prediction(T) = residual_load(T - lag_hours)",
            "lag_hours": self.lag_hours,
            "insufficient_history": (
                "rows in the dataset with no row at target - lag_hours yield "
                "None (never silently filled); they are excluded from metrics"
            ),
            "fit_state": "stateless (persistence baseline)",
        }

    def _restore(self, data):
        self.lag_hours = int(data["metadata"]["lag_hours"])
        self._fitted = True


__all__ = ["SeasonalNaiveModel"]