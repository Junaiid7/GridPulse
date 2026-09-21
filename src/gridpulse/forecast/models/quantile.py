"""LightGBM quantile-regression model for P10/P50/P90 (Phase 4C).

Three LightGBM gradient-boosted trees are fit on the **train window only** —
one per quantile (``alpha`` = 0.10, 0.50, 0.90) using LightGBM's native
``objective='quantile'`` pinball objective. All predictors come from the
leakage-safe issue-row feature columns (identical set to the Phase 4B ridge
baseline), so the information-cutoff guarantee carries over unchanged.

Design decisions (documented here so the phase report can cite them):

- **Train-only fit**: ``fit()`` only ever sees rows in ``[start, end)``. The
  caller (the benchmark) supplies the train window; validation/test rows are
  never passed to ``fit``.
- **Native LightGBM API** (``lgb.train`` / ``lgb.Booster``): the scikit-learn
  wrapper (``LGBMRegressor``) pulls in scikit-learn as an extra dependency;
  the native API needs only numpy (already LightGBM's transitive requirement),
  so no unrelated package is added.
- **Reproducibility**: ``seed``, ``deterministic=True``,
  ``force_row_wise=True``, single thread (``num_threads=1``). These make tree
  construction bit-for-bit reproducible on the same features.
- **Missing cells**: rows with any ``None`` feature are dropped from the fit
  (``n_rows_dropped_missing_features``) and yield an all-``None`` triple at
  prediction time — never a fabricated fill.
- **Insufficient observations**: if fewer than ``min_train_rows`` (default 5)
  complete rows are available, the model refuses to fit and records
  ``fitted=false``; every prediction is returned as an all-``None`` triple.
- **Quantile crossing**: raw triples are checked with
  :func:`gridpulse.forecast.probabilistic.detect_quantile_crossing` and
  corrected with :func:`correct_quantiles`, and the numbers are recorded in
  ``crossing_stats()`` (crossing is never silently hidden). Valid ordering is
  then enforced by the ``QuantileForecast`` contract.

Interface note: ``predict`` returns a list of
:class:`~gridpulse.forecast.probabilistic.QuantileForecast` (one per issue row
in the window) instead of the scalar ``float | None`` vector of the Phase 4B
point models; this is the probabilistic analogue of the same row interface.
"""

from __future__ import annotations

import base64
import json

import lightgbm as lgb
import numpy as np

from ..contract import ForecastingDataset
from ..probabilistic import (
    QUANTILES,
    QuantileForecast,
    correct_quantiles,
    detect_quantile_crossing,
)
from .base import ForecastModel, select_rows


class QuantileRegressionModel(ForecastModel):
    """LightGBM quantile-regression model producing P10/P50/P90 predictions."""

    name = "quantile_regression_lgbm"
    quantiles: tuple = QUANTILES

    def __init__(
        self,
        *,
        feature_columns: list | None = None,
        quantiles: tuple = QUANTILES,
        n_estimators: int = 120,
        learning_rate: float = 0.05,
        num_leaves: int = 15,
        min_child_samples: int = 3,
        random_state: int = 0,
        boost_from_average: bool = True,
        min_train_rows: int = 5,
    ):
        self._feature_columns = list(feature_columns or [])
        self.quantiles = tuple(sorted(float(q) for q in quantiles))
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.num_leaves = int(num_leaves)
        self.min_child_samples = int(min_child_samples)
        self.random_state = int(random_state)
        self.boost_from_average = bool(boost_from_average)
        self.min_train_rows = int(min_train_rows)
        self._fitted = False
        self._boosters: dict[float, lgb.Booster] = {}
        self._n_train_rows = 0
        self._n_dropped = 0
        self._crossing: dict = {
            "n_detected": 0,
            "n_corrected": 0,
            "n_triples": 0,
        }

    # ------------------------------------------------------------------ fit
    def fit(self, dataset: ForecastingDataset, *, start, end) -> None:
        cols = self._feature_columns
        rows = [r for r in select_rows(dataset, start, end) if r.target_mw is not None]
        full = [r for r in rows if r.has_features()]
        dropped = len(rows) - len(full)
        if len(full) < self.min_train_rows:
            self._fitted = False
            self._n_train_rows = len(full)
            self._n_dropped = dropped
            self._boosters = {}
            self._crossing = {"n_detected": 0, "n_corrected": 0, "n_triples": 0}
            return

        xs = np.asarray(
            [[float(r.features[c]) for c in cols] for r in full],
            dtype="float64",
        )
        ys = np.asarray([float(r.target_mw) for r in full], dtype="float64")

        common = dict(
            objective="quantile",
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            min_data_in_leaf=self.min_child_samples,
            subsample=1.0,          # deterministic
            subsample_freq=0,       # deterministic (no random bagging)
            colsample_bytree=1.0,   # deterministic
            seed=self.random_state,
            deterministic=True,
            force_row_wise=True,
            num_threads=1,
            boost_from_average=self.boost_from_average,
            verbosity=-1,
        )
        self._boosters = {}
        for alpha in self.quantiles:
            params = dict(common, alpha=float(alpha))
            train_set = lgb.Dataset(xs, label=ys)
            booster = lgb.train(
                params,
                train_set,
                num_boost_round=self.n_estimators,
            )
            self._boosters[float(alpha)] = booster

        self._fitted = True
        self._n_train_rows = len(full)
        self._n_dropped = dropped

    # -------------------------------------------------------------- predict
    def predict(self, dataset: ForecastingDataset, *, start, end):
        """Return a list of :class:`QuantileForecast` aligned to issue rows."""
        rows = select_rows(dataset, start, end)
        cols = self._feature_columns

        if not self._fitted or not self._boosters:
            self._crossing = {"n_detected": 0, "n_corrected": 0, "n_triples": 0}
            return [
                QuantileForecast(issue_time=r.issue_time, target_time=r.target_time)
                for r in rows
            ]

        valid_idx = [
            i
            for i, r in enumerate(rows)
            if all(r.features.get(c) is not None for c in cols)
        ]
        # Pre-fill with undefined triples.
        raw: dict[float, list[float | None]] = {
            alpha: [None] * len(rows) for alpha in self.quantiles
        }
        if valid_idx:
            xs = np.asarray(
                [[float(rows[i].features[c]) for c in cols] for i in valid_idx],
                dtype="float64",
            )
            for alpha in self.quantiles:
                arr = self._boosters[float(alpha)].predict(xs)
                for k, i in enumerate(valid_idx):
                    raw[float(alpha)][i] = float(arr[k])

        out: list[QuantileForecast] = []
        crossing = {"n_detected": 0, "n_corrected": 0, "n_triples": 0}
        for i, r in enumerate(rows):
            p10, p50, p90 = raw[0.10][i], raw[0.50][i], raw[0.90][i]
            if all(v is not None for v in (p10, p50, p90)):
                crossing["n_triples"] += 1
                if detect_quantile_crossing(p10, p50, p90):
                    crossing["n_detected"] += 1
                p10, p50, p90, corrected = correct_quantiles(p10, p50, p90)
                if corrected:
                    crossing["n_corrected"] += 1
            out.append(
                QuantileForecast(
                    issue_time=r.issue_time,
                    target_time=r.target_time,
                    p10=p10,
                    p50=p50,
                    p90=p90,
                )
            )
        self._crossing = crossing
        return out

    def crossing_stats(self) -> dict:
        """How often raw quantile triples crossed on the last ``predict`` call."""
        return dict(self._crossing)

    # ----------------------------------------------------------- interface
    def feature_columns(self) -> list:
        return list(self._feature_columns)

    def metadata(self) -> dict:
        return {
            "model": self.name,
            "target": self.target,
            "library": f"lightgbm=={lgb.__version__}",
            "quantiles": [float(a) for a in self.quantiles],
            "feature_columns": list(self._feature_columns),
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "random_state": self.random_state,
            "boost_from_average": self.boost_from_average,
            "min_train_rows": self.min_train_rows,
            "objective": "quantile (pinball)",
            "deterministic": True,
            "fitted": self._fitted,
            "n_train_rows": self._n_train_rows,
            "n_rows_dropped_missing_features": self._n_dropped,
            "missing_cell_policy": (
                "rows with any None feature are dropped from fit and yield an "
                "all-None P10/P50/P90 triple at predict (never silently filled)"
            ),
            "crossing_policy": (
                "raw triples are detected with detect_quantile_crossing, "
                "corrected by deterministic ascending sort (correct_quantiles); "
                "counts are reported via crossing_stats()"
            ),
        }

    def save(self, path) -> None:
        payload = {
            "metadata": self.metadata(),
            "boosters": [
                {
                    "quantile": alpha,
                    "model_str_b64": base64.b64encode(
                        self._boosters[alpha].model_to_string().encode("utf-8")
                    ).decode("ascii"),
                }
                for alpha in self.quantiles
                if self._fitted and alpha in self._boosters
            ],
        }
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, ensure_ascii=True, indent=2, sort_keys=True)
            fh.write("\n")

    def load(self, path) -> None:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        m = data["metadata"]
        self._feature_columns = list(m["feature_columns"])
        self.quantiles = tuple(float(q) for q in m["quantiles"])
        self.n_estimators = int(m["n_estimators"])
        self.learning_rate = float(m["learning_rate"])
        self.num_leaves = int(m["num_leaves"])
        self.min_child_samples = int(m["min_child_samples"])
        self.random_state = int(m["random_state"])
        self.boost_from_average = bool(m.get("boost_from_average", True))
        self.min_train_rows = int(m["min_train_rows"])
        self._n_train_rows = int(m["n_train_rows"])
        self._n_dropped = int(m["n_rows_dropped_missing_features"])
        self._fitted = bool(m["fitted"])
        self._boosters = {}
        if self._fitted:
            for entry in data["boosters"]:
                alpha = float(entry["quantile"])
                text = base64.b64decode(entry["model_str_b64"]).decode("utf-8")
                self._boosters[alpha] = lgb.Booster(model_str=text)
        self._crossing = {"n_detected": 0, "n_corrected": 0, "n_triples": 0}

    def _restore(self, data) -> None:  # pragma: no cover - load() bypasses base
        raise NotImplementedError("QuantileRegressionModel uses save()/load() directly")

    # -------------------------------------------------------------- helpers
    def to_dict(self) -> dict:
        """Plain-JSON summary used by the benchmark (no secrets ever)."""
        return {
            "name": self.name,
            "quantiles": [float(a) for a in self.quantiles],
            "feature_columns": list(self._feature_columns),
            "fitted": self._fitted,
            "n_train_rows": self._n_train_rows,
            "n_rows_dropped_missing_features": self._n_dropped,
        }


__all__ = ["QuantileRegressionModel"]