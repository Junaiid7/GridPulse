"""Linear-regression baseline (Phase 4B, pure standard library).

OLS with a tiny ridge on the diagonal (``ridge``, default 1e-6) solved via
normal equations ``(X^T X + ridge * I) beta = X^T y`` with a deterministic
Gaussian-elimination solver (partial pivot). No numpy.

Design decisions (documented here so the phase report can cite them):

- **Candidate features** are the feature table's lag / rolling / calendar /
  weather columns (only those *actually present* in the dataset are used).
- **Standardisation** is applied to predictors using statistics computed on
  the *train window only* (mean/std), then applied to validation/test. This
  keeps the projection leakage-free and makes the tiny ridge meaningful in a
  well-conditioned space.
- **Missing cells**: rows with any ``None`` feature are dropped from the fit
  (``n_dropped`` recorded) and yield ``None`` at prediction time — never a
  silent fill.
- **Insufficient observations**: if ``n_rows < n_columns`` the model refuses
  to fit and records ``fitted = false`` (every prediction returns ``None``).
"""

from __future__ import annotations

from ..contract import ForecastingDataset
from .base import ForecastModel, select_rows


class LinearRegressionModel(ForecastModel):
    name = "linear_regression_ridge"

    def __init__(
        self,
        *,
        feature_columns: list | None = None,
        ridge: float = 1e-6,
        fit_intercept: bool = True,
    ):
        # NOTE: stored on ``_feature_columns`` (not ``feature_columns``) so the
        # public ``feature_columns()`` method is not shadowed by the instance
        # attribute.
        self._feature_columns = list(feature_columns or [])
        self.ridge = float(ridge)
        self.fit_intercept = bool(fit_intercept)
        self._fitted = False
        self._beta: list[float] = []
        self._center: list[float] = []  # per-column mean  (train-only stats)
        self._scale: list[float] = []   # per-column std   (train-only stats)
        self._n_train_rows = 0
        self._n_dropped = 0

    # ------------------------------------------------------------------ fit
    def fit(self, dataset: ForecastingDataset, *, start, end) -> None:
        cols = self._feature_columns
        rows = [r for r in select_rows(dataset, start, end) if r.target_mw is not None]
        full = [r for r in rows if r.has_features()]
        dropped = len(rows) - len(full)
        n_cols = len(cols) + (1 if self.fit_intercept else 0)
        if len(full) < n_cols:
            self._fitted = False
            self._n_train_rows = len(full)
            self._n_dropped = dropped
            return

        design, y = self._build_design(full, cols)
        # Standardise using train-only statistics.
        center, scale = _column_stats(design)
        Xs = _standardise(design, center, scale)
        X = _append_ones(Xs) if self.fit_intercept else Xs
        self._beta = _ridge_ols(X, y, self.ridge)
        self._center, self._scale = center, scale
        self._fitted = True
        self._n_train_rows = len(full)
        self._n_dropped = dropped

    # -------------------------------------------------------------- predict
    def predict(self, dataset: ForecastingDataset, *, start, end):
        if not self._fitted:
            n = len(select_rows(dataset, start, end))
            return [None] * n
        cols = self._feature_columns
        out = []
        for row in select_rows(dataset, start, end):
            feats = [row.features[c] for c in cols]
            if any(v is None for v in feats):
                out.append(None)  # missing predictor -> undetermined, never filled
                continue
            xs = [float(v) for v in feats]
            Xs = _standardise_one(xs, self._center, self._scale)
            if self.fit_intercept:
                Xs = [1.0] + Xs
            out.append(sum(b * x for b, x in zip(self._beta, Xs)))
        return out

    # ----------------------------------------------------------- interface
    def feature_columns(self):
        return list(self._feature_columns)

    def metadata(self) -> dict:
        return {
            "model": self.name,
            "target": self.target,
            "feature_columns": list(self._feature_columns),
            "ridge": self.ridge,
            "fit_intercept": self.fit_intercept,
            "standardisation": "train-only mean/std",
            "fitted": self._fitted,
            "n_train_rows": self._n_train_rows,
            "n_rows_dropped_missing_features": self._n_dropped,
            "beta_coefficients": [float(b) for b in self._beta],
            "predictor_center_train_only": [float(v) for v in self._center],
            "predictor_scale_train_only": [float(v) for v in self._scale],
            "missing_cell_policy": "rows with any None feature are dropped from fit and yield None at predict (never silently filled)",
        }

    def _restore(self, data) -> None:
        m = data["metadata"]
        self._feature_columns = list(m["feature_columns"])
        self.ridge = float(m["ridge"])
        self.fit_intercept = bool(m["fit_intercept"])
        self._beta = [float(b) for b in m["beta_coefficients"]]
        self._center = [float(v) for v in m["predictor_center_train_only"]]
        self._scale = [float(v) for v in m["predictor_scale_train_only"]]
        self._n_train_rows = int(m["n_train_rows"])
        self._n_dropped = int(m["n_rows_dropped_missing_features"])
        self._fitted = bool(m["fitted"])

    # -------------------------------------------------------------- helpers
    def _build_design(self, rows, cols):
        design = [[float(r.features[c]) for c in cols] for r in rows]
        y = list(r.target_mw for r in rows)
        return design, y


# ---------------------------------------------------------------- linalg
def _column_stats(design):
    n = len(design)
    p = len(design[0])
    mean = [sum(row[j] for row in design) / n for j in range(p)]
    # population std (ddof=0) — deterministic and sufficient for scaling
    var = [sum((row[j] - mean[j]) ** 2 for row in design) / n for j in range(p)]
    scale = [v ** 0.5 if v > 0 else 1.0 for v in var]
    return mean, scale


def _standardise(design, center, scale):
    return [[(x - m) / s for x, m, s in zip(row, center, scale)] for row in design]


def _standardise_one(xs, center, scale):
    return [(x - m) / s for x, m, s in zip(xs, center, scale)]


def _append_ones(design):
    return [[1.0] + row for row in design]


def _ridge_ols(design, y, ridge):
    """Solve (X^T X + ridge * I) beta = X^T y via Gaussian elimination."""
    p = len(design[0])
    n = len(design)
    # Gram matrix + r.h.s. in one augmented matrix.
    a = [[0.0] * (p + 1) for _ in range(p)]
    for i in range(p):
        for j in range(p):
            s = 0.0
            for r in range(n):
                s += design[r][i] * design[r][j]
            a[i][j] = s
        a[i][i] += ridge
        b = 0.0
        for r in range(n):
            b += design[r][i] * y[r]
        a[i][p] = b
    return _gauss_solve(a)


def _gauss_solve(a):
    """Solve a square augmented matrix in place; partial pivot. Pure Python."""
    a = [row[:] for row in a]
    n = len(a)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-300:
            raise ValueError("singular design matrix; refusing to solve")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        pivot_val = a[col][col]
        for c in range(col, n + 1):
            a[col][c] /= pivot_val
        for r in range(n):
            if r == col:
                continue
            factor = a[r][col]
            if factor != 0.0:
                for c in range(col, n + 1):
                    a[r][c] -= factor * a[col][c]
    return [a[i][n] for i in range(n)]


__all__ = ["LinearRegressionModel"]