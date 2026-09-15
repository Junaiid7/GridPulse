"""Evaluation metrics for point forecasts (Phase 4B).

All metrics are pure-Python stdlib. MAPE is guarded near-zero: if
``|actual|`` is below a caller-specified threshold for any sample, those
samples are excluded; ``mape_invalid_count`` records how many were skipped,
and ``mape`` is ``None`` when none are valid. Metrics are never silently
misleading.
"""

from __future__ import annotations

import math
import random
from typing import Mapping, Optional, Sequence


def compute_point_metrics(
    actuals: Sequence[Optional[float]],
    predictions: Sequence[Optional[float]],
    *,
    mape_min_abs: float = 1e-3,
) -> dict:
    """Compute MAE, RMSE, MAPE, bias, median-absolute-error.

    Both lists must be the same length. ``None`` (missing/undetermined)
    entries are skipped (both on the target and the prediction side).
    ``n_predictions`` counts the number of entries where at least one value
    was valid (actual or prediction), for traceability; ``n_valid_pairs``
    counts rows where *both* actual and prediction are non-``None``.

    Parameters
    ----------
    actuals, predictions:
        Parallel sequences of floats or ``None``.
    mape_min_abs:
        Absolute value threshold below which the actual is too close to
        zero for MAPE. Those samples are excluded from MAPE; the count of
        excluded samples is stored in ``mape_invalid_count``.

    Returns
    -------
    dict with keys: ``n_predictions`` (total samples provided),
    ``n_valid_pairs`` (used by error metrics), ``mae``, ``rmse``,
    ``mape`` (``None`` if no valid samples), ``mape_invalid_count``,
    ``bias`` (mean signed error), ``median_absolute_error``.
    """
    n_total = len(actuals)
    if n_total != len(predictions):
        raise ValueError(f"length mismatch: {len(actuals)} actuals vs {len(predictions)} predictions")

    valid_pairs: list[tuple[float, float]] = []
    mape_denom_ok: list[tuple[float, float]] = []
    n_mape_skipped = 0
    for a, p in zip(actuals, predictions):
        if a is None or p is None:
            continue
        valid_pairs.append((a, p))
        if abs(a) >= mape_min_abs:
            mape_denom_ok.append((a, p))
        else:
            n_mape_skipped += 1

    n_valid = len(valid_pairs)
    result: dict = {
        "n_predictions": n_total,
        "n_valid_pairs": n_valid,
        "mae": None,
        "rmse": None,
        "mape": None,
        "mape_invalid_count": n_mape_skipped,
        "bias": None,
        "median_absolute_error": None,
    }
    if n_valid == 0:
        return result

    errors = [p - a for a, p in valid_pairs]
    abs_errors = [abs(e) for e in errors]
    result["mae"] = sum(abs_errors) / n_valid
    result["rmse"] = math.sqrt(sum(e * e for e in errors) / n_valid)
    result["bias"] = sum(errors) / n_valid
    sorted_abs = sorted(abs_errors)
    mid = (n_valid - 1) // 2
    if n_valid % 2 == 1:
        result["median_absolute_error"] = sorted_abs[mid]
    else:
        result["median_absolute_error"] = (sorted_abs[mid] + sorted_abs[mid + 1]) / 2

    if mape_denom_ok:
        pcts = [abs((a - p) / a) for a, p in mape_denom_ok]
        result["mape"] = (sum(pcts) / len(pcts)) * 100.0

    return result


def bootstrap_mae_ci(
    actuals: Sequence[Optional[float]],
    predictions: Sequence[Optional[float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> dict:
    """Bootstrap confidence interval for MAE (percentile method).

    Resamples the *valid pairs* (non-``None`` on both sides) with
    replacement ``n_boot`` times. Returns the point MAE and two-tailed
    percentile CI bounds.

    Returns
    -------
    dict: ``mae`` (point estimate), ``ci_low``, ``ci_high``, ``n_boot``,
    ``ci_level``, ``n_samples`` (number of valid pairs bootstrapped).
    """
    pairs = [(float(a), float(p)) for a, p in zip(actuals, predictions) if a is not None and p is not None]
    n_samples = len(pairs)
    if n_samples == 0:
        return {"mae": None, "ci_low": None, "ci_high": None, "n_boot": n_boot, "ci_level": ci_level, "n_samples": 0}
    rng = random.Random(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        sample = rng.choices(pairs, k=n_samples)
        boots.append(sum(abs(p - a) for a, p in sample) / n_samples)
    boots.sort()
    alpha = (1 - ci_level) / 2
    lo_idx = max(0, int(math.floor((n_boot - 1) * alpha)))
    hi_idx = min(n_boot - 1, int(math.floor((n_boot - 1) * (1 - alpha))))
    point_mae = sum(abs(p - a) for a, p in pairs) / n_samples
    return {"mae": point_mae, "ci_low": boots[lo_idx], "ci_high": boots[hi_idx], "n_boot": n_boot, "ci_level": ci_level, "n_samples": n_samples}


def bootstrap_mae_difference_ci(
    actuals: Sequence[Optional[float]],
    pred_a: Sequence[Optional[float]],
    pred_b: Sequence[Optional[float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> dict:
    """Bootstrap CI for MAE_A - MAE_B.

    A positive difference means model A has *higher* (worse) MAE.
    Samples only rows where both ``pred_a`` and ``pred_b`` and the actual
    are non-``None``.

    Returns
    -------
    dict: ``difference`` (point), ``ci_low``, ``ci_high``, ``n_boot``,
    ``ci_level``, ``n_samples``, ``mae_a``, ``mae_b``.
    """
    if len(actuals) != len(pred_a) or len(actuals) != len(pred_b):
        raise ValueError("all inputs must have the same length")
    triples = [
        (float(a), float(pa), float(pb))
        for a, pa, pb in zip(actuals, pred_a, pred_b)
        if a is not None and pa is not None and pb is not None
    ]
    n_samples = len(triples)
    empty = {"difference": None, "ci_low": None, "ci_high": None, "n_boot": n_boot, "ci_level": ci_level, "n_samples": 0, "mae_a": None, "mae_b": None}
    if n_samples == 0:
        return empty
    rng = random.Random(seed)
    boots: list[float] = []
    point_a = sum(abs(a - pa) for a, pa, _ in triples) / n_samples
    point_b = sum(abs(a - pb) for a, _, pb in triples) / n_samples
    for _ in range(n_boot):
        sample = rng.choices(triples, k=n_samples)
        a_err = sum(abs(a - pa) for a, pa, _ in sample) / n_samples
        b_err = sum(abs(a - pb) for a, _, pb in sample) / n_samples
        boots.append(a_err - b_err)
    boots.sort()
    alpha = (1 - ci_level) / 2
    lo_idx = max(0, int(math.floor((n_boot - 1) * alpha)))
    hi_idx = min(n_boot - 1, int(math.floor((n_boot - 1) * (1 - alpha))))
    return {"difference": point_a - point_b, "ci_low": boots[lo_idx], "ci_high": boots[hi_idx], "n_boot": n_boot, "ci_level": ci_level, "n_samples": n_samples, "mae_a": point_a, "mae_b": point_b}


__all__ = [
    "compute_point_metrics",
    "bootstrap_mae_ci",
    "bootstrap_mae_difference_ci",
]