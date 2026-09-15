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

from .probabilistic import NOMINAL_INTERVAL_COVERAGE


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


# =====================================================================
# Phase 4C — probabilistic (quantile) evaluation
# =====================================================================


def pinball_loss(
    actuals: Sequence[Optional[float]],
    quantile_preds: Sequence[Optional[float]],
    alpha: float,
) -> Optional[float]:
    """Mean pinball loss at quantile ``alpha``.

    Per-sample pinball: ``(y - q) * alpha`` when ``y >= q`` and
    ``(q - y) * (1 - alpha)`` when ``y < q``. Equivalently
    ``max(alpha*d, (alpha-1)*d)`` with ``d = y - q``. Lower is better; at
    ``alpha=0.5`` it is proportional to MAE. ``None`` samples (either side)
    are skipped. Returns ``None`` when there are no valid pairs.
    """
    if len(actuals) != len(quantile_preds):
        raise ValueError(
            f"length mismatch: {len(actuals)} actuals vs {len(quantile_preds)} predictions"
        )
    pairs = [
        (float(a), float(q))
        for a, q in zip(actuals, quantile_preds)
        if a is not None and q is not None
    ]
    if not pairs:
        return None
    total = 0.0
    for y, q in pairs:
        d = y - q
        total += d * alpha if d >= 0 else d * (alpha - 1.0)
    return total / len(pairs)


def empirical_coverage(
    actuals: Sequence[Optional[float]],
    quantile_preds: Sequence[Optional[float]],
) -> Optional[float]:
    """Fraction of actuals at or below the predicted quantile.

    For a well-calibrated quantile ``alpha`` this fraction ≈ ``alpha`` (so
    P10 empirical coverage should be ≈ 0.10, P50 ≈ 0.50, P90 ≈ 0.90).
    ``None`` samples are skipped; returns ``None`` with no valid pairs.
    """
    pairs = [
        (float(a), float(q))
        for a, q in zip(actuals, quantile_preds)
        if a is not None and q is not None
    ]
    if not pairs:
        return None
    return sum(1.0 if a <= q else 0.0 for a, q in pairs) / len(pairs)


def interval_coverage(
    actuals: Sequence[Optional[float]],
    lo: Sequence[Optional[float]],
    hi: Sequence[Optional[float]],
) -> Optional[float]:
    """Fraction of actuals inside the predictive interval ``[lo, hi]``.

    For the nominal 80% interval ``[P10, P90]`` this should be ≈ 0.80 when
    well calibrated. Rows with any ``None`` are skipped.
    """
    if len(actuals) != len(lo) or len(actuals) != len(hi):
        raise ValueError("actuals, lo and hi must have the same length")
    inside = 0
    n = 0
    for a, l, h in zip(actuals, lo, hi):
        if a is None or l is None or h is None:
            continue
        n += 1
        if l <= float(a) <= h:
            inside += 1
    if n == 0:
        return None
    return inside / n


def interval_width_stats(
    lo: Sequence[Optional[float]],
    hi: Sequence[Optional[float]],
) -> dict:
    """Distribution of interval widths ``hi - lo`` (sharpness).

    Returns ``mean``, ``median``, ``min``, ``max``, ``std`` (population) and
    ``n`` valid widths. All ``None`` (``max=None`` etc.) when no valid widths.
    """
    widths = [
        float(h) - float(l)
        for l, h in zip(lo, hi)
        if l is not None and h is not None
    ]
    empty = {"mean": None, "median": None, "min": None, "max": None, "std": None, "n": 0}
    if not widths:
        return empty
    widths.sort()
    n = len(widths)
    mean = sum(widths) / n
    mid = (n - 1) // 2
    if n % 2 == 1:
        median = widths[mid]
    else:
        median = (widths[mid] + widths[mid + 1]) / 2
    var = sum((w - mean) ** 2 for w in widths) / n
    return {
        "mean": mean,
        "median": median,
        "min": widths[0],
        "max": widths[-1],
        "std": var ** 0.5,
        "n": n,
    }


def _risk_score_stats(p10s, p50s, p90s) -> dict:
    """Mean/median risk score over valid triples; counts near-zero P50."""
    from .probabilistic import P50_EPS, risk_score

    scores = []
    n_p50_valid = 0
    n_near_zero = 0
    for p10, p50, p90 in zip(p10s, p50s, p90s):
        if p50 is None:
            continue
        n_p50_valid += 1
        if abs(float(p50)) < P50_EPS:
            n_near_zero += 1
            continue
        s = risk_score(p10, p50, p90)
        if s is not None:
            scores.append(s)
    empty = {"mean": None, "median": None, "n_p50_valid": n_p50_valid, "n_near_zero_p50": n_near_zero}
    if not scores:
        return empty
    scores.sort()
    n = len(scores)
    mean = sum(scores) / n
    mid = (n - 1) // 2
    median = scores[mid] if n % 2 == 1 else (scores[mid] + scores[mid + 1]) / 2
    return {"mean": mean, "median": median, "n_p50_valid": n_p50_valid, "n_near_zero_p50": n_near_zero}


def probabilistic_metrics(
    actuals: Sequence[Optional[float]],
    p10s: Sequence[Optional[float]],
    p50s: Sequence[Optional[float]],
    p90s: Sequence[Optional[float]],
) -> dict:
    """Aggregate probabilistic evaluation of P10/P50/P90 predictions.

    Requires all four sequences equal-length. Rows where *any* of
    actual/p10/p50/p90 is ``None`` are skipped consistently across every metric
    (no metric can inherit a different valid-pair count).

    Returns a dict with:
    ``n_triples``, ``pinball`` (per quantile), ``empirical_coverage`` (per
    quantile), ``interval_coverage`` (P10..P90), ``nominal_quantiles``,
    ``nominal_interval_coverage``, ``interval_width`` (distribution), and
    ``risk_score`` (mean/median + near-zero-P50 counts).
    """
    if not (len(actuals) == len(p10s) == len(p50s) == len(p90s)):
        raise ValueError("actuals, p10s, p50s and p90s must have the same length")
    valid = [
        (a, lo, mid, hi)
        for a, lo, mid, hi in zip(actuals, p10s, p50s, p90s)
        if None not in (a, lo, mid, hi)
    ]
    a = [t[0] for t in valid]
    lo = [t[1] for t in valid]
    mid = [t[2] for t in valid]
    hi = [t[3] for t in valid]

    def _empty() -> dict:
        return {
            "n_triples": 0,
            "pinball": {"0.10": None, "0.50": None, "0.90": None},
            "empirical_coverage": {"0.10": None, "0.50": None, "0.90": None},
            "interval_coverage": None,
            "nominal_quantiles": {"0.10": 0.10, "0.50": 0.50, "0.90": 0.90},
            "nominal_interval_coverage": NOMINAL_INTERVAL_COVERAGE,
            "interval_width": interval_width_stats(lo, hi),
            "risk_score": {"mean": None, "median": None, "n_p50_valid": 0, "n_near_zero_p50": 0},
        }

    if not valid:
        out = _empty()
        out["interval_width"] = interval_width_stats(lo, hi)
        out["risk_score"] = _risk_score_stats(lo, mid, hi)
        return out

    return {
        "n_triples": len(valid),
        "pinball": {
            "0.10": pinball_loss(a, lo, 0.10),
            "0.50": pinball_loss(a, mid, 0.50),
            "0.90": pinball_loss(a, hi, 0.90),
        },
        "empirical_coverage": {
            "0.10": empirical_coverage(a, lo),
            "0.50": empirical_coverage(a, mid),
            "0.90": empirical_coverage(a, hi),
        },
        "interval_coverage": interval_coverage(a, lo, hi),
        "nominal_quantiles": {"0.10": 0.10, "0.50": 0.50, "0.90": 0.90},
        "nominal_interval_coverage": NOMINAL_INTERVAL_COVERAGE,
        "interval_width": interval_width_stats(lo, hi),
        "risk_score": _risk_score_stats(lo, mid, hi),
    }


__all__ = [
    "compute_point_metrics",
    "bootstrap_mae_ci",
    "bootstrap_mae_difference_ci",
    "pinball_loss",
    "empirical_coverage",
    "interval_coverage",
    "interval_width_stats",
    "probabilistic_metrics",
]