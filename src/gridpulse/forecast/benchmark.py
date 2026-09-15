"""Baseline comparison A (seasonal naive) vs B (linear regression), Phase 4B.

Outputs a structured :class:`BenchmarkResult` that always carries an explicit
``data_status``; this phase runs only on fixtures, so it is
``"FIXTURE-VERIFIED"`` — never claimed as live-validated. When ENTSO-E access
arrives, the exact same code path runs on the real feature table and the flag
flips, but the code never overclaims by itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from .contract import build_forecasting_dataset
from .evaluate import (
    compute_point_metrics,
    bootstrap_mae_ci,
    bootstrap_mae_difference_ci,
    probabilistic_metrics,
)
from .models import LinearRegressionModel, QuantileRegressionModel, SeasonalNaiveModel
from .split import ChronologicalSplit, chronological_split_by_fraction

#: Candidate predictor columns for model B (subset actually present in the
#: feature table is chosen at fit time). Calendar + lags + rolling.
DEFAULT_LR_FEATURES = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_holiday",
    "lag_1h",
    "lag_24h",
    "lag_48h",
    "lag_168h",
    "rolling_mean_24h",
    "rolling_std_24h",
    "rolling_mean_168h",
]

#: Threshold below which MAPE is treated as undefined for a sample.
MAPE_MIN_ABS = 1e-3


def run_benchmark(
    feature_rows,
    *,
    split: Optional[ChronologicalSplit] = None,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 0,
    mape_min_abs: float = MAPE_MIN_ABS,
    issue_hour_utc: int = 6,
) -> "BenchmarkResult":
    """Run the phase-8 comparison and return the structured result.

    ``feature_rows`` are the pipeline feature-table rows (``read_table``
    output). The forecast dataset is built with the contract defaults, split
    chronologically (a caller-provided ``split`` is honoured verbatim, else a
    deterministic fraction split), both baselines are fit on the train window
    and evaluated on the *test* window. Model A's and model B's predictions
    are aligned on rows where BOTH models produced a value and the actual
    exists.

    Bootstrap CIs (MAE_A, MAE_B, and MAE_A - MAE_B) use ``seed`` for
    reproducibility.
    """
    dataset = build_forecasting_dataset(
        feature_rows, issue_hour_utc=issue_hour_utc, horizon_hours=24, naive_lag_hours=24
    )
    if len(dataset) == 0:
        raise ValueError("forecasting dataset is empty; cannot benchmark")

    if split is None:
        split = chronological_split_by_fraction(
            dataset.issue_times,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )

    features_present = set(dataset.predictor_columns)
    lr_features = [c for c in DEFAULT_LR_FEATURES if c in features_present]
    if not lr_features:
        raise ValueError("no default LR features present in the forecasting dataset")

    model_a = SeasonalNaiveModel(lag_hours=24)
    model_b = LinearRegressionModel(feature_columns=lr_features, ridge=1e-6)

    model_a.fit(dataset, start=split.train_start, end=split.train_end)
    model_b.fit(dataset, start=split.train_start, end=split.train_end)

    test_actuals = [
        r.target_mw
        for r in dataset.rows
        if split.test_start <= r.issue_time < split.test_end
    ]
    test_pred_a = model_a.predict(dataset, start=split.test_start, end=split.test_end)
    test_pred_b = model_b.predict(dataset, start=split.test_start, end=split.test_end)

    # Align both predictors on rows where each model is defined.
    aligned_actuals, aligned_a, aligned_b = _align(test_actuals, test_pred_a, test_pred_b)
    n_aligned = len(aligned_actuals)

    metrics_a = compute_point_metrics(aligned_actuals, aligned_a, mape_min_abs=mape_min_abs)
    metrics_b = compute_point_metrics(aligned_actuals, aligned_b, mape_min_abs=mape_min_abs)

    ci_a = bootstrap_mae_ci(aligned_actuals, aligned_a, seed=seed)
    ci_b = bootstrap_mae_ci(aligned_actuals, aligned_b, seed=seed)
    ci_diff = bootstrap_mae_difference_ci(aligned_actuals, aligned_a, aligned_b, seed=seed)

    model_results = [
        {
            "model": "A",
            "name": model_a.name,
            "rule": "prediction(T) = residual_load(T - 24h)",
            "metrics": metrics_a,
            "mae_ci": ci_a,
            "feature_columns": model_a.feature_columns(),
        },
        {
            "model": "B",
            "name": model_b.name,
            "rule": f"ridge linear regression on {', '.join(lr_features)}",
            "metrics": metrics_b,
            "mae_ci": ci_b,
            "feature_columns": lr_features,
            "fit": _summarise_fit(model_b.metadata()),
        },
    ]

    comparison = {
        "direction": "MAE_A - MAE_B",
        "difference": ci_diff["difference"],
        "ci_low": ci_diff["ci_low"],
        "ci_high": ci_diff["ci_high"],
        "ci_level": ci_diff["ci_level"],
        "n_boot": ci_diff["n_boot"],
        "method": "percentile bootstrap on aligned test pairs",
        "note": (
            "positive difference => model A ('seasonal naive') has higher MAE. "
            "An interval excluding 0 indicates a statistically suggestive "
            "difference on this fixture-only evaluation."
        ),
    }

    partition = _partition_counts(dataset, split)
    reproducibility = {
        "seed": seed,
        "gridpulse_version": _gridpulse_version(),
        "python_version": _python_version(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    info = {
        "data_status": "FIXTURE-VERIFIED",
        "target_column": dataset.metadata["target_column"],
        "horizon_hours": dataset.metadata["horizon_hours"],
        "naive_lag_hours": dataset.metadata["naive_lag_hours"],
        "issue_hour_utc": dataset.metadata["issue_hour_utc"],
        "asof_policy": dataset.metadata["asof_policy"],
        "n_forecast_rows": len(dataset),
        "predictor_columns": list(dataset.predictor_columns),
        "n_pred_a_defined": sum(p is not None for p in test_pred_a),
        "n_pred_b_defined": sum(p is not None for p in test_pred_b),
        "n_test_row_candidates": len(test_actuals),
        "n_aligned_rows_evaluated": n_aligned,
        "n_mape_invalid_a": metrics_a["mape_invalid_count"],
        "n_mape_invalid_b": metrics_b["mape_invalid_count"],
        "mape_min_abs": mape_min_abs,
        "statement": (
            "UNVERIFIED for live electricity: there is no ENTSO-E API key; "
            "the feature table behind this benchmark was generated from "
            "hand-crafted synthetic fixtures exercising the real pipeline. "
            "Weather sources are real Open-Meteo data in live runs, but this "
            "offline benchmark uses synthetic payloads end-to-end."
        ),
    }

    return BenchmarkResult(
        models=model_results,
        comparison=comparison,
        split=split.to_dict(),
        train_validation_test_counts=partition,
        info=info,
        reproducibility=reproducibility,
    )


# =====================================================================
# Phase 4C — probabilistic benchmark (A / B / C)
# =====================================================================


def run_probabilistic_benchmark(
    feature_rows,
    *,
    split: Optional[ChronologicalSplit] = None,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 0,
    mape_min_abs: float = MAPE_MIN_ABS,
    issue_hour_utc: int = 6,
    quantile_model_kwargs: Optional[dict] = None,
) -> "ProbabilisticBenchmarkResult":
    """Phase 4C comparison A / B / C.

    A = seasonal naive 24h (point), B = ridge linear regression (point),
    C = LightGBM quantile regression producing P10/P50/P90. All models are fit
    on the **train window only** and evaluated on the **test window**. C's
    point metrics come from its P50 median; its probabilistic metrics
    (pinball per quantile, empirical coverage, interval coverage/width, risk
    score) are evaluated on the same aligned test rows. DATA STATUS remains
    ``FIXTURE-VERIFIED`` — synthetic fixtures, never live electricity.
    """
    dataset = build_forecasting_dataset(
        feature_rows, issue_hour_utc=issue_hour_utc, horizon_hours=24, naive_lag_hours=24
    )
    if len(dataset) == 0:
        raise ValueError("forecasting dataset is empty; cannot benchmark")

    if split is None:
        split = chronological_split_by_fraction(
            dataset.issue_times,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )

    features_present = set(dataset.predictor_columns)
    lr_features = [c for c in DEFAULT_LR_FEATURES if c in features_present]
    if not lr_features:
        raise ValueError("no default LR features present in the forecasting dataset")

    n_test_rows = len([r for r in dataset.rows if split.test_start <= r.issue_time < split.test_end])

    # --- fit on train window only -------------------------------------
    model_a = SeasonalNaiveModel(lag_hours=24)
    model_b = LinearRegressionModel(feature_columns=lr_features, ridge=1e-6)
    model_c = QuantileRegressionModel(
        feature_columns=lr_features, random_state=seed, **(quantile_model_kwargs or {})
    )
    model_a.fit(dataset, start=split.train_start, end=split.train_end)
    model_b.fit(dataset, start=split.train_start, end=split.train_end)
    model_c.fit(dataset, start=split.train_start, end=split.train_end)

    test_actuals = [
        r.target_mw
        for r in dataset.rows
        if split.test_start <= r.issue_time < split.test_end
    ]
    pred_a = model_a.predict(dataset, start=split.test_start, end=split.test_end)
    pred_b = model_b.predict(dataset, start=split.test_start, end=split.test_end)
    fc_c = model_c.predict(dataset, start=split.test_start, end=split.test_end)

    p10 = [f.p10 for f in fc_c]
    p50 = [f.p50 for f in fc_c]
    p90 = [f.p90 for f in fc_c]

    # Align on rows where the actual AND every model produced a value.
    aligned = [
        (a, pa, pb, lo, mid, hi)
        for a, pa, pb, lo, mid, hi in zip(test_actuals, pred_a, pred_b, p10, p50, p90)
        if None not in (a, pa, pb, lo, mid, hi)
    ]
    aa = [t[0] for t in aligned]
    ap_a = [t[1] for t in aligned]
    ap_b = [t[2] for t in aligned]
    ap_lo = [t[3] for t in aligned]
    ap_mid = [t[4] for t in aligned]
    ap_hi = [t[5] for t in aligned]
    n_aligned = len(aa)

    metrics_a = compute_point_metrics(aa, ap_a, mape_min_abs=mape_min_abs)
    metrics_b = compute_point_metrics(aa, ap_b, mape_min_abs=mape_min_abs)
    metrics_c_point = compute_point_metrics(aa, ap_mid, mape_min_abs=mape_min_abs)
    metrics_c_prob = probabilistic_metrics(aa, ap_lo, ap_mid, ap_hi)

    ci_diff = bootstrap_mae_difference_ci(aa, ap_a, ap_b, seed=seed)
    crossing = model_c.crossing_stats()

    model_results = [
        {
            "model": "A",
            "name": model_a.name,
            "kind": "point",
            "rule": "prediction(T) = residual_load(T - 24h)",
            "metrics": metrics_a,
            "mae_ci": bootstrap_mae_ci(aa, ap_a, seed=seed),
            "feature_columns": model_a.feature_columns(),
        },
        {
            "model": "B",
            "name": model_b.name,
            "kind": "point",
            "rule": f"ridge linear regression on {', '.join(lr_features)}",
            "metrics": metrics_b,
            "mae_ci": bootstrap_mae_ci(aa, ap_b, seed=seed),
            "feature_columns": lr_features,
            "fit": _summarise_fit(model_b.metadata()),
        },
        {
            "model": "C",
            "name": model_c.name,
            "kind": "probabilistic",
            "rule": "LightGBM quantile regression (alpha 0.10/0.50/0.90) on "
                    f"{', '.join(lr_features)}",
            # P50 is C's point forecast; its MAE/RMSE/bias are reported
            # alongside A and B for comparability.
            "point_metrics_as_p50": metrics_c_point,
            "metrics": metrics_c_point,  # alias for the markdown table
            "probabilistic_metrics": metrics_c_prob,
            "quantile_crossing": crossing,
            "feature_columns": lr_features,
            "quantiles": [float(q) for q in model_c.quantiles],
            "fit": _summarise_fit(model_c.metadata()),
        },
    ]

    partition = _partition_counts(dataset, split)
    reproducibility = {
        "seed": seed,
        "quantile_model": _summarise_fit(model_c.metadata()),
        "gridpulse_version": _gridpulse_version(),
        "python_version": _python_version(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    calibration = {
        "nominal": {"p10": 0.10, "p50": 0.50, "p90": 0.90, "interval_80_coverage": 0.80},
        "empirical_observed": metrics_c_prob["empirical_coverage"],
        "interval_80_observed": metrics_c_prob["interval_coverage"],
        "note": (
            "empirical coverage on a deterministic synthetic fixture is not "
            "evidence of real-world calibration; it only demonstrates the "
            "measurement machinery. Real ENTSO-E data is required before any "
            "calibration claim can be made."
        ),
    }

    comparison = {
        "direction": "MAE_A - MAE_B",
        "difference": ci_diff["difference"],
        "ci_low": ci_diff["ci_low"],
        "ci_high": ci_diff["ci_high"],
        "ci_level": ci_diff["ci_level"],
        "n_boot": ci_diff["n_boot"],
        "method": "percentile bootstrap on aligned test pairs",
        "note": (
            "positive difference => model A ('seasonal naive') has higher MAE. "
            "An interval excluding 0 indicates a statistically suggestive "
            "difference on this fixture-only evaluation."
        ),
    }

    info = {
        "data_status": "FIXTURE-VERIFIED",
        "target_column": dataset.metadata["target_column"],
        "horizon_hours": dataset.metadata["horizon_hours"],
        "naive_lag_hours": dataset.metadata["naive_lag_hours"],
        "issue_hour_utc": dataset.metadata["issue_hour_utc"],
        "asof_policy": dataset.metadata["asof_policy"],
        "n_forecast_rows": len(dataset),
        "predictor_columns": list(dataset.predictor_columns),
        "n_test_row_candidates": n_test_rows,
        "n_aligned_rows_evaluated": n_aligned,
        "quantile_crossing": crossing,
        "mape_min_abs": mape_min_abs,
        "statement": (
            "UNVERIFIED for live electricity: there is no ENTSO-E API key; "
            "the feature table behind this benchmark was generated from "
            "hand-crafted synthetic fixtures exercising the real pipeline. "
            "Weather sources are real Open-Meteo data in live runs, but this "
            "offline benchmark uses synthetic payloads end-to-end."
        ),
    }

    return ProbabilisticBenchmarkResult(
        models=model_results,
        comparison=comparison,
        split=split.to_dict(),
        train_validation_test_counts=partition,
        info=info,
        calibration=calibration,
        reproducibility=reproducibility,
    )


@dataclass(frozen=True)
class ProbabilisticBenchmarkResult:
    """Structured Phase 4C benchmark output (JSON + Markdown renderable)."""

    models: list
    comparison: Mapping
    split: Mapping
    train_validation_test_counts: Mapping
    info: Mapping
    calibration: Mapping
    reproducibility: Mapping

    def to_dict(self) -> dict:
        return {
            "phase": "4C",
            "data_status": self.info["data_status"],
            "info": self.info,
            "models": self.models,
            "comparison": self.comparison,
            "split": self.split,
            "train_validation_test_counts": self.train_validation_test_counts,
            "calibration": self.calibration,
            "reproducibility": self.reproducibility,
        }

    def to_markdown(self) -> str:
        l = []
        l.append("# Probabilistic Forecasting Benchmark (Phase 4C)")
        l.append("")
        l.append(f"**DATA STATUS = `{self.info['data_status']}`**")
        l.append("")
        l.append(
            f"- Target: `{self.info['target_column']}` · horizon "
            f"{self.info['horizon_hours']} h · issue once daily at "
            f"{self.info['issue_hour_utc']}:00 UTC"
        )
        l.append(
            f"- Aligned test rows evaluated: `{self.info['n_aligned_rows_evaluated']}` "
            f"of `{self.info['n_test_row_candidates']}` "
            f"(train={self.train_validation_test_counts['train']}, "
            f"validation={self.train_validation_test_counts['validation']}, "
            f"test={self.train_validation_test_counts['test']} rows)"
        )
        l.append(f"- As-of policy: {self.info['asof_policy']}")
        l.append(f"- Statement: {self.info['statement']}")
        l.append("")
        l.append("## Point metrics (P50 for model C; aligned test rows)")
        l.append("")
        l.append("| model | MAE (MW) | RMSE (MW) | MAPE (%) | bias (MW) | n = |")
        l.append("|---|---|---|---|---|---|")
        for m in self.models:
            met = m["metrics"]
            label = m["name"] + ("" if m["kind"] == "point" else " (P50)")
            l.append(
                f"| {label} | {_fmt(met['mae'])} | {_fmt(met['rmse'])} | "
                f"{_fmt(met['mape'])} | {_fmt(met['bias'])} | {met['n_valid_pairs']} |"
            )
        l.append("")
        l.append("## Probabilistic metrics (model C)")
        l.append("")
        c = self.models[2]
        prob = c["probabilistic_metrics"]
        l.append("### Pinball loss (per quantile; lower is better)")
        l.append("")
        for q in ("0.10", "0.50", "0.90"):
            l.append(f"- P{q} pinball: `{_fmt(prob['pinball'][q])}`")
        l.append("")
        l.append("### Empirical quantile & interval coverage")
        l.append("")
        l.append("| quantile / interval | nominal | observed |")
        l.append("|---|---|---|")
        l.append(f"| P10 | 0.10 | {_fmt(prob['empirical_coverage']['0.10'])} |")
        l.append(f"| P50 | 0.50 | {_fmt(prob['empirical_coverage']['0.50'])} |")
        l.append(f"| P90 | 0.90 | {_fmt(prob['empirical_coverage']['0.90'])} |")
        l.append(f"| [P10, P90] (80%) | 0.80 | {_fmt(prob['interval_coverage'])} |")
        l.append("")
        l.append("### Interval width (sharpness; P90 - P10 in MW)")
        l.append("")
        w = prob["interval_width"]
        l.append(
            f"- mean `{_fmt(w['mean'])}` · median `{_fmt(w['median'])}` · "
            f"min `{_fmt(w['min'])}` · max `{_fmt(w['max'])}` · "
            f"std `{_fmt(w['std'])}`"
        )
        l.append("")
        l.append("### Risk score (relative uncertainty, `(P90-P10)/|P50|`)")
        l.append("")
        r = prob["risk_score"]
        l.append(
            f"- mean `{_fmt(r['mean'])}` · median `{_fmt(r['median'])}` · "
            f"n P50 near-zero `{r['n_near_zero_p50']}`"
        )
        l.append("")
        l.append("### Quantile crossing")
        l.append("")
        x = c["quantile_crossing"]
        l.append(
            f"- raw triples `{x['n_triples']}`, crossing detected `{x['n_detected']}`, "
            f"corrected `{x['n_corrected']}` (deterministic ascending sort; never hidden)"
        )
        l.append("")
        l.append("## Calibration note")
        l.append("")
        l.append(self.calibration["note"])
        l.append("")
        cmp = self.comparison
        l.append(
            f"## MAE difference A - B = `{_fmt(cmp['difference'])}` "
            f"(95% CI [{_fmt(cmp['ci_low'])}, {_fmt(cmp['ci_high'])}])"
        )
        l.append("")
        l.append(f"- {cmp['note']}")
        l.append("")
        l.append("## Reproducibility")
        l.append("")
        rp = self.reproducibility
        l.append(
            f"- seed `{rp['seed']}`, gridpulse `{rp['gridpulse_version']}`, "
            f"python `{rp['python_version']}`, generated `{rp['generated_at_utc']}` UTC"
        )
        qm = rp.get("quantile_model") or {}
        if qm:
            l.append(
                f"- quantile model: lightgbm `{qm.get('library', '?')}`, "
                f"n_estimators `{qm.get('n_estimators')}`, "
                f"learning_rate `{qm.get('learning_rate')}`, "
                f"num_leaves `{qm.get('num_leaves')}`, "
                f"min_child_samples `{qm.get('min_child_samples')}`, "
                f"random_state `{qm.get('random_state')}`, "
                f"fitted `{qm.get('fitted')}`, "
                f"n_train_rows `{qm.get('n_train_rows')}`"
            )
        l.append("")
        return "\n".join(l)

    def write(self, out_dir, *, stem: str = "forecast_benchmark_phase4c") -> list:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{stem}.json"
        md_path = out_dir / f"{stem}.md"
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        md_path.write_text(self.to_markdown() + "\n", encoding="utf-8", newline="\n")
        return [json_path, md_path]


@dataclass(frozen=True)
class BenchmarkResult:
    models: list
    comparison: Mapping
    split: Mapping
    train_validation_test_counts: Mapping
    info: Mapping
    reproducibility: Mapping

    def to_dict(self) -> dict:
        return {
            "phase": "4B",
            "data_status": self.info["data_status"],
            "info": self.info,
            "models": self.models,
            "comparison": self.comparison,
            "split": self.split,
            "train_validation_test_counts": self.train_validation_test_counts,
            "reproducibility": self.reproducibility,
        }

    def to_markdown(self) -> str:
        l = []
        l.append("# Forecasting Baseline Comparison (Phase 4B)")
        l.append("")
        l.append(f"**DATA STATUS = `{self.info['data_status']}`**")
        l.append("")
        l.append(
            f"- Target: `{self.info['target_column']}` · horizon "
            f"{self.info['horizon_hours']} h · issue once daily at "
            f"{self.info['issue_hour_utc']}:00 UTC"
        )
        l.append(
            f"- Aligned test pairs evaluated: "
            f"`{self.info['n_aligned_rows_evaluated']}` "
            f"(train={self.train_validation_test_counts['train']}, "
            f"validation={self.train_validation_test_counts['validation']}, "
            f"test={self.train_validation_test_counts['test']} rows)"
        )
        l.append(f"- As-of policy: {self.info['asof_policy']}")
        l.append(f"- Statement: {self.info['statement']}")
        l.append("")
        l.append("## Per-model metrics (test window, aligned rows)")
        l.append("")
        l.append("| model | MAE (MW) | RMSE (MW) | MAPE (%) | bias (MW) | median abs err (MW) | n = |")
        l.append("|---|---|---|---|---|---|---|")
        for m in self.models:
            met = m["metrics"]
            l.append(
                f"| {m['name']} | {_fmt(met['mae'])} | {_fmt(met['rmse'])} | "
                f"{_fmt(met['mape'])} | {_fmt(met['bias'])} | "
                f"{_fmt(met['median_absolute_error'])} | {met['n_valid_pairs']} |"
            )
        l.append("")
        l.append("## Bootstrap MAE CIs (percentile, fixed seed)")
        l.append("")
        l.append("| model | MAE (MW) | 95% CI |")
        l.append("|---|---|---|")
        for m in self.models:
            ci = m["mae_ci"]
            l.append(
                f"| {m['name']} | {_fmt(ci['mae'])} | "
                f"[{_fmt(ci['ci_low'])}, {_fmt(ci['ci_high'])}] |"
            )
        l.append("")
        c = self.comparison
        l.append(
            f"## MAE difference A - B = `{_fmt(c['difference'])}` "
            f"(95% CI [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}])"
        )
        l.append("")
        l.append(f"- {c['note']}")
        l.append("")
        l.append("## Reproducibility")
        l.append("")
        r = self.reproducibility
        l.append(
            f"- seed `{r['seed']}`, gridpulse `{r['gridpulse_version']}`, "
            f"python `{r['python_version']}`, generated "
            f"`{r['generated_at_utc']}` UTC"
        )
        l.append("")
        return "\n".join(l)

    def write(self, out_dir, *, stem: str = "forecast_benchmark_phase4b") -> list:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{stem}.json"
        md_path = out_dir / f"{stem}.md"
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        md_path.write_text(self.to_markdown() + "\n", encoding="utf-8", newline="\n")
        return [json_path, md_path]


def _align(actuals, pred_a, pred_b):
    aligned = [
        (a, pa, pb)
        for a, pa, pb in zip(actuals, pred_a, pred_b)
        if a is not None and pa is not None and pb is not None
    ]
    return (
        [t[0] for t in aligned],
        [t[1] for t in aligned],
        [t[2] for t in aligned],
    )


def _partition_counts(dataset, split):
    counts = {"train": 0, "validation": 0, "test": 0}
    for r in dataset.rows:
        if r.issue_time < split.train_end:
            counts["train"] += 1
        elif r.issue_time < split.validation_end:
            counts["validation"] += 1
        else:
            counts["test"] += 1
    return counts


def _summarise_fit(metadata: Mapping) -> dict:
    keep = [
        "fitted",
        "n_train_rows",
        "n_rows_dropped_missing_features",
        "ridge",
        "fit_intercept",
        "feature_columns",
        # Quantile (LightGBM) model hyperparameters for reproducibility.
        "library",
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "random_state",
        "boost_from_average",
    ]
    return {k: metadata[k] for k in keep if k in metadata}


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _gridpulse_version() -> str:
    try:
        from .. import __version__  # type: ignore[attr-defined]

        return str(__version__)
    except Exception:
        return "unknown"


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


__all__ = [
    "run_benchmark",
    "BenchmarkResult",
    "run_probabilistic_benchmark",
    "ProbabilisticBenchmarkResult",
]