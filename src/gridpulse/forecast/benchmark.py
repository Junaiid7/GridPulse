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
)
from .models import LinearRegressionModel, SeasonalNaiveModel
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


__all__ = ["run_benchmark", "BenchmarkResult"]