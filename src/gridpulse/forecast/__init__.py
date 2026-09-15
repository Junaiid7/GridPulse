"""Leakage-safe forecasting for GridPulse (Phase 4B point / Phase 4C probabilistic).

The forecast layer consumes the feature dataset produced by the Phase 4A
pipeline (``data/gold/nl/features.csv``) and turns it into a forecasting-ready
dataset with an explicit, documented information-cutoff convention:

- one forecast issued once daily at a fixed UTC hour;
- target ``T = issue_time + 24h``;
- every predictor comes from the feature row anchored at the **issue time**
  (all values computed strictly-before that instant by the upstream
  leakage-safe feature builders);
- the label is the realised residual load of the target hour.

Phase 4B added pure-standard-library point baselines (seasonal naive, ridge
linear regression) with MAE/RMSE/MAPE/bias/median-absolute-error evaluation.
Phase 4C adds **probabilistic** 24h-ahead forecasts: a LightGBM quantile
regression producing P10/P50/P90, quantile-crossing detection/correction,
pinball-loss / coverage / interval-width / sharpness / calibration diagnostics,
a relative risk score ``(P90-P10)/|P50|``, and an A/B/C benchmark.

All benchmarks are offline / fixture-driven. Open-Meteo is **live-verified**;
ENTSO-E electricity is **currently unavailable**; forecasts produced from
fixture data are labelled ``FIXTURE-VERIFIED`` and must never be presented as
live-validated results.
"""

from .contract import ForecastingDataset, ForecastRow, build_forecasting_dataset
from .split import ChronologicalSplit, chronological_split_by_fraction, partition_timestamps
from .evaluate import (
    compute_point_metrics,
    bootstrap_mae_ci,
    bootstrap_mae_difference_ci,
    pinball_loss,
    empirical_coverage,
    interval_coverage,
    interval_width_stats,
    probabilistic_metrics,
)
from .probabilistic import (
    QUANTILES,
    QuantileForecast,
    QuantileForecasts,
    correct_quantiles,
    detect_quantile_crossing,
    risk_score,
)
from .benchmark import (
    run_benchmark,
    BenchmarkResult,
    run_probabilistic_benchmark,
    ProbabilisticBenchmarkResult,
)

__all__ = [
    "ForecastingDataset",
    "ForecastRow",
    "build_forecasting_dataset",
    "ChronologicalSplit",
    "chronological_split_by_fraction",
    "partition_timestamps",
    "compute_point_metrics",
    "bootstrap_mae_ci",
    "bootstrap_mae_difference_ci",
    "pinball_loss",
    "empirical_coverage",
    "interval_coverage",
    "interval_width_stats",
    "probabilistic_metrics",
    "QUANTILES",
    "QuantileForecast",
    "QuantileForecasts",
    "correct_quantiles",
    "detect_quantile_crossing",
    "risk_score",
    "run_benchmark",
    "BenchmarkResult",
    "run_probabilistic_benchmark",
    "ProbabilisticBenchmarkResult",
]