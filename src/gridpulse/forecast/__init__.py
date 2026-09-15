"""Leakage-safe forecasting for GridPulse (Phase 4B).

The forecast layer consumes the feature dataset produced by the Phase 4A
pipeline (``data/gold/nl/features.csv``) and turns it into a forecasting-ready
dataset with an explicit, documented information-cutoff convention:

- one forecast issued once daily at a fixed UTC hour;
- target ``T = issue_time + 24h``;
- every predictor comes from the feature row anchored at the **issue time**
  (all values computed strictly-before that instant by the upstream
  leakage-safe feature builders);
- the label is the realised residual load of the target hour.

Everything here is pure standard library (no numpy/pandas/ML frameworks) and
all benchmarks are offline / fixture-driven.

Open-Meteo is **live-verified**; ENTSO-E electricity is **currently
unavailable**; forecasts produced from fixture data are labelled
``FIXTURE-VERIFIED`` and must never be presented as live-validated results.
"""

from .contract import ForecastingDataset, ForecastRow, build_forecasting_dataset
from .split import ChronologicalSplit, chronological_split_by_fraction, partition_timestamps
from .evaluate import compute_point_metrics, bootstrap_mae_ci, bootstrap_mae_difference_ci
from .benchmark import run_benchmark, BenchmarkResult

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
    "run_benchmark",
    "BenchmarkResult",
]