"""Phase 4B fixture-backed forecasting benchmark (offline, no API keys).

Drives the REAL GridPulse pipeline on **hand-crafted synthetic fixtures**
(electricity + weather) in a scratch directory, then runs the baseline
comparison (A = seasonal naive 24h vs B = ridge linear regression) and writes
the results to ``data/reports/forecast_benchmark_phase4b.{json,md}``.

IMPORTANT: everything electricity here is FIXTURE-VERIFIED. No real ENTSO-E
data is used or claimed. When an ``ENTSOE_API_KEY`` is configured, the same
``run_pipeline`` + ``run_benchmark`` path consumes the real feature table and
the ``data_status`` flag will reflect a live backend.

Run (from repo root):
    .venv/Scripts/python.exe scripts/phase4b_fixture_benchmark.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from support.synthetic_electricity import (  # noqa: E402
    features_rows,
    run_synthetic_pipeline,
)

from gridpulse.forecast.benchmark import run_benchmark  # noqa: E402

SEED = 0
REPORT_DIR = REPO_ROOT / "data" / "reports"


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="gp-phase4b-fixture-"))
    print("fixture pipeline data (temporary):", scratch)

    run = run_synthetic_pipeline(scratch)
    rows = features_rows(run)

    result = run_benchmark(rows, seed=SEED)
    paths = result.write(REPORT_DIR, stem="forecast_benchmark_phase4b")

    info = result.info
    counts = result.train_validation_test_counts
    print("-" * 64)
    print(f"DATA STATUS = {info['data_status']}")
    print(f"target            : {info['target_column']} | horizon {info['horizon_hours']}h | "
          f"issue {info['issue_hour_utc']}:00 UTC")
    print(f"forecast rows     : {info['n_forecast_rows']} "
          f"(train={counts['train']}, validation={counts['validation']}, "
          f"test={counts['test']})")
    print(f"test rows evaluated: {info['n_aligned_rows_evaluated']} of "
          f"{info['n_test_row_candidates']}")
    print("-" * 64)
    print(f"{'model':<28}{'MAE':>10}{'RMSE':>10}{'MAPE%':>10}{'bias':>10}{'n':>5}")
    for m in result.models:
        met = m["metrics"]
        mape = "-" if met["mape"] is None else f"{met['mape']:.2f}"
        print(f"{m['name']:<28}{met['mae']:>10.2f}{met['rmse']:>10.2f}"
              f"{mape:>10}{met['bias']:>10.2f}{met['n_valid_pairs']:>5}")
    c = result.comparison
    print("-" * 64)
    print(f"MAE_A - MAE_B = {c['difference']:.2f} MW "
          f"(95% CI [{c['ci_low']:.2f}, {c['ci_high']:.2f}])")
    print(f"NOTE: {c['note']}")
    print("-" * 64)
    print("wrote:", paths[0])
    print("wrote:", paths[1])
    print()
    print("ELECTRICITY: synthetic fixture only (FIXTURE-VERIFIED).")
    print("WEATHER:    synthetic fixture only in this offline demo.")
    print("FORECAST:   fixture-verified only; not validated against live data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())