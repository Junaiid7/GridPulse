"""Phase 4C fixture-backed probabilistic forecasting benchmark (offline).

Drives the REAL GridPulse pipeline on **hand-crafted synthetic fixtures**
(electricity + weather) in a scratch directory, then runs the probabilistic
benchmark — A = seasonal naive 24h, B = ridge linear regression, C = LightGBM
quantile regression (P10 / P50 / P90) — and writes the results to
``data/reports/forecast_benchmark_phase4c.{json,md}``.

IMPORTANT: everything electricity here is FIXTURE-VERIFIED. No real ENTSO-E
data is used or claimed. When an ``ENTSOE_API_KEY`` is configured, the same
``run_pipeline`` + ``run_probabilistic_benchmark`` path consumes the real
feature table and the ``data_status`` flag will reflect a live backend.

Run (from repo root):
    .venv/Scripts/python.exe scripts/phase4c_fixture_benchmark.py
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

from gridpulse.forecast.benchmark import run_probabilistic_benchmark  # noqa: E402

SEED = 0
REPORT_DIR = REPO_ROOT / "data" / "reports"


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="gp-phase4c-fixture-"))
    print("fixture pipeline data (temporary):", scratch)

    run = run_synthetic_pipeline(scratch)
    rows = features_rows(run)

    result = run_probabilistic_benchmark(rows, seed=SEED)
    paths = result.write(REPORT_DIR, stem="forecast_benchmark_phase4c")

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
        met = m["metrics"] if m["kind"] == "point" else m["point_metrics_as_p50"]
        mape = "-" if met["mape"] is None else f"{met['mape']:.2f}"
        print(f"{m['name']:<28}{met['mae']:>10.2f}{met['rmse']:>10.2f}"
              f"{mape:>10}{met['bias']:>10.2f}{met['n_valid_pairs']:>5}")
    c = result.models[2]["probabilistic_metrics"]
    x = info["quantile_crossing"]
    print("-" * 64)
    for q in ("0.10", "0.50", "0.90"):
        print(f"C pinball @ q={q:<4}: {c['pinball'][q]:>8.2f}   "
              f"empirical coverage: {c['empirical_coverage'][q]:.2f}")
    w = c["interval_width"]
    print(f"C interval [P10,P90] mean width: {w['mean']:.2f} MW "
          f"(median {w['median']:.2f}, n={w['n']})")
    rs = c["risk_score"]
    print(f"C risk score (P90-P10)/|P50|    : mean {rs['mean']:.3f} "
          f"(near-zero P50: {rs['n_near_zero_p50']})")
    print(f"C quantile crossings            : {x['n_detected']} detected / "
          f"{x['n_corrected']} corrected of {x['n_triples']} triples")
    print("-" * 64)
    comp = result.comparison
    print(f"MAE_A - MAE_B = {comp['difference']:.2f} MW "
          f"(95% CI [{comp['ci_low']:.2f}, {comp['ci_high']:.2f}])")
    print(f"NOTE: {comp['note']}")
    print(f"CALIBRATION note: {result.calibration['note']}")
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