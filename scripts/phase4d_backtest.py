"""Phase 4D-B fixture-backed dispatch backtest (offline).

Drives the REAL GridPulse pipeline on **hand-crafted synthetic fixtures**
(electricity) in a scratch directory, then runs the Phase 4D-B backtest:
24 per-offset P10/P50/P90 forecast models issued once daily, dispatches the
four Phase 4D-A battery strategies over the held-out test window, and settles
every schedule against *realised* gold residual + price. Writes the results
to ``data/reports/dispatch_backtest_phase4db.{json,md}``.

IMPORTANT: everything electricity here is FIXTURE-VERIFIED. The held-out
window is ~9 days and is *machinery validation* — it proves the measurement
harness, not real-world strategy superiority. No ENTSO-E API key is used or
claimed; ``data_status`` stays ``"FIXTURE-VERIFIED"`` until a real
feature/gold table is consumed.

Run (from repo root):
    .venv/Scripts/python.exe scripts/phase4d_backtest.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gridpulse.optimization import run_dispatch_backtest  # noqa: E402
from support.backtest_fixture import gold_hour_rows  # noqa: E402
from support.synthetic_electricity import features_rows, run_synthetic_pipeline  # noqa: E402

SEED = 0
N_BOOT = 2000
REPORT_DIR = REPO_ROOT / "data" / "reports"


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="gp-phase4db-fixture-"))
    print("fixture pipeline data (temporary):", scratch)

    run = run_synthetic_pipeline(scratch)
    rows = features_rows(run)
    gold = gold_hour_rows(run)

    result = run_dispatch_backtest(
        rows,
        gold,
        seed=SEED,
        n_boot=N_BOOT,
    )
    paths = result.write(REPORT_DIR, stem="dispatch_backtest_phase4db")

    info = result.info
    print("=" * 64)
    print(f"DATA STATUS = {info['data_status']}")
    print(f"target         : {info['target_column']} | horizon {info['horizon_hours']}h | "
          f"issue {info['issue_hour_utc']}:00 UTC once daily")
    print(f"profiles       : {info['n_profiles']} issue days "
          f"({info['n_profiles_complete']} complete; {info['n_dropped_days']} dropped)")
    print(f"dispatch days  : {info['n_dispatch_days']}")
    print(f"battery        : {info['battery']['capacity_mwh']} MWh / "
          f"{info['battery']['max_charge_power_mw']} MW, "
          f"round-trip {info['battery']['round_trip_efficiency']}")
    print(f"strategies     : {', '.join(info['strategies'])}")
    fs = info["forecast_summary"]
    fsp = fs.get("probabilistic_metrics") or {}
    if fsp:
        print("-" * 64)
        print(f"forecast       : {fs['n_aligned_pairs']} aligned pairs across 24 offsets")
        pb = fsp["pinball"]
        print(f"  pinball      : P10 {pb['0.10']:.2f} | P50 {pb['0.50']:.2f} | P90 {pb['0.90']:.2f} MW")
        ec = fsp["empirical_coverage"]
        print(f"  emp. coverage: P10 {ec['0.10']:.2f} | P50 {ec['0.50']:.2f} | P90 {ec['0.90']:.2f}")
        print(f"  80% interval : coverage {fsp['interval_coverage']:.2f}, "
              f"mean width {fsp['interval_width']['mean']:.2f} MW")
    print("-" * 64)
    print(f"{'strategy':<18}{'n':>5}{'mean cost':>14}{'median':>14}"
          f"{'savings vs nb':>14}{'exp-uncut':>10}{'neutral dMWh':>13}")
    for s in result.strategies:
        sav = s["mean_daily_savings_vs_no_battery_eur"]
        sav_s = "-" if sav is None else f"{sav:,.2f}"
        neu = s["mean_energy_neutrality_delta_mwh"]
        neu_s = "-" if neu is None else f"{neu:.3f}"
        print(f"{s['strategy']:<18}{s['n_days']:>5}{s['mean_daily_cost_eur']:>14,.2f}"
              f"{s['median_daily_cost_eur']:>14,.2f}{sav_s:>14}"
              f"{s['total_export_undercut_hours']:>10}{neu_s:>13}")
    print("-" * 64)
    for c in result.comparisons:
        print(f"{c['A']} vs {c['B']}: mean delta = {c['difference']:,.2f} EUR "
              f"(95% CI [{c['ci_low']:,.2f}, {c['ci_high']:,.2f}], n={c['n_days']})")
    print(f"NOTE: {result.comparisons[0]['note']}")
    print("-" * 64)
    print("wrote:", paths[0])
    print("wrote:", paths[1])
    print()
    print("ELECTRICITY: synthetic fixture only (FIXTURE-VERIFIED).")
    print("BACKTEST:    machinery validation on a ~9-day held-out synthetic window;")
    print("             no claim about real Dutch electricity or strategy superiority.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())