"""Phase 5 end-to-end orchestration execution script.

Runs the full GridPulse E2E orchestration (pipeline + probabilistic forecast +
dispatch backtest) on deterministic synthetic fixtures and writes reports.

Run (from repo root):
    .venv/Scripts/python.exe scripts/phase5_orchestration.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT), str(REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gridpulse.orchestration.runner import run_e2e_orchestration  # noqa: E402

REPORT_DIR = REPO_ROOT / "data" / "reports"


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="gp-phase5-orchestration-"))
    print("orchestration scratch dir:", scratch)

    print("Running Phase 5 E2E orchestration...")
    result = run_e2e_orchestration(
        skip_pipeline=False,
        scratch_dir=scratch,
        n_boot=2000,
        seed=0,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "orchestration_phase5.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, default=str)

    print("=" * 64)
    print(f"DATA STATUS = {result.data_status}")
    print(f"Pipeline: {result.info.get('pipeline_status')}")
    print(f"Forecast: {result.info.get('forecast_status')}")
    print(f"Backtest: {result.info.get('backtest_status')}")
    print(f"Duration: {result.reproducibility.get('duration_seconds', 'N/A'):.1f}s")
    print("-" * 64)
    print("wrote report:", json_path)
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
