"""CLI entry point for the GridPulse pipeline runner (Phase 4A).

Usage::

    python -m gridpulse.pipeline --start 2026-09-04 --end 2026-09-14

    python -m gridpulse.pipeline \\
        --start 2026-09-04 --end 2026-09-14 \\
        --require-entsoe \\
        --report-dir /path/to/reports

Exit codes:
  0  — run completed, report produced.
  1  — hard error (unexpected exception, nothing useful written).
  2  — ``--require-entsoe`` was set but ENTSO-E data is unavailable.
  3  — ``--require-gold`` was set but Gold was not produced.

No secrets are ever committed. Datasets remain Git-ignored.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path


def _parse_date(raw: str) -> datetime:
    """Parse a YYYY-MM-DD string as a UTC-aware midnight datetime."""
    return datetime.combine(date.fromisoformat(raw), datetime.min.time()).replace(tzinfo=timezone.utc)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gridpulse-pipeline",
        description="Run the GridPulse real-data pipeline (Phase 4A).",
    )
    parser.add_argument(
        "--start",
        type=_parse_date,
        default=None,
        help="Start of the request window (YYYY-MM-DD). Defaults to 8 days before --end.",
    )
    parser.add_argument(
        "--end",
        type=_parse_date,
        default=None,
        help="End of the request window exclusive (YYYY-MM-DD). Defaults to yesterday.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the data root (default: $GRIDPULSE_DATA_DIR or <project>/data).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Override report output directory (default: <data-dir>/reports).",
    )
    parser.add_argument(
        "--require-entsoe",
        action="store_true",
        default=False,
        help="Exit 2 if ENTSO-E datasets are unavailable.",
    )
    parser.add_argument(
        "--require-gold",
        action="store_true",
        default=False,
        help="Exit 3 if Gold dataset was not produced.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns exit code."""
    args = build_parser().parse_args(argv)

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Resolve dates
    today = datetime.combine(date.today(), datetime.min.time()).replace(tzinfo=timezone.utc)
    if args.end is None:
        end = today - timedelta(days=1)  # yesterday
    else:
        end = args.end
    if args.start is None:
        start = end - timedelta(days=8)
    else:
        start = args.start

    if end <= start:
        print("ERROR: --end must be after --start", file=sys.stderr)
        return 1

    # Optional data-dir override via env
    if args.data_dir is not None:
        import os
        os.environ["GRIDPULSE_DATA_DIR"] = str(args.data_dir.resolve())

    try:
        from .runner import run_pipeline
        run = run_pipeline(
            start=start,
            end=end,
            report_dir=args.report_dir,
        )
    except Exception as exc:
        logging.getLogger("gridpulse.pipeline").critical("Pipeline run failed: %r", exc, exc_info=True)
        return 1

    # Print summary (ASCII only — the Windows cp1252 console cannot encode
    # Unicode arrows/em-dashes reliably; the JSON/Markdown report keeps UTF-8).
    print("\n=== GridPulse Pipeline Run Complete ===")
    print(f"Window:      {start.isoformat()} -> {end.isoformat()}")
    print(f"ENTSO-E:     {'available' if run.entsoe_available else 'NOT available (no API key)'}")
    print(f"Imbalance:   {run.imbalance_note.splitlines()[0][:120] if run.imbalance_note else 'unknown'}")
    print(f"Gold:        {'PRODUCED' if run.gold.produced else 'NOT PRODUCED - ' + run.gold.reason}")
    print(f"Features:    {'PRODUCED' if run.features.produced else 'NOT PRODUCED - ' + run.features.reason}")
    if run.report_paths:
        for p in run.report_paths:
            print(f"Report:      {p}")
    print()

    # Exit codes
    if run.errors:
        print("ERRORS:", file=sys.stderr)
        for err in run.errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    if args.require_entsoe and not run.entsoe_available:
        print("ENTSO-E required but unavailable (exit 2)", file=sys.stderr)
        return 2

    if args.require_gold and not run.gold.produced:
        print("Gold required but not produced (exit 3)", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    from datetime import timedelta
    raise SystemExit(main())
