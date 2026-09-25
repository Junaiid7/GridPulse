"""CLI entry point for GridPulse orchestration.

Run with:
    python -m gridpulse.orchestrate

Or install as a console script (see pyproject.toml).
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

from gridpulse.orchestration.runner import run_e2e_orchestration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GridPulse E2E orchestration: pipeline + forecast + backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full E2E on synthetic fixtures (no API key needed)
    python -m gridpulse.orchestrate

    # Run with custom date range
    python -m gridpulse.orchestrate --start 2024-02-01 --end 2024-03-15

    # Skip forecast, only run pipeline and backtest
    python -m gridpulse.orchestrate --skip-forecast

    # Skip backtest, only run pipeline and forecast
    python -m gridpulse.orchestrate --skip-backtest

    # Use live ENTSO-E data (requires ENTSOE_API_KEY in the environment)
    python -m gridpulse.orchestrate --start 2024-02-01 --end 2024-03-15

    # Customize bootstrap samples and seed
    python -m gridpulse.orchestrate --n-boot 5000 --seed 42
        """,
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start datetime (ISO format, e.g., 2024-02-01)",
        default=None,
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End datetime (ISO format, e.g., 2024-03-15)",
        default=None,
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip the data pipeline step (use synthetic fixtures)",
    )
    parser.add_argument(
        "--skip-forecast",
        action="store_true",
        help="Skip the probabilistic forecasting step",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        help="Skip the dispatch backtesting step",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=2000,
        help="Number of bootstrap samples for backtest (default: 2000)",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Training fraction for forecast split (default: 0.7)",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
        help="Validation fraction for forecast split (default: 0.15)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (optional)",
    )
    return parser.parse_args()


def parse_datetime(s: str) -> datetime:
    """Parse datetime string in ISO format."""
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            # Try date-only format
            return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)
        except ValueError:
            raise ValueError(
                f"Invalid datetime format: {s}. Use ISO format (e.g., 2024-02-01)"
            )


def main() -> int:
    args = parse_args()

    # Parse datetime arguments
    start = parse_datetime(args.start) if args.start else None
    end = parse_datetime(args.end) if args.end else None

    print("=" * 64)
    print("GridPulse E2E Orchestration (Phase 5)")
    print("=" * 64)

    result = run_e2e_orchestration(
        start=start,
        end=end,
        skip_pipeline=args.skip_pipeline,
        skip_forecast=args.skip_forecast,
        skip_backtest=args.skip_backtest,
        seed=args.seed,
        n_boot=args.n_boot,
        forecast_split_fraction=args.train_fraction,
        forecast_validation_fraction=args.validation_fraction,
    )

    print(f"DATA STATUS = {result.data_status}")
    print()

    # Print component statuses
    info = result.info
    print(
        f"Pipeline:  {info.get('pipeline_status', info.get('skip_pipeline', 'unknown'))}"
    )
    print(
        f"Forecast:  {info.get('forecast_status', info.get('forecast_skipped', 'unknown'))}"
    )
    print(
        f"Backtest:  {info.get('backtest_status', info.get('backtest_skipped', 'unknown'))}"
    )

    if result.info.get("errors"):
        print()
        print("ERRORS:")
        for err in result.info["errors"]:
            print(f"  - {err}")

    print()
    print("-" * 64)
    print(f"Duration: {result.reproducibility.get('duration_seconds', 'N/A'):.1f}s")
    print("=" * 64)

    # Write output if requested
    if args.output:
        import json

        output_path = args.output
        with open(output_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        print(f"Wrote: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
