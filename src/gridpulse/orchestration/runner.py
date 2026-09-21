"""End-to-End orchestration for GridPulse.

Phase 5 ties together the data pipeline (Phase 4A), probabilistic forecasting
(Phase 4C), and dispatch backtesting (Phase 4D-B) into a single entry point.
The orchestration propagates data status from the pipeline to the downstream
components.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from gridpulse.config import Settings, get_settings
from gridpulse.forecast.benchmark import run_probabilistic_benchmark
from gridpulse.optimization.backtest import run_dispatch_backtest
from gridpulse.pipeline.runner import PipelineRun, run_pipeline
from gridpulse.pipeline.synthetic import (
    DEFAULT_END,
    DEFAULT_START,
    SyntheticEntsoeClient,
    SyntheticImbalanceSource,
    SyntheticOpenMeteoClient,
    build_synthetic_window,
)
from gridpulse.transformation.csvio import read_table

# Valid data status strings
VALID_DATA_STATUSES = frozenset(["FIXTURE-VERIFIED", "verified_live", "UNVERIFIED"])


@dataclass(frozen=True)
class OrchestrationResult:
    """Result of a full E2E orchestration run.

    Attributes
    ----------
    phase : str
        Always ``"5"`` for Phase 5.
    data_status : str
        The propagated data status: ``FIXTURE-VERIFIED``, ``verified_live``, or
        ``UNVERIFIED``.
    pipeline : PipelineRun or None
        The pipeline run result. ``None`` if pipeline was skipped.
    forecast : dict or None
        The probabilistic benchmark result as a dict. ``None`` if forecasting
        was skipped.
    backtest : dict or None
        The dispatch backtest result as a dict. ``None`` if backtesting was
        skipped.
    info : dict
        Metadata about the orchestration run, including timestamps, inputs, and
        any errors encountered.
    reproducibility : dict
        Reproducibility metadata (seed, versions, etc.).
    """

    phase: str = "5"
    data_status: str = "FIXTURE-VERIFIED"
    pipeline: PipelineRun | None = None
    forecast: dict | None = None
    backtest: dict | None = None
    info: dict = field(default_factory=dict)
    reproducibility: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "phase": self.phase,
            "data_status": self.data_status,
            "pipeline": _pipeline_to_dict(self.pipeline),
            "forecast": self.forecast,
            "backtest": self.backtest,
            "info": self.info,
            "reproducibility": self.reproducibility,
        }


def _pipeline_to_dict(pipeline: PipelineRun | None) -> dict | None:
    """Convert PipelineRun to a JSON-serializable dict."""
    if pipeline is None:
        return None
    return {
        "settings": {
            "entsoe_api_key_set": pipeline.settings.entsoe_api_key is not None,
        },
        "requested_start": pipeline.requested_start.isoformat(),
        "requested_end": pipeline.requested_end.isoformat(),
        "started_at": pipeline.started_at.isoformat() if pipeline.started_at else None,
        "finished_at": pipeline.finished_at.isoformat()
        if pipeline.finished_at
        else None,
        "entsoe_available": pipeline.entsoe_available,
        "ok": pipeline.ok,
        "errors": pipeline.errors,
    }


def run_e2e_orchestration(
    *,
    # Pipeline options
    start: datetime | None = None,
    end: datetime | None = None,
    data_status: str | None = None,
    settings: Settings | None = None,
    skip_pipeline: bool = False,
    # Forecast options
    skip_forecast: bool = False,
    forecast_split_fraction: float = 0.7,
    forecast_validation_fraction: float = 0.15,
    # Backtest options
    skip_backtest: bool = False,
    n_boot: int = 2000,
    # Common
    seed: int = 0,
    scratch_dir: Path | None = None,
) -> OrchestrationResult:
    """Run the full GridPulse end-to-end orchestration.

    This function coordinates the data pipeline, probabilistic forecasting, and
    dispatch backtesting into a single workflow. It propagates the data status
    from the pipeline to the downstream components.

    Parameters
    ----------
    start : datetime, optional
        Start datetime for the pipeline. Defaults to DEFAULT_START (2024-01-15)
        for synthetic fixtures.
    end : datetime, optional
        End datetime for the pipeline. Defaults to DEFAULT_END (2024-03-15)
        for synthetic fixtures.
    data_status : str, optional
        Explicit data status: "FIXTURE-VERIFIED", "verified_live", or
        "UNVERIFIED". If None, inferred from pipeline result or fixture use.
    settings : Settings, optional
        Pipeline settings (includes API keys, data directories). If None,
        uses get_settings() or builds synthetic settings.
    skip_pipeline : bool, optional
        If True, skip the data pipeline and use synthetic fixtures for
        forecasting and backtesting. Default is False.
    skip_forecast : bool, optional
        If True, skip the probabilistic forecasting step. Default is False.
    forecast_split_fraction : float, optional
        Fraction of data for training in the forecast (default 0.7).
    forecast_validation_fraction : float, optional
        Fraction of data for validation in the forecast (default 0.15).
    skip_backtest : bool, optional
        If True, skip the dispatch backtesting step. Default is False.
    n_boot : int, optional
        Number of bootstrap samples for backtest comparisons (default 2000).
    seed : int, optional
        Random seed for reproducibility (default 0).
    scratch_dir : Path, optional
        Directory for temporary pipeline outputs. If not provided, uses a
        system temp directory.

    Returns
    -------
    OrchestrationResult
        The result of the orchestration, including all component results and
        metadata.
    """
    # Validate explicit data_status if provided
    if data_status is not None and data_status not in VALID_DATA_STATUSES:
        raise ValueError(
            f"Invalid data_status={data_status!r}. "
            f"Must be one of {sorted(VALID_DATA_STATUSES)}"
        )

    # Initialize state
    pipeline: PipelineRun | None = None
    run_for_data: PipelineRun | None = None
    inferred_status = data_status or "FIXTURE-VERIFIED"
    errors: list[str] = []
    info: dict = {}

    # Record start time
    started_at = datetime.now(UTC)

    # Determine if we're using synthetic fixtures
    use_synthetic = skip_pipeline or (
        settings is None and data_status in (None, "FIXTURE-VERIFIED")
    )

    # Set defaults for synthetic run
    if use_synthetic:
        if start is None:
            start = DEFAULT_START
        if end is None:
            end = DEFAULT_END
        if scratch_dir is None:
            scratch_dir = Path(tempfile.mkdtemp(prefix="gp-orchestration-"))

    # --- Run pipeline (or skip to fixtures) ---------------------------------
    if skip_pipeline:
        # Pipeline step is skipped; generate synthetic fixtures for downstream steps
        inferred_status = data_status or "FIXTURE-VERIFIED"
        info = {
            "skip_pipeline": True,
            "pipeline_skipped": "Using synthetic fixtures for downstream components",
            "start": start.isoformat(),
            "end": end.isoformat(),
        }
        if not (skip_forecast and skip_backtest):
            try:
                fixtures = build_synthetic_window(start, end)
                run_for_data = run_pipeline(
                    start=fixtures.start,
                    end=fixtures.end,
                    settings=Settings(project_root=scratch_dir, data_root=scratch_dir),
                    open_meteo_client=SyntheticOpenMeteoClient(fixtures),
                    entsoe_client=SyntheticEntsoeClient(fixtures),
                    imbalance_source=SyntheticImbalanceSource(),
                    report_dir=scratch_dir / "reports",
                )
            except Exception as e:
                errors.append(f"Synthetic fixture generation error: {e}")
                inferred_status = "UNVERIFIED"
    elif use_synthetic:
        # Run synthetic pipeline
        try:
            fixtures = build_synthetic_window(start, end)
            pipeline = run_pipeline(
                start=fixtures.start,
                end=fixtures.end,
                settings=Settings(project_root=scratch_dir, data_root=scratch_dir),
                open_meteo_client=SyntheticOpenMeteoClient(fixtures),
                entsoe_client=SyntheticEntsoeClient(fixtures),
                imbalance_source=SyntheticImbalanceSource(),
                report_dir=scratch_dir / "reports",
            )
            run_for_data = pipeline
            inferred_status = data_status or "FIXTURE-VERIFIED"
            info = {
                "pipeline_status": "ok" if pipeline.ok else "errors",
                "entsoe_available": False,
                "synthetic": True,
                "pipeline_errors": pipeline.errors,
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
        except Exception as e:
            errors.append(f"Synthetic pipeline error: {e}")
            inferred_status = "UNVERIFIED"
            info = {
                "pipeline_status": "failed",
                "pipeline_error": str(e),
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
    else:
        # Run real pipeline with provided settings
        try:
            if settings is None:
                settings = get_settings()

            pipeline = run_pipeline(
                start=start,
                end=end,
                settings=settings,
                report_dir=scratch_dir,
            )
            run_for_data = pipeline

            # Infer data status from pipeline result if not explicit
            if data_status is None:
                if pipeline.entsoe_available:
                    inferred_status = "verified_live"
                elif pipeline.errors:
                    inferred_status = "UNVERIFIED"
                else:
                    inferred_status = "FIXTURE-VERIFIED"
            else:
                inferred_status = data_status

            info = {
                "pipeline_status": "ok" if pipeline.ok else "errors",
                "entsoe_available": pipeline.entsoe_available,
                "pipeline_errors": pipeline.errors,
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
        except Exception as e:
            errors.append(f"Pipeline error: {e}")
            inferred_status = data_status or "UNVERIFIED"
            info = {
                "pipeline_status": "failed",
                "pipeline_error": str(e),
                "start": start.isoformat() if start else "not set",
                "end": end.isoformat() if end else "not set",
            }

    # --- Run probabilistic forecasting ---------------------------------------
    forecast_result: dict | None = None

    if skip_forecast:
        info["forecast_skipped"] = True
    else:
        try:
            # Load feature rows from pipeline
            if run_for_data is None:
                raise ValueError("Cannot run forecast without pipeline data")

            rows = list(read_table(Path(run_for_data.features.csv_path)))

            # Run the benchmark with the propagated data status
            benchmark = run_probabilistic_benchmark(
                rows,
                train_fraction=forecast_split_fraction,
                validation_fraction=forecast_validation_fraction,
                seed=seed,
            )
            forecast_result = benchmark.to_dict()
            info["forecast_status"] = "ok"
        except Exception as e:
            errors.append(f"Forecast error: {e}")
            info["forecast_status"] = "failed"
            info["forecast_error"] = str(e)

    # --- Run dispatch backtesting --------------------------------------------
    backtest_result: dict | None = None

    if skip_backtest:
        info["backtest_skipped"] = True
    else:
        try:
            # Load feature and gold rows from pipeline
            if run_for_data is None:
                raise ValueError("Cannot run backtest without pipeline data")

            rows = list(read_table(Path(run_for_data.features.csv_path)))
            gold = list(read_table(Path(run_for_data.gold.csv_path)))

            # Run the backtest with the propagated data status
            backtest = run_dispatch_backtest(
                rows,
                gold,
                seed=seed,
                n_boot=n_boot,
            )
            backtest_result = backtest.to_dict()
            info["backtest_status"] = "ok"
        except Exception as e:
            errors.append(f"Backtest error: {e}")
            info["backtest_status"] = "failed"
            info["backtest_error"] = str(e)

    # --- Assemble final result -----------------------------------------------
    finished_at = datetime.now(UTC)

    reproducibility = {
        "seed": seed,
        "n_boot": n_boot,
        "forecast_train_fraction": forecast_split_fraction,
        "forecast_validation_fraction": forecast_validation_fraction,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": (finished_at - started_at).total_seconds(),
    }

    info["data_status"] = inferred_status
    if errors:
        info["errors"] = errors

    return OrchestrationResult(
        phase="5",
        data_status=inferred_status,
        pipeline=pipeline,
        forecast=forecast_result,
        backtest=backtest_result,
        info=info,
        reproducibility=reproducibility,
    )
