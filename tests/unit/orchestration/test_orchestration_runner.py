"""Unit tests for Phase 5 end-to-end orchestration runner."""

from __future__ import annotations

from pathlib import Path

from gridpulse.orchestration.runner import (
    VALID_DATA_STATUSES,
    OrchestrationResult,
    run_e2e_orchestration,
)


def test_orchestration_skip_pipeline_runs_successfully(tmp_path: Path):
    """Test running E2E orchestration with skip_pipeline=True (uses synthetic fixtures)."""
    result = run_e2e_orchestration(
        skip_pipeline=True,
        scratch_dir=tmp_path,
        n_boot=100,  # fast for testing
    )

    assert isinstance(result, OrchestrationResult)
    assert result.phase == "5"
    assert result.data_status == "FIXTURE-VERIFIED"
    assert result.data_status in VALID_DATA_STATUSES

    # Forecast and backtest should be present
    assert result.forecast is not None
    assert result.backtest is not None

    # Check serialization
    d = result.to_dict()
    assert d["phase"] == "5"
    assert d["data_status"] == "FIXTURE-VERIFIED"
    assert "forecast" in d
    assert "backtest" in d
    assert "reproducibility" in d


def test_orchestration_full_e2e_run(tmp_path: Path):
    """Test running full E2E orchestration including pipeline, forecast, and backtest."""
    result = run_e2e_orchestration(
        skip_pipeline=False,
        skip_forecast=False,
        skip_backtest=False,
        scratch_dir=tmp_path,
        n_boot=100,
    )

    assert isinstance(result, OrchestrationResult)
    assert result.phase == "5"
    assert result.data_status == "FIXTURE-VERIFIED"
    assert result.pipeline is not None
    assert result.forecast is not None
    assert result.backtest is not None
    assert result.pipeline.ok is True


def test_orchestration_skip_steps(tmp_path: Path):
    """Test skipping individual orchestration steps."""
    result = run_e2e_orchestration(
        skip_pipeline=False,
        skip_forecast=True,
        skip_backtest=True,
        scratch_dir=tmp_path,
    )

    assert result.pipeline is not None
    assert result.forecast is None
    assert result.backtest is None
    assert result.data_status == "FIXTURE-VERIFIED"
    assert result.info.get("forecast_skipped") is True
    assert result.info.get("backtest_skipped") is True
