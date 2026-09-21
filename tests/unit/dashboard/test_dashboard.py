"""Unit and integration tests for the GridPulse Streamlit operational dashboard (Phase 6C)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

MOCK_ORCH_REPORT = {
    "phase": "5",
    "data_status": "FIXTURE-VERIFIED",
    "pipeline": {"ok": True, "entsoe_available": True},
    "reproducibility": {"duration_seconds": 10.5},
    "forecast": {
        "metrics": {"mean_pinball_loss": 0.15, "mae": 50.0, "coverage": 0.82},
        "predictions": [
            {
                "target_utc": "2024-01-15T00:00:00Z",
                "p10": 1000.0,
                "p50": 1200.0,
                "p90": 1400.0,
                "actual": 1150.0,
            },
            {
                "target_utc": "2024-01-15T01:00:00Z",
                "p10": 1100.0,
                "p50": 1300.0,
                "p90": 1500.0,
                "actual": 1250.0,
            },
        ],
    },
    "backtest": {
        "n_days": 2,
        "n_dropped_days": 0,
        "strategies": {
            "greedy_arbitrage": {
                "total_cost": 100.0,
                "total_revenue": 200.0,
                "net_value": 100.0,
                "export_undercut_hours": 0,
            }
        },
    },
}


def test_dashboard_app_startup() -> None:
    """Test that the dashboard app loads and runs without unhandled exceptions using mocked orchestration."""
    app_path = str(
        Path(__file__).resolve().parent.parent.parent.parent
        / "src"
        / "gridpulse"
        / "dashboard"
        / "app.py"
    )
    with patch(
        "gridpulse.dashboard.app.get_orchestration_results",
        return_value=MOCK_ORCH_REPORT,
    ):
        at = AppTest.from_file(app_path)
        at.run(timeout=30)

        assert not at.exception, f"Dashboard raised exception: {at.exception}"
        title_texts = [t.value for t in at.title] + [h.value for h in at.header]
        assert any("GridPulse" in t or "Dashboard" in t for t in title_texts)


def test_dashboard_navigation_and_tabs() -> None:
    """Test navigating through the dashboard navigation radio options without unhandled exceptions."""
    app_path = str(
        Path(__file__).resolve().parent.parent.parent.parent
        / "src"
        / "gridpulse"
        / "dashboard"
        / "app.py"
    )
    with patch(
        "gridpulse.dashboard.app.get_orchestration_results",
        return_value=MOCK_ORCH_REPORT,
    ):
        at = AppTest.from_file(app_path)
        at.run(timeout=30)

        assert not at.exception

        # Test Probabilistic Forecasting tab
        if at.sidebar.radio:
            at.sidebar.radio[0].set_value("Probabilistic Forecasting")
            at.run()
            assert not at.exception

        # Test Battery Dispatch & Optimization tab
        if at.sidebar.radio:
            at.sidebar.radio[0].set_value("Battery Dispatch & Optimization")
            at.run()
            assert not at.exception
