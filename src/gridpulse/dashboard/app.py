"""GridPulse 2.0 — Streamlit Operational Dashboard (Phase 6C).

An offline-compatible operational dashboard for uncertainty-aware residual-load
forecasting and battery storage dispatch optimisation in the Dutch electricity market.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from gridpulse.optimization.backtest import run_dispatch_backtest
from gridpulse.optimization.battery import BatteryConfig
from gridpulse.orchestration.runner import run_e2e_orchestration

st.set_page_config(
    page_title="GridPulse 2.0 Operational Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading GridPulse E2E Orchestration report...")
def get_orchestration_results() -> dict[str, Any]:
    """Load cached E2E orchestration report or run orchestration if missing."""
    report_path = Path("data/reports/orchestration_phase5.json")
    if report_path.exists():
        try:
            with open(report_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    result = run_e2e_orchestration(skip_pipeline=False, n_boot=1000, seed=0)
    return result.to_dict()


def main() -> None:
    st.title("⚡ GridPulse 2.0 — Operational Dashboard")
    st.markdown(
        "Uncertainty-aware residual-load forecasting and battery storage dispatch optimisation "
        "for the Dutch electricity market."
    )

    # Load orchestration results
    try:
        orch = get_orchestration_results()
    except Exception as e:
        st.error(f"Failed to execute orchestration: {e}")
        return

    data_status = orch.get("data_status", "FIXTURE-VERIFIED")

    # Sidebar status & navigation
    st.sidebar.header("Operational Status")
    if data_status == "FIXTURE-VERIFIED":
        st.sidebar.warning(
            f"Data Status: **{data_status}**\n\n(Deterministic synthetic fixtures)"
        )
    else:
        st.sidebar.success(f"Data Status: **{data_status}**")

    st.sidebar.markdown("---")
    tab_choice = st.sidebar.radio(
        "Navigation",
        [
            "Orchestration Overview",
            "Probabilistic Forecasting",
            "Battery Dispatch & Optimization",
        ],
    )

    if tab_choice == "Orchestration Overview":
        render_orchestration_overview(orch)
    elif tab_choice == "Probabilistic Forecasting":
        render_probabilistic_forecasting(orch)
    elif tab_choice == "Battery Dispatch & Optimization":
        render_battery_optimization(orch)


def render_orchestration_overview(orch: dict[str, Any]) -> None:
    st.header("Pipeline & Orchestration Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Phase", orch.get("phase", "5"))
    with col2:
        st.metric("Data Status", orch.get("data_status", "FIXTURE-VERIFIED"))
    with col3:
        pipe = orch.get("pipeline") or {}
        st.metric("Pipeline OK", str(pipe.get("ok", True)))
    with col4:
        repro = orch.get("reproducibility") or {}
        duration = repro.get("duration_seconds", 0.0)
        st.metric("Duration (s)", f"{duration:.2f}s" if duration else "N/A")

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Pipeline Details")
        st.json(orch.get("pipeline", {}))
    with col_r:
        st.subheader("Reproducibility & Metadata")
        st.json(orch.get("reproducibility", {}))

    st.subheader("Full Orchestration JSON Report")
    with st.expander("View Raw JSON Report"):
        st.json(orch)


def render_probabilistic_forecasting(orch: dict[str, Any]) -> None:
    st.header("Probabilistic Residual-Load Forecasting")

    forecast_data = orch.get("forecast")
    if not forecast_data:
        st.warning("No forecast data available in orchestration results.")
        return

    st.markdown(
        "Quantile regression forecasts (P10, P50, P90) modeling residual load uncertainty "
        "derived from market and weather features."
    )

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    metrics = forecast_data.get("metrics", {})
    with col1:
        st.metric("Mean Pinball Loss", f"{metrics.get('mean_pinball_loss', 0.0):.3f}")
    with col2:
        st.metric("P50 MAE", f"{metrics.get('mae', 0.0):.3f}")
    with col3:
        st.metric("Coverage (P10-P90)", f"{metrics.get('coverage', 0.8):.1%}")

    # Plot uncertainty bands if predictions available
    predictions = forecast_data.get("predictions", [])
    if predictions:
        st.subheader("Residual Load Forecast & Uncertainty Bands (P10 - P90)")

        times = [
            p.get("target_utc") or p.get("target_local") or str(i)
            for i, p in enumerate(predictions)
        ]
        p10 = [p.get("p10", p.get("y_pred_0.1", 0)) for p in predictions]
        p50 = [p.get("p50", p.get("y_pred_0.5", 0)) for p in predictions]
        p90 = [p.get("p90", p.get("y_pred_0.9", 0)) for p in predictions]
        actuals = [p.get("actual", p.get("y_actual", None)) for p in predictions]

        fig = go.Figure()
        # Uncertainty band (P10 to P90)
        fig.add_trace(
            go.Scatter(
                x=times + times[::-1],
                y=p90 + p10[::-1],
                fill="toself",
                fillcolor="rgba(0, 100, 255, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name="P10-P90 Uncertainty Band",
                showlegend=True,
            )
        )
        # P50 Median forecast
        fig.add_trace(
            go.Scatter(
                x=times,
                y=p50,
                mode="lines",
                line=dict(color="rgb(0, 100, 255)", width=2),
                name="P50 Forecast",
            )
        )
        if any(a is not None for a in actuals):
            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=actuals,
                    mode="lines+markers",
                    line=dict(color="rgb(34, 139, 34)", width=1.5),
                    name="Actual Residual Load",
                )
            )

        fig.update_layout(
            title="Residual Load Quantile Forecasts",
            xaxis_title="Time",
            yaxis_title="Residual Load (MW)",
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "Detailed per-timestamp prediction series not found in forecast summary."
        )

    with st.expander("View Raw Forecast Summary"):
        st.json(forecast_data)


def render_battery_optimization(orch: dict[str, Any]) -> None:
    st.header("Battery Storage Dispatch & Optimization")

    st.markdown(
        "Configure battery storage parameters and evaluate optimal dispatch strategies "
        "(e.g., price arbitrage, uncertainty-aware curtailment)."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Battery Configuration")
    capacity = st.sidebar.number_input(
        "Capacity (MWh)", value=10.0, min_value=0.1, max_value=1000.0, step=1.0
    )
    max_power = st.sidebar.number_input(
        "Max Power (MW)", value=5.0, min_value=0.1, max_value=500.0, step=0.5
    )
    initial_soc = st.sidebar.number_input(
        "Initial SoC (MWh)", value=5.0, min_value=0.0, max_value=capacity, step=0.5
    )
    efficiency = st.sidebar.slider(
        "Round-trip Efficiency", min_value=0.5, max_value=1.0, value=0.9, step=0.01
    )
    cycle_cost = st.sidebar.number_input(
        "Cycle Degradation Cost (€/MWh)",
        value=1.0,
        min_value=0.0,
        max_value=50.0,
        step=0.5,
    )

    backtest_data = orch.get("backtest")
    if backtest_data:
        st.subheader("Historical Dispatch Backtest Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Strategies Evaluated", len(backtest_data.get("strategies", {})))
        with col2:
            st.metric("Evaluated Days", backtest_data.get("n_days", 0))
        with col3:
            st.metric("Dropped Infeasible Days", backtest_data.get("n_dropped_days", 0))

        strategies = backtest_data.get("strategies", {})
        if strategies:
            st.subheader("Strategy Performance Comparison")
            summary_rows = []
            strat_items = (
                strategies.items()
                if isinstance(strategies, dict)
                else [
                    (s.get("strategy", f"strat_{i}"), s)
                    for i, s in enumerate(strategies)
                ]
            )
            for strat_name, strat_res in strat_items:
                summary_rows.append(
                    {
                        "Strategy": strat_name,
                        "Total Cost (€)": strat_res.get("total_cost", 0.0),
                        "Total Revenue (€)": strat_res.get("total_revenue", 0.0),
                        "Net Value (€)": strat_res.get("net_value", 0.0),
                        "Export Undercut Hours": strat_res.get(
                            "export_undercut_hours", 0
                        ),
                    }
                )
            st.dataframe(summary_rows, use_container_width=True)

        with st.expander("View Raw Backtest Report"):
            st.json(backtest_data)
    else:
        st.info(
            "No backtest data in orchestration report. Run manual dispatch simulation below."
        )

    st.subheader("Interactive Battery Dispatch Simulation")
    if st.button("Run Quick Battery Dispatch Simulation"):
        try:
            battery_cfg = BatteryConfig(
                capacity_mwh=capacity,
                max_power_mw=max_power,
                initial_soc_mwh=initial_soc,
                efficiency=efficiency,
                cycle_cost_eur_per_mwh=cycle_cost,
            )
            # Run offline backtest with custom battery config
            bt_res = run_dispatch_backtest(battery=battery_cfg)
            st.success(
                f"Simulation completed successfully across {bt_res.n_days} days!"
            )
            st.json(bt_res.to_dict())
        except Exception as e:
            st.error(f"Dispatch simulation failed: {e}")


if __name__ == "__main__":
    main()
