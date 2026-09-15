"""Forecasting dataset contract: issue selection, predictors, label, vintage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gridpulse.forecast.contract import (
    CORE_FEATURE_COLUMNS,
    ForecastRow,
    build_forecasting_dataset,
)

UTC = timezone.utc


def _by_ts(rows):
    return {datetime.fromisoformat(r["target_utc"]): r for r in rows}


def test_issue_selection_and_cadence(feature_rows):
    ds = build_forecasting_dataset(feature_rows)
    assert len(ds) > 0
    for r in ds.rows:
        assert r.issue_time.hour == 6          # once-daily issue at 06:00 UTC
        assert r.issue_time.minute == 0
        assert r.target_time - r.issue_time == timedelta(hours=24)
    assert ds.metadata["issue_cadence"] == "once_daily"
    assert ds.metadata["issue_hour_utc"] == 6
    assert ds.metadata["horizon_hours"] == 24
    assert ds.metadata["naive_lag_hours"] == 24
    assert ds.metadata["rows_requested"] == len(feature_rows)


def test_predictors_are_all_core_plus_weather(feature_rows, synthetic_pipeline):
    ds = build_forecasting_dataset(feature_rows)
    assert set(CORE_FEATURE_COLUMNS) <= set(ds.predictor_columns)
    weather_cols = {c for c in synthetic_pipeline.features.columns if c.startswith("weather_")}
    assert weather_cols <= set(ds.predictor_columns)
    # No target/label column sneaks into predictors.
    assert "residual_load_mw" not in ds.predictor_columns
    assert "target_utc" not in ds.predictor_columns

    # Feature rows produce non-None predictors for inner-window issues after
    # sufficient warm-up. lag_168h needs 7 days, so skip the first ~7 days.
    inner = [r for r in ds.rows if ds.rows.index(r) > 7]
    for r in inner:
        assert r.has_features()


def test_label_and_vintage_from_the_right_rows(feature_rows):
    rows = feature_rows
    by_ts = _by_ts(rows)
    ds = build_forecasting_dataset(rows)
    for r in ds.rows:
        issue_raw = by_ts[r.issue_time]
        target_raw = by_ts[r.target_time]
        # label = realised residual of the TARGET hour
        assert r.target_mw == float(target_raw["residual_load_mw"])
        # naive vintage = residual at (target - 24h) == the issue row's residual
        assert r.naive_vintage_mw == float(issue_raw["residual_load_mw"])


def test_non_issue_rows_are_not_counts(feature_rows):
    ds = build_forecasting_dataset(feature_rows)
    n_issues = sum(1 for r in ds.rows)
    n_mismatch = sum(1 for r in feature_rows if datetime.fromisoformat(r["target_utc"]).hour != 6)
    assert ds.metadata["rows_mismatched_issue_hour"] == n_mismatch
    assert ds.metadata["rows_skipped_no_label"] >= 0
    assert ds.metadata["rows_requested"] == n_issues + n_mismatch + ds.metadata["rows_skipped_no_label"]


def test_rows_are_sorted_by_issue_time(feature_rows):
    ds = build_forecasting_dataset(feature_rows)
    times = [r.issue_time for r in ds.rows]
    assert times == sorted(times)
    assert ds.issue_times == tuple(times)


def test_horizon_via_contract_parameter(feature_rows):
    # A 1-hour horizon at the same issue hour simply shifts the label window.
    ds = build_forecasting_dataset(feature_rows, issue_hour_utc=6, horizon_hours=1)
    for r in ds.rows:
        assert (r.target_time - r.issue_time).total_seconds() == 3600