"""Phase 4C LightGBM quantile-regression model: fit, predict, ordering,
missing cells, insufficient rows, determinism, save/load."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from gridpulse.forecast.contract import build_forecasting_dataset
from gridpulse.forecast.models import QuantileRegressionModel
from gridpulse.forecast.probabilistic import QUANTILES

UTC = UTC


def _linear_rows(
    n: int, *, start: datetime = datetime(2024, 1, 1, 6, 0, tzinfo=UTC)
) -> list[dict]:
    rows = []
    for i in range(n):
        t = start + timedelta(hours=24 * i)
        x = float(i)
        rows.append(
            {
                "target_utc": t.isoformat(),
                "target_local": t.isoformat(),
                "residual_load_mw": repr(2.0 * x + 3.0),
                "x_feat": repr(x),
            }
        )
    return rows


def test_fit_and_predict_produce_ordered_quantile_triples(feature_rows):
    from gridpulse.forecast.contract import build_forecasting_dataset
    from gridpulse.forecast.split import chronological_split_by_fraction

    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    cols = [c for c in ds.predictor_columns if not c.startswith("weather_")][:6]
    model = QuantileRegressionModel(feature_columns=cols, random_state=0)
    model.fit(ds, start=split.train_start, end=split.train_end)
    assert model.metadata()["fitted"]
    assert model.metadata()["quantiles"] == list(QUANTILES)

    preds = model.predict(ds, start=split.validation_start, end=split.validation_end)
    rows = [
        r
        for r in ds.rows
        if split.validation_start <= r.issue_time < split.validation_end
    ]
    assert len(preds) == len(rows)
    for p, r in zip(preds, rows):
        assert p.issue_time == r.issue_time
        assert p.target_time == r.target_time
        if p.has_quantiles:
            # Contract guarantees ordering even if the raw triples crossed.
            assert p.p10 <= p.p50 <= p.p90
    # Crossing stats are present and self-consistent.
    x = model.crossing_stats()
    assert x["n_triples"] >= 0
    assert 0 <= x["n_detected"] <= x["n_triples"]
    assert x["n_corrected"] <= x["n_detected"]


def test_recovers_linear_relation_in_sample():
    """On a deterministic linear row sequence the P50 should recover the
    relation for rows inside the trained feature range. (Out-of-sample recovery
    is not expected: LightGBM trees are piecewise-constant and cannot
    extrapolate beyond the feature range they were fit on; the correct check
    for tree granularity is in-sample.)"""
    rows = _linear_rows(30)
    ds = build_forecasting_dataset(rows)
    lo = ds.rows[1].issue_time
    mid = ds.rows[20].issue_time
    model = QuantileRegressionModel(
        feature_columns=["x_feat"], random_state=0, n_estimators=120
    )
    model.fit(ds, start=lo, end=mid)
    assert model.metadata()["fitted"]
    # Predict the SAME window that was fit on (in-sample recovery).
    preds = model.predict(ds, start=lo, end=mid)
    evals = [
        (p, r)
        for p, r in zip(preds, [r for r in ds.rows if lo <= r.issue_time < mid])
        if p.has_quantiles
    ]
    if not evals:
        pytest.skip("no complete quantile triples available")
    median_errors = [abs(p.p50 - r.target_mw) for p, r in evals]
    # Piecewise-constant tree granularity on a clean line: in-sample max error
    # stays small.
    assert max(median_errors) < 5.0
    assert sum(median_errors) / len(median_errors) < 2.0


def test_missing_cells_yield_none_never_filled():
    from gridpulse.forecast.contract import build_forecasting_dataset
    from gridpulse.forecast.split import chronological_split_by_fraction

    rows = _linear_rows(30)
    ds = build_forecasting_dataset(rows)
    split = chronological_split_by_fraction(
        ds.issue_times, train_fraction=0.7, validation_fraction=0.15
    )
    # Pick a specific validation-row issue time, then blank its predictor in a
    # fresh copy of the rows so that the *prediction-time* cell is missing
    # (while training stays clean).
    val_rows = [
        r
        for r in ds.rows
        if split.validation_start <= r.issue_time < split.validation_end
    ]
    assert val_rows
    blank_issue = val_rows[0].issue_time.isoformat()
    mutated = [dict(row) for row in rows]
    for row in mutated:
        if row["target_utc"] == blank_issue:
            row["x_feat"] = ""
    ds2 = build_forecasting_dataset(mutated)

    # Train on the pristine dataset (no missing cells in train).
    model = QuantileRegressionModel(feature_columns=["x_feat"], random_state=0)
    model.fit(ds, start=split.train_start, end=split.train_end)
    assert model.metadata()["n_rows_dropped_missing_features"] == 0
    assert model.metadata()["fitted"]

    # The blanked validation row yields an all-None triple — never a fill.
    preds = model.predict(ds2, start=split.validation_start, end=split.validation_end)
    by_issue = {p.issue_time: p for p in preds}
    assert by_issue[val_rows[0].issue_time].has_quantiles is False
    # Every other validation row still produces quantiles.
    others = [p for ts, p in by_issue.items() if ts != val_rows[0].issue_time]
    assert any(p.has_quantiles for p in others)


def test_too_few_rows_means_unfitted_and_all_none():
    rows = _linear_rows(3)
    ds = build_forecasting_dataset(rows)
    model = QuantileRegressionModel(feature_columns=["x_feat"], min_train_rows=5)
    model.fit(ds, start=ds.rows[0].issue_time, end=ds.rows[-1].issue_time)
    assert not model.metadata()["fitted"]
    preds = model.predict(
        ds, start=ds.rows[0].issue_time, end=ds.rows[-1].issue_time + timedelta(hours=1)
    )
    assert all(not p.has_quantiles for p in preds)


def test_determinism_under_fixed_seed(feature_rows):
    from gridpulse.forecast.contract import build_forecasting_dataset
    from gridpulse.forecast.split import chronological_split_by_fraction

    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    cols = [c for c in ds.predictor_columns if not c.startswith("weather_")][:6]

    m1 = QuantileRegressionModel(feature_columns=cols, random_state=11)
    m1.fit(ds, start=split.train_start, end=split.train_end)
    p1 = [
        (p.p10, p.p50, p.p90)
        for p in m1.predict(ds, start=split.test_start, end=split.test_end)
    ]

    m2 = QuantileRegressionModel(feature_columns=cols, random_state=11)
    m2.fit(ds, start=split.train_start, end=split.train_end)
    p2 = [
        (p.p10, p.p50, p.p90)
        for p in m2.predict(ds, start=split.test_start, end=split.test_end)
    ]
    assert p1 == p2


def test_save_load_round_trip(feature_rows, tmp_path):
    from gridpulse.forecast.contract import build_forecasting_dataset
    from gridpulse.forecast.split import chronological_split_by_fraction

    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    cols = [c for c in ds.predictor_columns if not c.startswith("weather_")][:6]
    model = QuantileRegressionModel(feature_columns=cols, random_state=3)
    model.fit(ds, start=split.train_start, end=split.train_end)

    path = tmp_path / "quantile.json"
    model.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["model"] == "quantile_regression_lgbm"
    assert len(payload["boosters"]) == 3  # one base64 tree string per quantile

    loaded = QuantileRegressionModel()
    loaded.load(path)
    assert loaded.metadata()["fitted"] == model.metadata()["fitted"]
    a = [
        (p.p10, p.p50, p.p90)
        for p in model.predict(ds, start=split.test_start, end=split.test_end)
    ]
    b = [
        (p.p10, p.p50, p.p90)
        for p in loaded.predict(ds, start=split.test_start, end=split.test_end)
    ]
    assert a == b


def test_quantiles_used_are_the_p10_p50_p90_triple():
    model = QuantileRegressionModel(feature_columns=["x_feat"])
    assert model.quantiles == (0.10, 0.50, 0.90)
    # Metadata records them.
    assert model.metadata()["quantiles"] == [0.10, 0.50, 0.90]


def test_crossing_stats_reset_after_refit():
    rows = _linear_rows(30)
    ds = build_forecasting_dataset(rows)
    lo = ds.rows[0].issue_time
    model = QuantileRegressionModel(feature_columns=["x_feat"], min_train_rows=200)
    model.fit(ds, start=lo, end=lo + timedelta(hours=1))
    # Unfitted predict must reset crossing stats to zeros.
    model.predict(ds, start=lo, end=lo + timedelta(hours=25))
    assert model.crossing_stats() == {"n_detected": 0, "n_corrected": 0, "n_triples": 0}
