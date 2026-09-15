"""Ridge/OLS linear-regression baseline: recovery, missing-cell dropping,
train-only standardisation, insufficient rows, save/load."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gridpulse.forecast.contract import build_forecasting_dataset
from gridpulse.forecast.models import LinearRegressionModel

UTC = timezone.utc


def _linear_rows(n: int, *, start: datetime = datetime(2024, 1, 1, 6, 0, tzinfo=UTC)) -> list[dict]:
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


def test_recovers_linear_relation():
    rows = _linear_rows(24)
    ds = build_forecasting_dataset(rows)
    model = LinearRegressionModel(feature_columns=["x_feat"], ridge=1e-9)
    lo, mid = ds.rows[1].issue_time, ds.rows[12].issue_time
    hi = ds.rows[-1].issue_time + timedelta(hours=1)
    model.fit(ds, start=lo, end=mid)
    assert model.metadata()["fitted"]
    preds = model.predict(ds, start=mid, end=hi)
    for r, p in zip([r for r in ds.rows if mid <= r.issue_time < hi], preds):
        # The target is 24h ahead of the issue, so the target row's x_feat
        # determines the label. Extract it from the target_mw back-calculation.
        expected_y = float(r.target_mw)
        assert p is not None
        assert p == pytest.approx(expected_y, abs=1e-6)


def test_missing_cells_are_dropped_never_filled():
    rows = _linear_rows(24)
    # Third row loses its feature -> that issue row is incomplete.
    missing_issue = datetime(2024, 1, 3, 6, 0, tzinfo=UTC)
    for row in rows:
        if row["target_utc"] == missing_issue.isoformat():
            row["x_feat"] = ""
    ds = build_forecasting_dataset(rows)
    model = LinearRegressionModel(feature_columns=["x_feat"])
    lo, hi = ds.rows[1].issue_time, ds.rows[-1].issue_time + timedelta(hours=1)
    model.fit(ds, start=lo, end=hi)
    meta = model.metadata()
    assert meta["n_rows_dropped_missing_features"] >= 1
    preds = model.predict(ds, start=lo, end=hi)
    rows_in_window = [r for r in ds.rows if lo <= r.issue_time < hi]
    missing = [r for r in rows_in_window if not r.has_features()]
    assert len(missing) == 1
    assert preds[rows_in_window.index(missing[0])] is None


def test_train_only_standardisation_is_deterministic():
    rows = _linear_rows(24)
    ds = build_forecasting_dataset(rows)
    lo, mid = ds.rows[1].issue_time, ds.rows[12].issue_time
    m1 = LinearRegressionModel(feature_columns=["x_feat"])
    m1.fit(ds, start=lo, end=mid)
    m2 = LinearRegressionModel(feature_columns=["x_feat"])
    m2.fit(ds, start=lo, end=mid)
    assert m1.metadata()["beta_coefficients"] == m2.metadata()["beta_coefficients"]
    assert m1.metadata()["predictor_center_train_only"] == m2.metadata()["predictor_center_train_only"]


def test_too_few_rows_means_unfitted_and_all_none():
    # 3 daily rows -> 2 issues; a fit window holding a single complete row is
    # fewer observations than (features + intercept), so the model must refuse.
    rows = _linear_rows(3)
    ds = build_forecasting_dataset(rows)
    assert len(ds) == 2
    model = LinearRegressionModel(feature_columns=["x_feat"])
    lo = ds.rows[1].issue_time
    hi = ds.rows[1].issue_time + timedelta(hours=1)  # window with just 1 row
    model.fit(ds, start=lo, end=hi)
    assert not model.metadata()["fitted"]
    assert model.predict(ds, start=lo, end=hi) == [None]


def test_save_load_round_trip(tmp_path):
    rows = _linear_rows(24)
    ds = build_forecasting_dataset(rows)
    model = LinearRegressionModel(feature_columns=["x_feat"], ridge=1e-8)
    lo, mid = ds.rows[0].issue_time, ds.rows[12].issue_time
    model.fit(ds, start=lo, end=mid)

    path = tmp_path / "lr.json"
    model.save(path)
    loaded = LinearRegressionModel()
    loaded.load(path)
    assert loaded.feature_columns() == model.feature_columns()
    assert loaded.metadata()["ridge"] == pytest.approx(1e-8)
    assert loaded.metadata()["fitted"] == model.metadata()["fitted"]
    assert loaded.metadata()["beta_coefficients"] == pytest.approx(model.metadata()["beta_coefficients"])

    # Loaded model predicts identically.
    hi = ds.rows[-1].issue_time + timedelta(hours=1)
    assert loaded.predict(ds, start=lo, end=hi) == model.predict(ds, start=lo, end=hi)