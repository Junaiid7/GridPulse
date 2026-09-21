"""Unit tests for Phase 6A model persistence (save_model, load_model)."""

from __future__ import annotations

import json
from datetime import UTC

import numpy as np
import pytest

from gridpulse.forecast.contract import build_forecasting_dataset
from gridpulse.forecast.models import QuantileRegressionModel
from gridpulse.forecast.persistence import load_model, save_model
from gridpulse.forecast.split import chronological_split_by_fraction

UTC = UTC


def test_save_load_round_trip(feature_rows, tmp_path):
    """Test saving and loading a fitted model preserves round-trip state."""
    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    cols = [c for c in ds.predictor_columns if not c.startswith("weather_")][:4]

    model = QuantileRegressionModel(feature_columns=cols, random_state=42)
    model.fit(ds, start=split.train_start, end=split.train_end)
    assert model.metadata()["fitted"]

    model_dir = tmp_path / "test_model_bundle"
    saved_path = save_model(model, model_dir, data_status="FIXTURE-VERIFIED")
    assert saved_path == model_dir.resolve()
    assert (model_dir / "metadata.json").exists()
    assert (model_dir / "booster_p10.txt").exists()
    assert (model_dir / "booster_p50.txt").exists()
    assert (model_dir / "booster_p90.txt").exists()

    loaded_model = load_model(model_dir)
    assert loaded_model.metadata()["fitted"]
    assert loaded_model.feature_columns() == model.feature_columns()
    assert loaded_model.quantiles == model.quantiles
    assert loaded_model.n_estimators == model.n_estimators
    assert loaded_model.learning_rate == model.learning_rate


def test_hyperparameter_and_metadata_preservation(feature_rows, tmp_path):
    """Test that all hyperparameters and metadata fields are preserved upon load."""
    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    cols = [c for c in ds.predictor_columns if not c.startswith("weather_")][:3]

    model = QuantileRegressionModel(
        feature_columns=cols,
        quantiles=(0.1, 0.5, 0.9),
        n_estimators=50,
        learning_rate=0.08,
        num_leaves=7,
        min_child_samples=2,
        random_state=123,
        boost_from_average=False,
        min_train_rows=3,
    )
    model.fit(ds, start=split.train_start, end=split.train_end)

    model_dir = tmp_path / "bundle_meta"
    save_model(model, model_dir)

    loaded = load_model(model_dir)
    assert loaded.feature_columns() == cols
    assert loaded.quantiles == (0.1, 0.5, 0.9)
    assert loaded.n_estimators == 50
    assert loaded.learning_rate == 0.08
    assert loaded.num_leaves == 7
    assert loaded.min_child_samples == 2
    assert loaded.random_state == 123
    assert loaded.boost_from_average is False
    assert loaded.min_train_rows == 3

    meta = loaded.metadata()
    assert meta["model"] == "quantile_regression_lgbm"
    assert meta["fitted"] is True


def test_prediction_parity_after_load(feature_rows, tmp_path):
    """Test that predictions from loaded model match original model via np.testing.assert_allclose()."""
    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    cols = [c for c in ds.predictor_columns if not c.startswith("weather_")][:4]

    model = QuantileRegressionModel(feature_columns=cols, random_state=42)
    model.fit(ds, start=split.train_start, end=split.train_end)

    orig_preds = model.predict(
        ds, start=split.validation_start, end=split.validation_end
    )

    model_dir = tmp_path / "bundle_parity"
    save_model(model, model_dir)
    loaded = load_model(model_dir)

    loaded_preds = loaded.predict(
        ds, start=split.validation_start, end=split.validation_end
    )

    assert len(orig_preds) == len(loaded_preds)
    for op, lp in zip(orig_preds, loaded_preds):
        assert op.issue_time == lp.issue_time
        assert op.target_time == lp.target_time
        if op.p10 is not None and lp.p10 is not None:
            np.testing.assert_allclose(lp.p10, op.p10, rtol=1e-7, atol=1e-7)
        else:
            assert op.p10 == lp.p10

        if op.p50 is not None and lp.p50 is not None:
            np.testing.assert_allclose(lp.p50, op.p50, rtol=1e-7, atol=1e-7)
        else:
            assert op.p50 == lp.p50

        if op.p90 is not None and lp.p90 is not None:
            np.testing.assert_allclose(lp.p90, op.p90, rtol=1e-7, atol=1e-7)
        else:
            assert op.p90 == lp.p90


def test_unfitted_model_rejection(tmp_path):
    """Test that attempting to save an unfitted model raises ValueError."""
    model = QuantileRegressionModel(feature_columns=["hour"])
    with pytest.raises(ValueError, match="unfitted model"):
        save_model(model, tmp_path / "unfitted")


def test_missing_or_corrupted_bundle_errors(tmp_path):
    """Test loading non-existent paths, missing metadata, or missing boosters raises appropriate errors."""
    # 1. Non-existent path
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "non_existent")

    # 2. Directory missing metadata.json
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="metadata not found"):
        load_model(empty_dir)

    # 3. Corrupted metadata JSON
    corrupt_meta = tmp_path / "corrupt_meta"
    corrupt_meta.mkdir()
    (corrupt_meta / "metadata.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid model metadata JSON"):
        load_model(corrupt_meta)

    # 4. Missing booster file
    missing_booster = tmp_path / "missing_booster"
    missing_booster.mkdir()
    payload = {
        "metadata": {"feature_columns": ["hour"], "quantiles": [0.5], "fitted": True},
        "booster_files": {"0.5": "non_existent_booster.txt"},
    }
    (missing_booster / "metadata.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="Missing booster file"):
        load_model(missing_booster)
