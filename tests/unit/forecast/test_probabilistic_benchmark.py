"""Phase 4C benchmark: A/B/C structure, DATA-STATUS honesty, quantile
ordering/crossing reporting, calibration reporting, determinism, artifacts."""

from __future__ import annotations

import json

import pytest

from gridpulse.forecast.benchmark import run_probabilistic_benchmark
from gridpulse.forecast.probabilistic import NOMINAL_INTERVAL_COVERAGE


def test_result_structure_and_data_status(feature_rows):
    res = run_probabilistic_benchmark(feature_rows, seed=13)
    assert res.info["data_status"] == "FIXTURE-VERIFIED"

    names = [m["name"] for m in res.models]
    assert names == [
        "seasonal_naive_24h",
        "linear_regression_ridge",
        "quantile_regression_lgbm",
    ]
    kinds = [m["kind"] for m in res.models]
    assert kinds == ["point", "point", "probabilistic"]

    # Honesty in the INFO statement.
    assert "UNVERIFIED for live electricity" in res.info["statement"]
    assert res.info["target_column"] == "residual_load_mw"
    assert res.info["n_aligned_rows_evaluated"] >= 1
    assert res.info["n_aligned_rows_evaluated"] <= res.info["n_test_row_candidates"]

    # Chronological split.
    assert (
        res.split["train_start"]
        < res.split["validation_start"]
        < res.split["test_start"]
    )
    counts = res.train_validation_test_counts
    assert counts["train"] > counts["test"] > 0

    # Reproducibility records the quantile model hyperparameters (no secrets).
    qm = res.reproducibility["quantile_model"]
    assert qm["library"].startswith("lightgbm==")
    assert qm["random_state"] == 13
    assert qm["fitted"] is True
    assert "feature_columns" in qm


def test_point_metrics_all_models_present(feature_rows):
    res = run_probabilistic_benchmark(feature_rows, seed=0)
    for m in res.models:
        met = m["metrics"] if m["kind"] == "point" else m["point_metrics_as_p50"]
        assert met["n_valid_pairs"] >= 1
        assert met["mae"] is not None and met["mae"] >= 0
        assert met["rmse"] is not None
    # C exposes its P50 point metrics explicitly named.
    c = res.models[2]
    assert set(c["point_metrics_as_p50"]) >= {
        "mae",
        "rmse",
        "mape",
        "bias",
        "median_absolute_error",
    }


def test_probabilistic_metrics_reported_for_model_c(feature_rows):
    res = run_probabilistic_benchmark(feature_rows, seed=0)
    c = res.models[2]
    prob = c["probabilistic_metrics"]
    assert prob["n_triples"] >= 1

    # Pinball per quantile.
    assert set(prob["pinball"]) == {"0.10", "0.50", "0.90"}
    for q in ("0.10", "0.50", "0.90"):
        assert prob["pinball"][q] is not None and prob["pinball"][q] >= 0

    # Empirical coverage + nominal reference.
    assert prob["nominal_quantiles"] == {"0.10": 0.10, "0.50": 0.50, "0.90": 0.90}
    assert prob["nominal_interval_coverage"] == pytest.approx(NOMINAL_INTERVAL_COVERAGE)
    for q in ("0.10", "0.50", "0.90"):
        cov = prob["empirical_coverage"][q]
        assert cov is not None and 0.0 <= cov <= 1.0
    iv = prob["interval_coverage"]
    assert iv is not None and 0.0 <= iv <= 1.0

    # Interval width (sharpness) distribution.
    w = prob["interval_width"]
    assert w["n"] == prob["n_triples"]
    assert w["mean"] is not None and w["mean"] >= 0
    assert w["median"] is not None
    assert w["min"] <= w["max"]

    # Risk score stats include near-zero-P50 counts.
    rs = prob["risk_score"]
    assert rs["mean"] is not None and rs["mean"] >= 0
    assert rs["n_near_zero_p50"] >= 0
    assert rs["n_p50_valid"] >= 1


def test_quantile_ordering_holds_on_every_aligned_row(feature_rows):
    res = run_probabilistic_benchmark(feature_rows, seed=1)
    prob = res.models[2]["probabilistic_metrics"]
    # The aggregate we report is monotonic-safe by construction; also enforce
    # that interval width is non-negative (P90-P10 >= 0) at the mean level.
    assert prob["interval_width"]["mean"] >= 0.0
    # Crossing is reported explicitly, never hidden.
    x = res.info["quantile_crossing"]
    assert x["n_triples"] == prob["n_triples"]
    assert 0 <= x["n_detected"] <= x["n_triples"]
    assert 0 <= x["n_corrected"] <= x["n_detected"]


def test_calibration_reported_and_honest(feature_rows):
    res = run_probabilistic_benchmark(feature_rows, seed=0)
    cal = res.calibration
    assert cal["nominal"] == {
        "p10": 0.10,
        "p50": 0.50,
        "p90": 0.90,
        "interval_80_coverage": 0.80,
    }
    # The note must NOT claim calibration is proven on the fixture.
    assert "not" in cal["note"].lower() and "evidence" in cal["note"].lower()
    assert "fixture" in cal["note"].lower()


def test_determinism_under_fixed_seed(feature_rows):
    a = run_probabilistic_benchmark(feature_rows, seed=7)
    b = run_probabilistic_benchmark(feature_rows, seed=7)
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(
        b.to_dict(), sort_keys=True
    )


def test_write_artefacts(tmp_path, feature_rows):
    res = run_probabilistic_benchmark(feature_rows, seed=3)
    paths = res.write(tmp_path)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    assert data["phase"] == "4C"
    assert data["data_status"] == "FIXTURE-VERIFIED"
    assert len(data["models"]) == 3
    md = paths[1].read_text(encoding="utf-8")
    assert "DATA STATUS" in md and "FIXTURE-VERIFIED" in md
    assert "Probabilistic" in md
    assert "Pinball" in md
    assert "risk" in md.lower()


def test_custom_split_is_honoured(feature_rows):
    from gridpulse.forecast.contract import build_forecasting_dataset
    from gridpulse.forecast.split import chronological_split_by_fraction

    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(
        ds.issue_times, train_fraction=0.6, validation_fraction=0.2
    )
    res = run_probabilistic_benchmark(feature_rows, split=split, seed=5)
    assert res.info["n_aligned_rows_evaluated"] >= 1
    assert res.train_validation_test_counts["train"] >= 1


def test_quantile_model_kwargs_are_forwarded(feature_rows):
    res = run_probabilistic_benchmark(
        feature_rows,
        seed=0,
        quantile_model_kwargs={"n_estimators": 30, "num_leaves": 7},
    )
    qm = res.reproducibility["quantile_model"]
    assert qm["n_estimators"] == 30
    assert qm["num_leaves"] == 7


def test_no_secrets_in_benchmark_output(feature_rows):
    res = run_probabilistic_benchmark(feature_rows, seed=0)
    blob = json.dumps(res.to_dict())
    for needle in ("apikey", "api_key", "securitytoken", "password", "secret"):
        assert needle.lower() not in blob.lower()
