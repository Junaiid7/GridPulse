"""Phase-8 baseline comparison: structure, DATA STATUS flag, determinism,
write artifacts."""

from __future__ import annotations

import json

from gridpulse.forecast.benchmark import run_benchmark


def test_result_structure_and_data_status(feature_rows):
    res = run_benchmark(feature_rows, seed=13)
    assert res.info["data_status"] == "FIXTURE-VERIFIED"

    names = [m["name"] for m in res.models]
    assert names == ["seasonal_naive_24h", "linear_regression_ridge"]  # A then B

    for m in res.models:
        met = m["metrics"]
        assert met["n_valid_pairs"] >= 3
        assert met["mae"] is not None and met["mae"] >= 0
        assert met["rmse"] is not None
        assert met["mape_invalid_count"] >= 0

    comp = res.comparison
    assert comp["direction"] == "MAE_A - MAE_B"
    assert comp["difference"] is not None
    assert comp["ci_low"] <= comp["ci_high"]
    assert comp["n_boot"] > 0

    # INFO honesty.
    assert "UNVERIFIED for live electricity" in res.info["statement"]
    assert res.info["target_column"] == "residual_load_mw"
    assert res.info["n_aligned_rows_evaluated"] <= res.info["n_test_row_candidates"]

    # Chronological split reported with monotonic boundaries.
    assert res.split["train_start"] < res.split["validation_start"] < res.split["test_start"]
    counts = res.train_validation_test_counts
    assert counts["train"] > counts["test"] > 0

    # Reproducibility.
    assert res.reproducibility["seed"] == 13
    assert isinstance(res.reproducibility["gridpulse_version"], str)


def test_naive_is_a_sane_baseline_on_the_synthetic(feature_rows):
    """On a strongly-24h-periodic residual, the naive 24h baseline must be far
    better than a blind constant (i.e. MAE is small relative to the residual
    scale); we do NOT hard-assert a specific winner — the fixture report states
    the observed A vs B outcome."""
    res = run_benchmark(feature_rows, seed=0)
    naive = res.models[0]["metrics"]           # model A = seasonal naive
    assert naive["mae"] is not None
    assert naive["mae"] < 800.0                 # ~±50-300 MW drift, not thousands


def test_determinism_under_fixed_seed(feature_rows):
    a = run_benchmark(feature_rows, seed=7)
    b = run_benchmark(feature_rows, seed=7)
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)


def test_write_artefacts(tmp_path, feature_rows):
    res = run_benchmark(feature_rows, seed=3)
    paths = res.write(tmp_path)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)
    data = json.loads(paths[0].read_text(encoding="utf-8"))
    assert data["data_status"] == "FIXTURE-VERIFIED"
    assert len(data["models"]) == 2
    md = paths[1].read_text(encoding="utf-8")
    assert "DATA STATUS" in md and "FIXTURE-VERIFIED" in md
    assert "MAE difference" in md


def test_custom_split_is_honoured(feature_rows):
    from gridpulse.forecast.contract import build_forecasting_dataset
    from gridpulse.forecast.split import chronological_split_by_fraction

    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times, train_fraction=0.6, validation_fraction=0.2)
    res = run_benchmark(feature_rows, split=split, seed=5)
    assert res.info["n_aligned_rows_evaluated"] >= 1
    # A smaller train window still yields a complete result.
    assert res.train_validation_test_counts["train"] >= 1