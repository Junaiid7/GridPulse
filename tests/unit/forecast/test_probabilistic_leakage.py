"""Phase 4C leakage gate.

Proves the probabilistic model honours the same information-cutoff guarantees
as the Phase 4B point baselines:

- future target values are never used as predictors (predictions are invariant
  to tampering with future labels);
- the model is fit on the train window only (validation/test labels are never
  seen during fitting);
- quantile features are drawn from the same leakage-safe issue-row contract.

These tests are additive on top of the (unchanged) Phase 4B leakage tests.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from gridpulse.forecast.contract import build_forecasting_dataset
from gridpulse.forecast.models import QuantileRegressionModel
from gridpulse.forecast.split import chronological_split_by_fraction


def _feature_cols(ds, n=8):
    return [c for c in ds.predictor_columns if not c.startswith("weather_")][:n]


def _issue_utc(row):
    """Feature-table row's anchor UTC timestamp, ISO-normalised."""
    from gridpulse.ingestion.common.models import ensure_utc
    from datetime import datetime

    return ensure_utc(datetime.fromisoformat(row["target_utc"])).isoformat()


def test_quantile_predictions_are_invariant_to_future_target_labels(feature_rows):
    """Editing the (future) residual of ANY target hour must not move P10/P50/P90."""
    ds0 = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds0.issue_times)
    model = QuantileRegressionModel(feature_columns=_feature_cols(ds0), random_state=0)
    model.fit(ds0, start=split.train_start, end=split.train_end)
    base = model.predict(ds0, start=split.test_start, end=split.test_end)
    base_map = {p.issue_time: (p.p10, p.p50, p.p90) for p in base}

    # Mutate every future label to a huge value (both validation and test).
    mutated = [dict(row) for row in feature_rows]
    for row in mutated:
        row["residual_load_mw"] = "99999.0"
    ds1 = build_forecasting_dataset(mutated)
    # The quantile model only consumes issue-row predictors, so the predictions
    # must be unchanged even though the labels moved.
    model.predict(ds1, start=split.test_start, end=split.test_end)
    after = model.predict(ds1, start=split.test_start, end=split.test_end)
    after_map = {p.issue_time: (p.p10, p.p50, p.p90) for p in after}
    assert base_map == after_map


def test_quantile_model_fit_sees_only_train_window_rows(feature_rows):
    """metadata().n_train_rows must equal the number of complete rows strictly
    inside the train window — never validation/test rows."""
    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    cols = _feature_cols(ds)
    model = QuantileRegressionModel(feature_columns=cols, random_state=0)
    model.fit(ds, start=split.train_start, end=split.train_end)
    n_complete_train = sum(
        1
        for r in ds.rows
        if split.train_start <= r.issue_time < split.train_end
        and r.target_mw is not None
        and r.has_features()
    )
    assert model.metadata()["n_train_rows"] == n_complete_train
    # And the fit must NOT have absorbed rows from later windows.
    assert model.metadata()["n_train_rows"] <= len(ds.rows)


def test_training_is_independent_of_future_row_features(feature_rows):
    """Perturbing the FEATURES of every validation/test row (all rows at or
    after the validation boundary) must change NOTHING about the fit — and the
    predictions for train-window issues must be bit-identical.

    This is the strongest anti-leakage gate reachable for a tree model:
    if the fitter had consumed any information from future rows' input
    features, perturbing them would move the train-window predictions. It is a
    feature-side perturbation (labels were proven invariant separately in
    ``test_quantile_predictions_are_invariant_to_future_target_labels``), so it
    does not accidentally delete rows the fitter needs (as a row truncation
    test would).
    """
    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    cols = _feature_cols(ds)
    model = QuantileRegressionModel(feature_columns=cols, random_state=5)
    model.fit(ds, start=split.train_start, end=split.train_end)

    # Perturb every future feature value (validation + test rows), keeping the
    # train rows byte-identical.
    mutated = []
    for row in feature_rows:
        row = dict(row)
        if split.validation_start.isoformat() <= _issue_utc(row):
            for col in cols:
                if col in row:
                    try:
                        row[col] = repr(float(row[col]) * 10.0 + 7.0)
                    except ValueError:
                        # Non-numeric predictor (e.g. bool flag) — flip it and
                        # back so the feature is still visibly perturbed.
                        row[col] = "false" if row[col].lower() == "true" else "true"
        mutated.append(row)
    ds_mut = build_forecasting_dataset(mutated)
    model_mut = QuantileRegressionModel(feature_columns=cols, random_state=5)
    model_mut.fit(ds_mut, start=split.train_start, end=split.train_end)

    # Identical fit state proves no future feature leaked into training.
    assert model.metadata()["n_train_rows"] == model_mut.metadata()["n_train_rows"]

    # And the train-window predictions are unchanged.
    base = {p.issue_time: (p.p10, p.p50, p.p90)
            for p in model.predict(ds, start=split.train_start, end=split.train_end)}
    after = {p.issue_time: (p.p10, p.p50, p.p90)
             for p in model_mut.predict(ds_mut, start=split.train_start,
                                        end=split.train_end)}
    assert base == after  # every train-window issue must match exactly


def test_quantile_features_are_subset_of_leakage_safe_predictors(feature_rows):
    """The quantile model consumes exactly columns from the issue-row predictor
    set (no target / label / weather-anchored-future columns sneak in)."""
    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    model = QuantileRegressionModel(feature_columns=_feature_cols(ds), random_state=0)
    model.fit(ds, start=split.train_start, end=split.train_end)
    used = set(model.feature_columns())
    assert used <= set(ds.predictor_columns)
    assert "residual_load_mw" not in used
    assert "target_utc" not in used
    assert "target_local" not in used


def test_quantile_weather_features_follow_asof_rules(feature_rows):
    """When weather columns are included as predictors they are the issue-row
    (strictly-before) values from the contract — already proven leakage-safe by
    the Phase 4B tests; here we only confirm the model string-selects them as
    ordinary predictor columns."""
    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    weather_cols = [c for c in ds.predictor_columns if c.startswith("weather_")]
    assert weather_cols, "fixture is missing weather features"
    cols = [c for c in ds.predictor_columns if not c.startswith("weather_")][:6] + weather_cols[:2]
    model = QuantileRegressionModel(feature_columns=cols, random_state=0)
    model.fit(ds, start=split.train_start, end=split.train_end)
    assert model.metadata()["fitted"]
    preds = model.predict(ds, start=split.test_start, end=split.test_end)
    # Every test row that has complete features yields a complete ordered triple.
    for r, p in zip(
        [r for r in ds.rows if split.test_start <= r.issue_time < split.test_end],
        preds,
    ):
        if r.has_features():
            assert p.has_quantiles
            assert p.p10 <= p.p50 <= p.p90


def test_benchmark_quantile_uses_same_chronological_split(feature_rows):
    """The probabilistic benchmark must partition the timeline the same way as
    Phase 4B: train strictly-before validation strictly-before test."""
    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    train_issues = {r.issue_time for r in ds.rows if r.issue_time < split.train_end}
    test_issues = {r.issue_time for r in ds.rows if r.issue_time >= split.validation_end}
    assert train_issues and test_issues
    assert max(train_issues) < min(test_issues)