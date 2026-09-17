"""Shared fixtures for the optimization tests (Phase 4D-A + 4D-B).

Runs the REAL GridPulse pipeline ONCE per session on synthetic fixtures so the
Phase 4D-A dispatch tests and the Phase 4D-B backtest tests share one feature
table and one gold (realised residual + price) table instead of re-running the
pipeline per test/module.
"""

from __future__ import annotations

import pytest

from support.backtest_fixture import gold_by_timestamp, gold_hour_rows
from support.synthetic_electricity import features_rows, run_synthetic_pipeline


@pytest.fixture(scope="session")
def synthetic_pipeline(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("synthetic-pipeline-opt")
    return run_synthetic_pipeline(data_dir)


@pytest.fixture(scope="session")
def feature_rows(synthetic_pipeline):
    return features_rows(synthetic_pipeline)


@pytest.fixture(scope="session")
def gold_rows(synthetic_pipeline):
    return gold_hour_rows(synthetic_pipeline)


@pytest.fixture(scope="session")
def gold_by_ts(gold_rows):
    return gold_by_timestamp(gold_rows)