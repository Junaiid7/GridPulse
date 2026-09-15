"""Shared synthetic-pipeline fixture for the forecast tests.

Runs the REAL GridPulse pipeline (Bronze->Silver->Gold->Features) ONCE per
session on synthetic fixtures, so contract / leakage / benchmark tests reuse
one feature table rather than re-running the pipeline per test/module.
"""

from __future__ import annotations

import pytest

from support.synthetic_electricity import features_rows, run_synthetic_pipeline


@pytest.fixture(scope="session")
def synthetic_pipeline(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("synthetic-pipeline")
    return run_synthetic_pipeline(data_dir)


@pytest.fixture(scope="session")
def feature_rows(synthetic_pipeline):
    return features_rows(synthetic_pipeline)