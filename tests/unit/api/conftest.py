"""Shared synthetic-pipeline fixture for API tests."""

from __future__ import annotations

import pytest
from support.synthetic_electricity import features_rows, run_synthetic_pipeline


@pytest.fixture(scope="session")
def synthetic_pipeline(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("synthetic-pipeline-api")
    return run_synthetic_pipeline(data_dir)


@pytest.fixture(scope="session")
def feature_rows(synthetic_pipeline):
    return features_rows(synthetic_pipeline)
