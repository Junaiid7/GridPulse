"""Comprehensive tests for the GridPulse FastAPI serving layer (Phase 6B).

Tests all four endpoints with the TestClient: successful requests, validation
errors, missing resources, path-traversal safety, feature-schema mismatches,
dispatch edge cases, and malformed reports.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from gridpulse.forecast.contract import build_forecasting_dataset
from gridpulse.forecast.models import QuantileRegressionModel
from gridpulse.forecast.persistence import save_model
from gridpulse.forecast.split import chronological_split_by_fraction

UTC = UTC


@pytest.fixture
def api_dirs(tmp_path):
    """Temporary model and reports directories for testing."""
    model_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"
    model_dir.mkdir()
    reports_dir.mkdir()
    return {"model_dir": model_dir, "reports_dir": reports_dir}


@pytest.fixture
def fitted_model_bundle(feature_rows, api_dirs):
    """Save a fitted QuantileRegressionModel to the test model directory."""
    ds = build_forecasting_dataset(feature_rows)
    split = chronological_split_by_fraction(ds.issue_times)
    cols = [c for c in ds.predictor_columns if not c.startswith("weather_")][:4]
    model = QuantileRegressionModel(
        feature_columns=cols, random_state=42, n_estimators=10
    )
    model.fit(ds, start=split.train_start, end=split.train_end)
    bundle_path = api_dirs["model_dir"] / "test_model"
    save_model(model, bundle_path, data_status="FIXTURE-VERIFIED")
    return {
        "bundle_path": bundle_path,
        "feature_columns": cols,
        "model_name": "test_model",
    }


@pytest.fixture
def orchestration_report(api_dirs):
    """Write a minimal orchestration report to the test reports directory."""
    report = {
        "data_status": "FIXTURE-VERIFIED",
        "forecast": {"model": "quantile_regression_lgbm"},
        "dispatch": {"strategies": ["lp_p50"]},
    }
    report_path = api_dirs["reports_dir"] / "orchestration_phase5.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh)
    return report


@pytest.fixture
def client(api_dirs):
    """TestClient with temporary directories."""
    from gridpulse.api.app import create_app

    app = create_app(
        model_dir=api_dirs["model_dir"], reports_dir=api_dirs["reports_dir"]
    )
    return TestClient(app)


# =============================================================================
# GET /health
# =============================================================================


def test_health_endpoint_returns_ok(client):
    """GET /health returns 200 with status and version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert data["model_dir_configured"] is True
    assert data["reports_dir_configured"] is True
    assert data["data_status"] == "FIXTURE-VERIFIED"


def test_requests_include_correlation_and_timing_headers(client):
    """Requests echo supplied correlation IDs and expose server timing."""
    response = client.get("/health", headers={"X-Correlation-ID": "trace-123"})
    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "trace-123"
    assert float(response.headers["X-Response-Time-Ms"]) >= 0


# =============================================================================
# POST /forecast/predict — successful requests
# =============================================================================


def test_forecast_predict_successful(client, fitted_model_bundle):
    """POST /forecast/predict with valid request returns P10/P50/P90."""
    cols = fitted_model_bundle["feature_columns"]
    payload = {
        "model_name": "test_model",
        "rows": [
            {
                "issue_time": "2024-01-01T06:00:00Z",
                "target_time": "2024-01-02T06:00:00Z",
                "features": {c: 1.0 for c in cols},
            }
        ],
    }
    response = client.post("/forecast/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "test_model"
    assert data["n_rows"] == 1
    assert len(data["forecasts"]) == 1
    fc = data["forecasts"][0]
    assert fc["issue_time"] == "2024-01-01T06:00:00+00:00"
    assert fc["target_time"] == "2024-01-02T06:00:00+00:00"
    assert fc["p10"] is not None
    assert fc["p50"] is not None
    assert fc["p90"] is not None
    assert fc["risk_score"] is not None
    assert data["feature_columns"] == cols
    assert data["data_status"] == "FIXTURE-VERIFIED"


def test_forecast_predict_multiple_rows(client, fitted_model_bundle):
    """POST /forecast/predict handles multiple rows."""
    cols = fitted_model_bundle["feature_columns"]
    payload = {
        "model_name": "test_model",
        "rows": [
            {
                "issue_time": f"2024-01-{d:02d}T06:00:00Z",
                "target_time": f"2024-01-{d + 1:02d}T06:00:00Z",
                "features": {c: float(d) for c in cols},
            }
            for d in range(1, 4)
        ],
    }
    response = client.post("/forecast/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["n_rows"] == 3
    assert len(data["forecasts"]) == 3


def test_forecast_predict_with_none_features(client, fitted_model_bundle):
    """POST /forecast/predict allows None feature values (incomplete row)."""
    cols = fitted_model_bundle["feature_columns"]
    features = {c: 1.0 for c in cols}
    features[cols[0]] = None  # one missing
    payload = {
        "model_name": "test_model",
        "rows": [
            {
                "issue_time": "2024-01-01T06:00:00Z",
                "target_time": "2024-01-02T06:00:00Z",
                "features": features,
            }
        ],
    }
    response = client.post("/forecast/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    # Model may return None predictions for incomplete rows
    assert len(data["forecasts"]) == 1


# =============================================================================
# POST /forecast/predict — validation errors
# =============================================================================


def test_forecast_predict_model_not_found(client):
    """POST /forecast/predict with non-existent model returns 404."""
    payload = {
        "model_name": "nonexistent",
        "rows": [
            {
                "issue_time": "2024-01-01T06:00:00Z",
                "target_time": "2024-01-02T06:00:00Z",
                "features": {"hour": 6.0},
            }
        ],
    }
    response = client.post("/forecast/predict", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert data["detail"]["error"] == "model_not_found"


def test_forecast_predict_invalid_model_name(client):
    """POST /forecast/predict with path-traversal name returns 400."""
    payload = {
        "model_name": "../etc/passwd",
        "rows": [
            {
                "issue_time": "2024-01-01T06:00:00Z",
                "target_time": "2024-01-02T06:00:00Z",
                "features": {"hour": 6.0},
            }
        ],
    }
    response = client.post("/forecast/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation rejects pattern


def test_forecast_predict_empty_rows(client):
    """POST /forecast/predict with empty rows list returns 422."""
    payload = {"model_name": "test_model", "rows": []}
    response = client.post("/forecast/predict", json=payload)
    assert response.status_code == 422


def test_forecast_predict_empty_features(client):
    """POST /forecast/predict with empty features dict returns 422."""
    payload = {
        "model_name": "test_model",
        "rows": [
            {
                "issue_time": "2024-01-01T06:00:00Z",
                "target_time": "2024-01-02T06:00:00Z",
                "features": {},
            }
        ],
    }
    response = client.post("/forecast/predict", json=payload)
    assert response.status_code == 422


def test_forecast_predict_feature_schema_mismatch_extra(client, fitted_model_bundle):
    """POST /forecast/predict with unexpected feature columns returns 422."""
    cols = fitted_model_bundle["feature_columns"]
    features = {c: 1.0 for c in cols}
    features["unexpected_column"] = 99.0
    payload = {
        "model_name": "test_model",
        "rows": [
            {
                "issue_time": "2024-01-01T06:00:00Z",
                "target_time": "2024-01-02T06:00:00Z",
                "features": features,
            }
        ],
    }
    response = client.post("/forecast/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["error"] == "feature_schema_mismatch"
    assert "unexpected feature columns" in data["detail"]["detail"]


def test_forecast_predict_feature_schema_mismatch_missing(client, fitted_model_bundle):
    """POST /forecast/predict with missing required columns returns 422."""
    cols = fitted_model_bundle["feature_columns"]
    features = {c: 1.0 for c in cols[:-1]}  # drop last
    payload = {
        "model_name": "test_model",
        "rows": [
            {
                "issue_time": "2024-01-01T06:00:00Z",
                "target_time": "2024-01-02T06:00:00Z",
                "features": features,
            }
        ],
    }
    response = client.post("/forecast/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["error"] == "feature_schema_mismatch"
    assert "missing feature columns" in data["detail"]["detail"]


def test_forecast_predict_extra_fields_rejected(client, fitted_model_bundle):
    """POST /forecast/predict with extra top-level fields returns 422."""
    cols = fitted_model_bundle["feature_columns"]
    payload = {
        "model_name": "test_model",
        "rows": [
            {
                "issue_time": "2024-01-01T06:00:00Z",
                "target_time": "2024-01-02T06:00:00Z",
                "features": {c: 1.0 for c in cols},
            }
        ],
        "extra_field": "not_allowed",
    }
    response = client.post("/forecast/predict", json=payload)
    assert response.status_code == 422


# =============================================================================
# POST /optimization/dispatch — successful requests
# =============================================================================


def test_optimization_dispatch_lp_p50_success(client):
    """POST /optimization/dispatch with lp_p50 strategy returns a valid result."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    payload = {
        "strategy": "lp_p50",
        "issue_time": base.isoformat(),
        "target_times": [(base + timedelta(hours=h)).isoformat() for h in range(24)],
        "price_eur_mwh": [50.0] * 24,
        "residual_load_mw": [100.0] * 24,
        "battery": None,
        "curtailment_allowed": False,
    }
    response = client.post("/optimization/dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "lp_p50"
    assert len(data["charge_mw"]) == 24
    assert len(data["discharge_mw"]) == 24
    assert len(data["soc_mwh"]) == 24
    assert "simulated_cost_eur" in data
    assert data["status"] in ("optimal", "ok")
    assert data["data_status"] == "FIXTURE-VERIFIED"


def test_optimization_dispatch_scenario_lp_success(client):
    """POST /optimization/dispatch with scenario_lp and P10/P50/P90 succeeds."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    payload = {
        "strategy": "scenario_lp",
        "issue_time": base.isoformat(),
        "target_times": [(base + timedelta(hours=h)).isoformat() for h in range(24)],
        "price_eur_mwh": [50.0] * 24,
        "residual_load_mw": [100.0] * 24,
        "scenario_residual_load_mw": {
            "p10": [90.0] * 24,
            "p50": [100.0] * 24,
            "p90": [110.0] * 24,
        },
        "curtailment_allowed": False,
    }
    response = client.post("/optimization/dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "scenario_lp"
    assert "scenario_costs_eur" in data


def test_optimization_dispatch_custom_battery(client):
    """POST /optimization/dispatch with custom battery config succeeds."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    payload = {
        "strategy": "lp_p50",
        "issue_time": base.isoformat(),
        "target_times": [(base + timedelta(hours=h)).isoformat() for h in range(24)],
        "price_eur_mwh": [50.0] * 24,
        "residual_load_mw": [100.0] * 24,
        "battery": {
            "capacity_mwh": 100.0,
            "max_charge_power_mw": 20.0,
            "max_discharge_power_mw": 20.0,
            "initial_soc": 0.5,
        },
    }
    response = client.post("/optimization/dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["battery"]["capacity_mwh"] == 100.0


def test_optimization_dispatch_no_battery_strategy(client):
    """POST /optimization/dispatch with no_battery strategy succeeds."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    payload = {
        "strategy": "no_battery",
        "issue_time": base.isoformat(),
        "target_times": [(base + timedelta(hours=h)).isoformat() for h in range(24)],
        "price_eur_mwh": [50.0] * 24,
        "residual_load_mw": [100.0] * 24,
    }
    response = client.post("/optimization/dispatch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["strategy"] == "no_battery"
    assert all(ch == 0.0 for ch in data["charge_mw"])
    assert all(dis == 0.0 for dis in data["discharge_mw"])


# =============================================================================
# POST /optimization/dispatch — validation errors
# =============================================================================


def test_optimization_dispatch_unknown_strategy(client):
    """POST /optimization/dispatch with unknown strategy returns 422."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    payload = {
        "strategy": "unknown_strategy",
        "issue_time": base.isoformat(),
        "target_times": [(base + timedelta(hours=h)).isoformat() for h in range(24)],
        "price_eur_mwh": [50.0] * 24,
        "residual_load_mw": [100.0] * 24,
    }
    response = client.post("/optimization/dispatch", json=payload)
    assert response.status_code == 422


def test_optimization_dispatch_wrong_horizon_length(client):
    """POST /optimization/dispatch with non-24 target_times returns 422."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    payload = {
        "strategy": "lp_p50",
        "issue_time": base.isoformat(),
        "target_times": [(base + timedelta(hours=h)).isoformat() for h in range(23)],
        "price_eur_mwh": [50.0] * 23,
        "residual_load_mw": [100.0] * 23,
    }
    response = client.post("/optimization/dispatch", json=payload)
    assert response.status_code == 422


def test_optimization_dispatch_negative_price(client):
    """POST /optimization/dispatch with negative price returns 422."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    prices = [50.0] * 24
    prices[10] = -10.0
    payload = {
        "strategy": "lp_p50",
        "issue_time": base.isoformat(),
        "target_times": [(base + timedelta(hours=h)).isoformat() for h in range(24)],
        "price_eur_mwh": prices,
        "residual_load_mw": [100.0] * 24,
    }
    response = client.post("/optimization/dispatch", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["error"] == "invalid_dispatch_input"


def test_optimization_dispatch_scenario_lp_without_scenarios(client):
    """POST /optimization/dispatch scenario_lp without scenarios returns 422."""
    base = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    payload = {
        "strategy": "scenario_lp",
        "issue_time": base.isoformat(),
        "target_times": [(base + timedelta(hours=h)).isoformat() for h in range(24)],
        "price_eur_mwh": [50.0] * 24,
        "residual_load_mw": [100.0] * 24,
    }
    response = client.post("/optimization/dispatch", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["detail"]["error"] == "dispatch_infeasible"


# =============================================================================
# GET /orchestration/latest — successful requests
# =============================================================================


def test_orchestration_latest_default_report(client, orchestration_report):
    """GET /orchestration/latest returns the default orchestration_phase5 report."""
    response = client.get("/orchestration/latest")
    assert response.status_code == 200
    data = response.json()
    assert data["report_name"] == "orchestration_phase5"
    assert data["data_status"] == "FIXTURE-VERIFIED"
    assert "report" in data
    assert data["report"]["data_status"] == "FIXTURE-VERIFIED"


def test_orchestration_latest_custom_report_name(client, api_dirs):
    """GET /orchestration/latest with custom report_name query param."""
    custom_report = {"custom": "data", "data_status": "test"}
    report_path = api_dirs["reports_dir"] / "custom_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(custom_report, fh)
    response = client.get("/orchestration/latest?report_name=custom_report")
    assert response.status_code == 200
    data = response.json()
    assert data["report_name"] == "custom_report"
    assert data["report"]["custom"] == "data"


# =============================================================================
# GET /orchestration/latest — validation errors
# =============================================================================


def test_orchestration_latest_report_not_found(client):
    """GET /orchestration/latest with non-existent report returns 404."""
    response = client.get("/orchestration/latest?report_name=nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert data["detail"]["error"] == "report_not_found"


def test_orchestration_latest_invalid_report_name(client):
    """GET /orchestration/latest with path-traversal name returns 422."""
    response = client.get("/orchestration/latest?report_name=../secret")
    assert response.status_code == 422  # Query validation rejects pattern


def test_orchestration_latest_malformed_json(client, api_dirs):
    """GET /orchestration/latest with malformed JSON returns 400."""
    bad_path = api_dirs["reports_dir"] / "malformed.json"
    with open(bad_path, "w", encoding="utf-8") as fh:
        fh.write("not-json{")
    response = client.get("/orchestration/latest?report_name=malformed")
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "invalid_report"


def test_orchestration_latest_not_a_dict(client, api_dirs):
    """GET /orchestration/latest with non-dict JSON returns 400."""
    list_path = api_dirs["reports_dir"] / "list_report.json"
    with open(list_path, "w", encoding="utf-8") as fh:
        json.dump([1, 2, 3], fh)
    response = client.get("/orchestration/latest?report_name=list_report")
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "invalid_report"


# =============================================================================
# Path-traversal safety
# =============================================================================


def test_forecast_predict_path_traversal_rejected(client):
    """Model names with path separators are rejected by Pydantic."""
    payload = {
        "model_name": "../../etc/passwd",
        "rows": [
            {
                "issue_time": "2024-01-01T06:00:00Z",
                "target_time": "2024-01-02T06:00:00Z",
                "features": {"hour": 6.0},
            }
        ],
    }
    response = client.post("/forecast/predict", json=payload)
    assert response.status_code == 422  # Pattern validation


def test_orchestration_latest_path_traversal_rejected(client):
    """Report names with path separators are rejected by query pattern."""
    response = client.get("/orchestration/latest?report_name=../etc/passwd")
    assert response.status_code == 422


# =============================================================================
# API Security & Reliability (Phase 7B)
# =============================================================================


def test_api_key_auth_required_when_configured(api_dirs, orchestration_report):
    """Endpoints require X-API-Key when api_key is configured and dev_mode is False."""
    from gridpulse.api.app import create_app
    from gridpulse.config import PROJECT_ROOT, Settings

    settings = Settings(
        project_root=PROJECT_ROOT,
        data_root=api_dirs["model_dir"].parent,
        api_key="super-secret-key-123",
        dev_mode=False,
    )
    app = create_app(
        model_dir=api_dirs["model_dir"],
        reports_dir=api_dirs["reports_dir"],
        settings=settings,
    )
    secured_client = TestClient(app)

    # Missing header -> 401
    resp = secured_client.get("/orchestration/latest")
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"

    # Invalid header -> 401
    resp = secured_client.get(
        "/orchestration/latest", headers={"X-API-Key": "wrong-key"}
    )
    assert resp.status_code == 401
    assert resp.json()["error"] == "unauthorized"

    # Valid header -> 200
    resp = secured_client.get(
        "/orchestration/latest", headers={"X-API-Key": "super-secret-key-123"}
    )
    assert resp.status_code == 200


def test_api_key_auth_bypassed_for_health(api_dirs):
    """GET /health is public and never requires an API key."""
    from gridpulse.api.app import create_app
    from gridpulse.config import PROJECT_ROOT, Settings

    settings = Settings(
        project_root=PROJECT_ROOT,
        data_root=api_dirs["model_dir"].parent,
        api_key="super-secret-key-123",
        dev_mode=False,
    )
    app = create_app(
        model_dir=api_dirs["model_dir"],
        reports_dir=api_dirs["reports_dir"],
        settings=settings,
    )
    secured_client = TestClient(app)

    resp = secured_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_key_auth_bypassed_in_dev_mode(api_dirs, orchestration_report):
    """When dev_mode is True, API key verification is bypassed."""
    from gridpulse.api.app import create_app
    from gridpulse.config import PROJECT_ROOT, Settings

    settings = Settings(
        project_root=PROJECT_ROOT,
        data_root=api_dirs["model_dir"].parent,
        api_key="super-secret-key-123",
        dev_mode=True,
    )
    app = create_app(
        model_dir=api_dirs["model_dir"],
        reports_dir=api_dirs["reports_dir"],
        settings=settings,
    )
    dev_client = TestClient(app)

    resp = dev_client.get("/orchestration/latest")
    assert resp.status_code == 200


def test_cors_middleware_headers(api_dirs):
    """CORS middleware returns configured origin headers."""
    from gridpulse.api.app import create_app
    from gridpulse.config import PROJECT_ROOT, Settings

    settings = Settings(
        project_root=PROJECT_ROOT,
        data_root=api_dirs["model_dir"].parent,
        cors_origins=["http://localhost:8501", "https://app.gridpulse.nl"],
        dev_mode=True,
    )
    app = create_app(
        model_dir=api_dirs["model_dir"],
        reports_dir=api_dirs["reports_dir"],
        settings=settings,
    )
    cors_client = TestClient(app)

    resp = cors_client.options(
        "/health",
        headers={
            "Origin": "https://app.gridpulse.nl",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.headers.get("access-control-allow-origin") == "https://app.gridpulse.nl"


def test_global_exception_handler(api_dirs):
    """Unhandled exceptions are caught and return a structured 500 without leaking stack traces."""
    from gridpulse.api.app import create_app
    from gridpulse.config import PROJECT_ROOT, Settings

    settings = Settings(
        project_root=PROJECT_ROOT,
        data_root=api_dirs["model_dir"].parent,
        dev_mode=True,
    )
    app = create_app(
        model_dir=api_dirs["model_dir"],
        reports_dir=api_dirs["reports_dir"],
        settings=settings,
    )

    @app.get("/trigger-unhandled-error")
    def trigger_error():
        raise RuntimeError(
            "database connection crashed with sensitive connection string"
        )

    unhandled_client = TestClient(app, raise_server_exceptions=False)
    resp = unhandled_client.get("/trigger-unhandled-error")
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"] == "internal_server_error"
    assert "database connection crashed" not in data["detail"]
    assert data["detail"] == "An unexpected error occurred"


# =============================================================================
# Performance & Observability (Phase 7C)
# =============================================================================


def test_correlation_id_and_response_time_headers(client):
    """Every response includes X-Correlation-ID and X-Response-Time-Ms headers."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "X-Correlation-ID" in response.headers
    assert "X-Response-Time-Ms" in response.headers
    # Verify UUID-like format
    correlation_id = response.headers["X-Correlation-ID"]
    assert len(correlation_id) >= 32


def test_model_cache_efficiency(api_dirs, fitted_model_bundle, client):
    """ModelCache tracks hits and misses; repeated calls should hit the cache."""
    from gridpulse.api.app import create_app

    app = create_app(
        model_dir=api_dirs["model_dir"], reports_dir=api_dirs["reports_dir"]
    )
    cache_client = TestClient(app)

    # First call: miss
    cols = fitted_model_bundle["feature_columns"]
    payload = {
        "model_name": "test_model",
        "rows": [
            {
                "issue_time": "2024-01-01T06:00:00Z",
                "target_time": "2024-01-02T06:00:00Z",
                "features": {c: 1.0 for c in cols},
            }
        ],
    }

    # Access cache via app state
    cache = app.state.model_cache
    initial_misses = cache.misses
    initial_hits = cache.hits

    resp1 = cache_client.post("/forecast/predict", json=payload)
    assert resp1.status_code == 200
    assert cache.misses == initial_misses + 1
    assert cache.hits == initial_hits

    # Second call: hit
    resp2 = cache_client.post("/forecast/predict", json=payload)
    assert resp2.status_code == 200
    assert cache.misses == initial_misses + 1
    assert cache.hits == initial_hits + 1
