"""FastAPI serving layer for GridPulse (Phase 6B).

Exposes four endpoints:

* ``GET  /health`` — liveness plus configured-directory status.
* ``POST /forecast/predict`` — P10/P50/P90 inference from a persisted model.
* ``POST /optimization/dispatch`` — battery dispatch for a 24-hour horizon.
* ``GET  /orchestration/latest`` — last written orchestration JSON report.

Callers never supply filesystem paths. Model bundles are loaded from a
configured directory by a safe identifier; reports likewise. The factory
:func:`create_app` is the public construction point so tests can inject
temporary directories without mutating process-global state.
"""

from __future__ import annotations

import hmac
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gridpulse import __version__
from gridpulse.config import PROJECT_ROOT, get_settings
from gridpulse.forecast.contract import ForecastingDataset, ForecastRow
from gridpulse.forecast.probabilistic import risk_score
from gridpulse.optimization.battery import BatteryConfig
from gridpulse.optimization.contract import DispatchInfeasible, DispatchInput
from gridpulse.optimization.strategies import run_dispatch

from .dependencies import (
    DEFAULT_REPORT_NAME,
    ModelCache,
    PathTraversalError,
    load_orchestration_report,
)
from .schemas import (
    ForecastPredictRequest,
    ForecastPredictResponse,
    ForecastRowOutput,
    HealthResponse,
    OptimizationDispatchRequest,
    OptimizationDispatchResponse,
    OrchestrationLatestResponse,
)

LOGGER = logging.getLogger("gridpulse.api")

DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "data" / "reports"


def _http_error(status_code: int, error: str, detail: str) -> HTTPException:
    return HTTPException(
        status_code=status_code, detail={"error": error, "detail": detail}
    )


def _validate_feature_schema(
    requested: dict[str, float | None], expected: list[str]
) -> None:
    """Reject extra or missing feature keys relative to the persisted model schema."""
    requested_keys = set(requested)
    expected_keys = set(expected)
    extra = sorted(requested_keys - expected_keys)
    missing = sorted(expected_keys - requested_keys)
    if extra or missing:
        parts = []
        if extra:
            parts.append(f"unexpected feature columns: {extra}")
        if missing:
            parts.append(f"missing feature columns: {missing}")
        raise _http_error(422, "feature_schema_mismatch", "; ".join(parts))


def _rows_to_dataset(
    request: ForecastPredictRequest, feature_columns: list[str]
) -> ForecastingDataset:
    rows: list[ForecastRow] = []
    for item in request.rows:
        _validate_feature_schema(item.features, feature_columns)
        n_incomplete = sum(1 for c in feature_columns if item.features.get(c) is None)
        rows.append(
            ForecastRow(
                issue_time=item.issue_time,
                target_time=item.target_time,
                features=dict(item.features),
                n_incomplete=n_incomplete,
            )
        )
    return ForecastingDataset(
        rows=tuple(rows), predictor_columns=tuple(feature_columns)
    )


def _prediction_window(dataset: ForecastingDataset) -> tuple[datetime, datetime]:
    """Half-open ``[start, end)`` covering every submitted issue time."""
    issue_times = [row.issue_time for row in dataset.rows]
    start = min(issue_times)
    end = max(issue_times) + timedelta(microseconds=1)
    if start >= end:  # pragma: no cover - guarded by min_length=1
        raise _http_error(422, "invalid_window", "prediction window is empty")
    return start, end


def _battery_from_schema(schema) -> BatteryConfig:
    if schema is None:
        return BatteryConfig()
    kwargs = {k: v for k, v in schema.model_dump().items() if v is not None}
    return BatteryConfig(**kwargs)


def create_app(
    *,
    model_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    settings=None,
) -> FastAPI:
    """Application factory with environment-driven security defaults."""
    settings = settings or get_settings()
    resolved_model_dir = (
        Path(model_dir)
        if model_dir is not None
        else Path(settings.data_root) / "models"
    )
    resolved_reports_dir = (
        Path(reports_dir)
        if reports_dir is not None
        else Path(settings.data_root) / "reports"
    )
    cache = ModelCache(resolved_model_dir)

    app = FastAPI(
        title="GridPulse API",
        version=__version__,
        description="Probabilistic residual-load forecasts and battery dispatch.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.state.model_dir = resolved_model_dir
    app.state.reports_dir = resolved_reports_dir
    app.state.model_cache = cache
    app.state.settings = settings

    @app.middleware("http")
    async def _correlation_id_middleware(request: Request, call_next):
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        LOGGER.info(
            "HTTP request started method=%s path=%s correlation_id=%s",
            request.method,
            request.url.path,
            correlation_id,
        )
        start_time = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
        LOGGER.info(
            "HTTP request completed method=%s path=%s status=%d duration_ms=%.2f correlation_id=%s",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            correlation_id,
        )
        return response

    @app.middleware("http")
    async def _api_key_middleware(request: Request, call_next):
        # An unset key preserves local/offline compatibility. Once configured,
        # every non-liveness endpoint is protected unless explicit dev mode is on.
        if settings.api_key and not settings.dev_mode and request.url.path != "/health":
            supplied = request.headers.get("X-API-Key")
            if not supplied or not hmac.compare_digest(supplied, settings.api_key):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "unauthorized",
                        "detail": "A valid X-API-Key header is required",
                    },
                )
        return await call_next(request)

    @app.exception_handler(PathTraversalError)
    async def _traversal_handler(
        _request: Request, exc: PathTraversalError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_name", "detail": str(exc)},
        )

    @app.exception_handler(Exception)
    async def _global_exception_handler(
        _request: Request, exc: Exception
    ) -> JSONResponse:
        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content=exc.detail
                if isinstance(exc.detail, dict)
                else {"detail": exc.detail},
            )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "detail": "An unexpected error occurred",
            },
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            model_dir_configured=resolved_model_dir.exists(),
            reports_dir_configured=resolved_reports_dir.exists(),
            data_status="FIXTURE-VERIFIED",
        )

    @app.post("/forecast/predict", response_model=ForecastPredictResponse)
    def forecast_predict(body: ForecastPredictRequest) -> ForecastPredictResponse:
        try:
            model = cache.get(body.model_name)
        except FileNotFoundError as exc:
            raise _http_error(404, "model_not_found", str(exc)) from exc
        except ValueError as exc:
            raise _http_error(400, "invalid_model_name", str(exc)) from exc

        feature_columns = model.feature_columns()
        if not feature_columns:
            raise _http_error(
                500, "model_schema_missing", "persisted model has no feature_columns"
            )

        dataset = _rows_to_dataset(body, feature_columns)
        start, end = _prediction_window(dataset)
        try:
            forecasts = model.predict(dataset, start=start, end=end)
        except Exception as exc:  # never leak stack traces to the client
            raise _http_error(500, "prediction_failed", str(exc)) from exc

        data_status = getattr(model, "_data_status", None) or "FIXTURE-VERIFIED"

        outputs = [
            ForecastRowOutput(
                issue_time=fc.issue_time.astimezone(UTC).isoformat(),
                target_time=fc.target_time.astimezone(UTC).isoformat(),
                p10=fc.p10,
                p50=fc.p50,
                p90=fc.p90,
                risk_score=risk_score(fc.p10, fc.p50, fc.p90),
            )
            for fc in forecasts
        ]
        return ForecastPredictResponse(
            model_name=body.model_name,
            n_rows=len(outputs),
            forecasts=outputs,
            feature_columns=list(feature_columns),
            data_status=str(data_status),
        )

    @app.post("/optimization/dispatch", response_model=OptimizationDispatchResponse)
    def optimization_dispatch(
        body: OptimizationDispatchRequest,
    ) -> OptimizationDispatchResponse:
        try:
            battery = _battery_from_schema(body.battery)
            inputs = DispatchInput(
                issue_time=body.issue_time,
                target_times=body.target_times,
                price_eur_mwh=body.price_eur_mwh,
                residual_load_mw=body.residual_load_mw,
                scenario_residual_load_mw=body.scenario_residual_load_mw or {},
                battery=battery,
                curtailment_allowed=body.curtailment_allowed,
                terminal_soc=body.terminal_soc,
            )
            result = run_dispatch(inputs, body.strategy)
        except DispatchInfeasible as exc:
            raise _http_error(422, "dispatch_infeasible", str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise _http_error(422, "invalid_dispatch_input", str(exc)) from exc

        payload = result.to_dict()
        return OptimizationDispatchResponse(**payload)

    @app.get("/orchestration/latest", response_model=OrchestrationLatestResponse)
    def orchestration_latest(
        report_name: str = Query(
            default=DEFAULT_REPORT_NAME, pattern=r"^[A-Za-z0-9_-]+$"
        ),
    ) -> OrchestrationLatestResponse:
        try:
            report = load_orchestration_report(resolved_reports_dir, report_name)
        except FileNotFoundError as exc:
            raise _http_error(404, "report_not_found", str(exc)) from exc
        except ValueError as exc:
            raise _http_error(400, "invalid_report", str(exc)) from exc
        data_status = report.get("data_status")
        if data_status is not None:
            data_status = str(data_status)
        return OrchestrationLatestResponse(
            report_name=report_name,
            data_status=data_status,
            report=report,
        )

    return app


# Module-level app for ``uvicorn gridpulse.api.app:app``. Directories are the
# repository defaults; tests should call :func:`create_app` instead.
app = create_app()


__all__ = ["create_app", "app", "DEFAULT_MODEL_DIR", "DEFAULT_REPORTS_DIR"]
