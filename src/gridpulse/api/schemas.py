"""Pydantic request and response schemas for the GridPulse FastAPI layer (Phase 6B).

All request models are strict: extra fields are rejected, model names are
restricted to a safe character class, and no filesystem paths are accepted
from the caller.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gridpulse.optimization.strategies import STRATEGIES

_MODEL_NAME_PATTERN = r"^[A-Za-z0-9_-]+$"


class StrictModel(BaseModel):
    """Base schema: extra fields are a validation error, not silently ignored."""

    model_config = ConfigDict(extra="forbid")


class HealthResponse(StrictModel):
    status: str
    version: str
    model_dir_configured: bool
    reports_dir_configured: bool
    data_status: str = "FIXTURE-VERIFIED"


class ForecastFeatureRow(StrictModel):
    issue_time: datetime
    target_time: datetime
    features: dict[str, float | None]

    @field_validator("features")
    @classmethod
    def _features_not_empty(
        cls, value: dict[str, float | None]
    ) -> dict[str, float | None]:
        if not value:
            raise ValueError("features must not be empty")
        return value


class ForecastPredictRequest(StrictModel):
    model_name: str = Field(
        ..., pattern=_MODEL_NAME_PATTERN, min_length=1, max_length=64
    )
    rows: list[ForecastFeatureRow] = Field(..., min_length=1)

    @field_validator("model_name")
    @classmethod
    def _no_dot_segments(cls, value: str) -> str:
        # Pattern already forbids slashes and dots; keep the check explicit.
        if value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("model_name must be a safe identifier, not a path")
        return value


class ForecastRowOutput(StrictModel):
    issue_time: str
    target_time: str
    p10: float | None = None
    p50: float | None = None
    p90: float | None = None
    risk_score: float | None = None


class ForecastPredictResponse(StrictModel):
    model_name: str
    n_rows: int
    forecasts: list[ForecastRowOutput]
    feature_columns: list[str]
    data_status: str = "FIXTURE-VERIFIED"


class BatteryConfigSchema(StrictModel):
    """Optional battery overrides; fields match :class:`BatteryConfig`."""

    capacity_mwh: float | None = Field(default=None, gt=0)
    max_charge_power_mw: float | None = Field(default=None, gt=0)
    max_discharge_power_mw: float | None = Field(default=None, gt=0)
    round_trip_efficiency: float | None = Field(default=None, gt=0, le=1)
    soc_min: float | None = Field(default=None, ge=0, le=1)
    soc_max: float | None = Field(default=None, ge=0, le=1)
    initial_soc: float | None = Field(default=None, ge=0, le=1)
    charge_efficiency: float | None = Field(default=None, gt=0, le=1)
    discharge_efficiency: float | None = Field(default=None, gt=0, le=1)


class OptimizationDispatchRequest(StrictModel):
    strategy: str
    issue_time: datetime
    target_times: list[datetime] = Field(..., min_length=24, max_length=24)
    price_eur_mwh: list[float] = Field(..., min_length=24, max_length=24)
    residual_load_mw: list[float] = Field(..., min_length=24, max_length=24)
    scenario_residual_load_mw: dict[str, list[float]] | None = None
    battery: BatteryConfigSchema | None = None
    curtailment_allowed: bool = False
    terminal_soc: float | None = Field(default=None, ge=0, le=1)

    @field_validator("strategy")
    @classmethod
    def _known_strategy(cls, value: str) -> str:
        if value not in STRATEGIES:
            raise ValueError(
                f"unknown strategy {value!r}; expected one of {sorted(STRATEGIES)}"
            )
        return value


class OptimizationDispatchResponse(StrictModel):
    strategy: str
    issue_time: str
    target_times: list[str]
    charge_mw: list[float]
    discharge_mw: list[float]
    soc_mwh: list[float]
    grid_demand_mw: list[float]
    residual_load_mw: list[float]
    scenario_residual_load_mw: dict[str, list[float]]
    scenario_costs_eur: dict[str, float]
    simulated_cost_eur: float
    battery: dict[str, Any]
    status: str
    solver: str
    message: str | None = None
    data_status: str = "FIXTURE-VERIFIED"


class OrchestrationLatestResponse(StrictModel):
    report_name: str
    data_status: str | None = None
    report: dict[str, Any]


class ErrorDetail(StrictModel):
    error: str
    detail: str


__all__ = [
    "BatteryConfigSchema",
    "ErrorDetail",
    "ForecastFeatureRow",
    "ForecastPredictRequest",
    "ForecastPredictResponse",
    "ForecastRowOutput",
    "HealthResponse",
    "OptimizationDispatchRequest",
    "OptimizationDispatchResponse",
    "OrchestrationLatestResponse",
]
