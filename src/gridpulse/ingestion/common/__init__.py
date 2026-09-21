"""Shared building blocks for data ingestion (HTTP, models, storage, validation)."""

from .errors import (
    AuthError,
    EmptyResponseError,
    HttpError,
    IngestionError,
    MalformedResponseError,
    NoDataError,
    QueryTooLargeError,
    RateLimitError,
    ValidationError,
)
from .errors import (
    ConfigurationError as IngestionConfigurationError,
)
from .http import HttpClient, HttpResponse, mask_url
from .models import (
    DataPoint,
    FetchResult,
    TimeRange,
    TimeSeries,
    chunk_range,
    ensure_utc,
)
from .storage import BronzeWrite, write_bronze
from .validation import (
    Issue,
    ValidationReport,
    aggregate,
    validate_non_negative,
    validate_required_fields,
    validate_timestamps,
    validate_units,
)

__all__ = [
    "AuthError",
    "IngestionConfigurationError",
    "EmptyResponseError",
    "HttpError",
    "IngestionError",
    "MalformedResponseError",
    "NoDataError",
    "QueryTooLargeError",
    "RateLimitError",
    "ValidationError",
    "HttpClient",
    "HttpResponse",
    "mask_url",
    "DataPoint",
    "FetchResult",
    "TimeRange",
    "TimeSeries",
    "chunk_range",
    "ensure_utc",
    "BronzeWrite",
    "write_bronze",
    "Issue",
    "ValidationReport",
    "aggregate",
    "validate_non_negative",
    "validate_required_fields",
    "validate_timestamps",
    "validate_units",
]