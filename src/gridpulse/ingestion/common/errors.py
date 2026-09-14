"""Exception hierarchy for the ingestion layer."""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion errors."""


class ConfigurationError(IngestionError):
    """Raised when required configuration is missing or invalid."""


class HttpError(IngestionError):
    """A request reached the server but failed with a non-transient HTTP status."""

    def __init__(self, message: str, *, status: int | None = None, body: bytes | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


class AuthError(HttpError):
    """HTTP 401/403: authentication or authorisation failed. Never retried."""

    def __init__(self, message: str, *, status: int, body: bytes | None = None):
        super().__init__(message, status=status, body=body)


class RateLimitError(HttpError):
    """HTTP 429: provider throttling. The caller should back off and retry later."""

    def __init__(self, message: str, *, status: int = 429, retry_after: float | None = None, body: bytes | None = None):
        super().__init__(message, status=status, body=body)
        self.retry_after = retry_after


class QueryTooLargeError(HttpError):
    """The provider rejected the request because it exceeds the element limit.

    The requested period should be split into smaller pieces (see the client's
    automatic halving fallback)."""


class EmptyResponseError(IngestionError):
    """The response body was empty or contained no usable content."""


class NoDataError(IngestionError):
    """The provider reported no data for the requested period (e.g. "No
    matching data found")."""


class MalformedResponseError(IngestionError):
    """The response could not be parsed into the expected structure."""


class ValidationError(IngestionError):
    """Ingestion-level validation failed (see the attached report).

    Carries the offending :class:`~gridpulse.ingestion.common.validation.ValidationReport`.
    """

    def __init__(self, report: object, message: str = "ingestion validation failed"):
        super().__init__(message)
        self.report = report