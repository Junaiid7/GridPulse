"""A dependency-free HTTP client built on :mod:`urllib`.

Provides timeouts, bounded retries with backoff for transient failures,
Retry-After handling, HTTP status classification into the ingestion error
hierarchy, and URL sanitisation so API keys never reach the logs.

The client is deliberately thin and injectable: tests supply a fake
``opener`` (an object with ``open(request, timeout)``) so no live network is
touched.
"""

from __future__ import annotations

import http.client
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .errors import AuthError, HttpError, RateLimitError

LOGGER = logging.getLogger("gridpulse.ingestion.http")

TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
SENSITIVE_PARAM_NAMES = {"securitytoken", "apikey", "api_key", "key", "token", "x-api-key"}


@dataclass(frozen=True)
class HttpResponse:
    """Normalised HTTP response."""

    status: int
    headers: Mapping[str, str]
    body: bytes
    url: str

    @property
    def text(self) -> str:
        content_type = (self.headers.get("content-type") or "").lower()
        encoding = "utf-8"
        if "charset=" in content_type:
            candidate = content_type.split("charset=", 1)[-1].strip().strip('"').split(";")[0]
            if candidate:
                encoding = candidate
        return self.body.decode(encoding, errors="replace")


def mask_url(url: str) -> str:
    """Redact sensitive query parameters for logging and persisted metadata."""
    parsed = urllib.parse.urlsplit(url)
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    masked = [
        (k, "<redacted>" if k.lower() in SENSITIVE_PARAM_NAMES else v)
        for k, v in pairs
    ]
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(masked)))


def _headers_to_map(headers: Any) -> dict[str, str]:
    try:
        return {str(k).lower(): str(v) for k, v in headers.items()}
    except AttributeError:
        return dict(headers or {})


def _retry_after(headers: Any) -> float | None:
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        return min(float(raw), 120.0)
    except (TypeError, ValueError):
        return None


class HttpClient:
    """Robust GET client with retries and error classification."""

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        retries: int = 3,
        backoff: float = 0.5,
        opener: Any = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if retries < 1:
            raise ValueError("retries must be >= 1")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.opener = opener if opener is not None else urllib.request.build_opener()
        self.log = logger or LOGGER

    def get(
        self,
        url: str,
        *,
        params: Optional[Mapping[str, str | int]] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> HttpResponse:
        final_url = url
        if params:
            query = urllib.parse.urlencode(list(params.items()))
            final_url = f"{url}{'&' if '?' in url else '?'}{query}"

        request = urllib.request.Request(final_url, headers=dict(headers or {}))
        request.add_header("User-Agent", "gridpulse/2.0 (GridPulse ingestion)")

        for attempt in range(1, self.retries + 1):
            self.log.debug("GET %s (attempt %d/%d)", mask_url(final_url), attempt, self.retries)
            try:
                raw = self.opener.open(request, timeout=self.timeout)
                status = int(getattr(raw, "status", getattr(raw, "code", 200)))
                headers_map = _headers_to_map(getattr(raw, "headers", {}))
                body = raw.read()
                self.log.debug("GET %s -> %d (%d bytes)", mask_url(final_url), status, len(body))
                return HttpResponse(status, headers_map, body, final_url)
            except urllib.error.HTTPError as exc:
                code = int(exc.code)
                body = exc.read() if exc.fp is not None else b""
                retry_after = _retry_after(getattr(exc, "headers", {}))
                self.log.warning("GET %s -> HTTP %d", mask_url(final_url), code)
                if code in (401, 403):
                    raise AuthError(
                        f"Authentication/authorisation failed (HTTP {code}). "
                        "Check ENTSOE_API_KEY.",
                        status=code,
                        body=body,
                    ) from exc
                if code == 429:
                    retry = RateLimitError("Rate limited (HTTP 429)", status=429, retry_after=retry_after, body=body)
                elif code in TRANSIENT_HTTP_CODES:
                    retry = HttpError(f"Transient HTTP error {code}", status=code, body=body)
                else:
                    raise HttpError(f"HTTP {code}", status=code, body=body) from exc
                if attempt == self.retries:
                    raise retry from exc
                self._sleep_before_retry(attempt, retry_after)
            except (
                urllib.error.URLError,
                socket.timeout,
                TimeoutError,
                ConnectionError,
                http.client.RemoteDisconnected,
                http.client.HTTPException,
            ) as exc:
                self.log.warning("GET %s connection error: %r", mask_url(final_url), exc)
                if attempt == self.retries:
                    raise HttpError(
                        f"Connection failure after {self.retries} attempts: {exc!r}"
                    ) from exc
                self._sleep_before_retry(attempt, None)

        raise HttpError("Unreachable: retry loop exhausted")  # pragma: no cover

    def _sleep_before_retry(self, attempt: int, retry_after: float | None) -> None:
        delay = retry_after if retry_after is not None else self.backoff * (2 ** (attempt - 1))
        self.log.info("Retrying in %.1fs", delay)
        time.sleep(delay)