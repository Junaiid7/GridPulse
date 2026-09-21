"""Centralised application configuration.

GridPulse keeps configuration in one place rather than scattering constants
through the codebase. Settings are resolved from environment variables
(see ``.env.example``) with defaults suited to local development, so the
package imports and tests run without any external services.

The data tier layout (bronze / silver / gold) is declared here so every
component writes to the same canonical locations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Repository root: <repo>/src/gridpulse/config.py -> parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Directory storing all local data (raw, processed and analytical tiers).
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

#: Default HTTP behaviour for ingestion clients.
DEFAULT_HTTP_TIMEOUT_S = 30.0
DEFAULT_HTTP_RETRIES = 3


class ConfigurationError(Exception):
    """Raised when required configuration is missing or invalid."""


def _env_log_level() -> str:
    return os.environ.get("GRIDPULSE_LOG_LEVEL", "INFO").strip().upper()


@dataclass(frozen=True)
class Settings:
    """Resolved runtime settings for GridPulse."""

    project_root: Path
    data_root: Path
    log_level: str = "INFO"
    http_timeout_s: float = DEFAULT_HTTP_TIMEOUT_S
    http_retries: int = DEFAULT_HTTP_RETRIES
    #: ENTSO-E API key. Never defaults to a real key and never hardcoded in
    #: code; provided via the ``ENTSOE_API_KEY`` environment variable.
    entsoe_api_key: str | None = None
    #: Optional path to a holidays CSV (see ``features.holiday``). Point
    #: ``GRIDPULSE_HOLIDAYS_CSV`` at authoritative data for production models.
    holidays_csv: Path | None = None
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    api_key: str | None = None
    dev_mode: bool = False
    json_logs: bool = False

    @property
    def bronze_dir(self) -> Path:
        """Raw, immutable inputs as ingested (Bronze)."""
        return self.data_root / "bronze"

    @property
    def silver_dir(self) -> Path:
        """Cleaned, validated, conformed data (Silver)."""
        return self.data_root / "silver"

    @property
    def gold_dir(self) -> Path:
        """Curated analytical tables ready for use (Gold)."""
        return self.data_root / "gold"


def get_settings() -> Settings:
    """Build settings from the environment, falling back to local defaults."""
    cors_raw = os.environ.get("GRIDPULSE_CORS_ORIGINS", "*")
    cors_origins = [o.strip() for o in cors_raw.split(",")] if cors_raw else ["*"]
    return Settings(
        project_root=PROJECT_ROOT,
        data_root=Path(os.environ.get("GRIDPULSE_DATA_DIR", DEFAULT_DATA_DIR)),
        log_level=_env_log_level(),
        http_timeout_s=float(os.environ.get("GRIDPULSE_HTTP_TIMEOUT_S", DEFAULT_HTTP_TIMEOUT_S)),
        http_retries=int(os.environ.get("GRIDPULSE_HTTP_RETRIES", DEFAULT_HTTP_RETRIES)),
        entsoe_api_key=os.environ.get("ENTSOE_API_KEY") or None,
        holidays_csv=_opt_path(os.environ.get("GRIDPULSE_HOLIDAYS_CSV")),
        cors_origins=cors_origins,
        api_key=os.environ.get("GRIDPULSE_API_KEY"),
        dev_mode=os.environ.get("GRIDPULSE_DEV_MODE", "false").lower() == "true",
        json_logs=os.environ.get("GRIDPULSE_JSON_LOGS", "false").lower() == "true",
    )


def _opt_path(raw: str | None) -> Path | None:
    """Return ``Path(raw)`` when set, else ``None`` (skips empty strings)."""
    if not raw:
        return None
    return Path(raw)


def require_entsoe_api_key(settings: Settings) -> str:
    """Return the ENTSO-E API key or raise if it is not configured."""
    if not settings.entsoe_api_key:
        raise ConfigurationError(
            "ENTSOE_API_KEY is not configured. Set it in the environment or "
            ".env before fetching from the ENTSO-E Transparency Platform. "
            "Never hardcode the key in source code."
        )
    return settings.entsoe_api_key


__all__ = [
    "Settings",
    "ConfigurationError",
    "get_settings",
    "require_entsoe_api_key",
    "PROJECT_ROOT",
    "DEFAULT_DATA_DIR",
    "DEFAULT_HTTP_TIMEOUT_S",
    "DEFAULT_HTTP_RETRIES",
]