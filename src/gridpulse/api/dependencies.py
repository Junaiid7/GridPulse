"""Thread-safe model cache and configured-path helpers (Phase 6B).

Callers never supply filesystem paths. Model bundles are resolved as
``<model_dir>/<safe_name>`` and reports as ``<reports_dir>/<safe_name>.json``.
Path-traversal attempts are rejected before any disk access.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from gridpulse.forecast.models import QuantileRegressionModel
from gridpulse.forecast.persistence import load_model

MODEL_NAME_REGEX = re.compile(r"^[A-Za-z0-9_-]+$")
REPORT_NAME_REGEX = re.compile(r"^[A-Za-z0-9_-]+$")

DEFAULT_REPORT_NAME = "orchestration_phase5"


class PathTraversalError(ValueError):
    """Raised when a requested name would resolve outside the configured directory."""


def _require_safe_name(name: str, pattern: re.Pattern[str], *, kind: str) -> str:
    if not isinstance(name, str) or not pattern.fullmatch(name):
        raise ValueError(
            f"Invalid {kind} name {name!r}; must match {pattern.pattern} "
            "(no path separators, dots, or parent references)"
        )
    return name


def resolve_inside(base: Path, name: str, *, kind: str) -> Path:
    """Join ``name`` onto ``base`` and reject any result outside ``base``.

    ``name`` must already be a single safe identifier (no slashes). The
    resolved path is required to be a child of the resolved base directory.
    """
    base_resolved = Path(base).resolve()
    candidate = (base_resolved / name).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise PathTraversalError(
            f"{kind} name {name!r} resolves outside the configured directory"
        ) from exc
    if candidate == base_resolved:
        raise PathTraversalError(
            f"{kind} name {name!r} must name a child path, not the directory itself"
        )
    return candidate


class ModelCache:
    """Thread-safe in-memory cache of loaded :class:`QuantileRegressionModel` bundles."""

    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)
        self._cache: dict[str, QuantileRegressionModel] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, model_name: str) -> QuantileRegressionModel:
        """Return a cached (or freshly loaded) fitted model for ``model_name``."""
        safe = _require_safe_name(model_name, MODEL_NAME_REGEX, kind="model")
        bundle = resolve_inside(self.model_dir, safe, kind="model")
        key = str(bundle)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self.hits += 1
                return cached
            self.misses += 1
            if not bundle.exists() or not bundle.is_dir():
                raise FileNotFoundError(f"Model bundle not found: {safe}")
            model = load_model(bundle)
            self._cache[key] = model
            return model

    def invalidate(self, model_name: str | None = None) -> None:
        """Drop one cached model, or the entire cache when ``model_name`` is None."""
        with self._lock:
            if model_name is None:
                self._cache.clear()
                return
            safe = _require_safe_name(model_name, MODEL_NAME_REGEX, kind="model")
            bundle = resolve_inside(self.model_dir, safe, kind="model")
            self._cache.pop(str(bundle), None)


def load_orchestration_report(
    reports_dir: Path | str, report_name: str = DEFAULT_REPORT_NAME
) -> dict:
    """Load a JSON orchestration report from the configured reports directory."""
    safe = _require_safe_name(report_name, REPORT_NAME_REGEX, kind="report")
    path = resolve_inside(Path(reports_dir), f"{safe}.json", kind="report")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Orchestration report not found: {safe}")
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid orchestration report JSON at {safe}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Orchestration report {safe} must be a JSON object")
    return payload


__all__ = [
    "DEFAULT_REPORT_NAME",
    "MODEL_NAME_REGEX",
    "ModelCache",
    "PathTraversalError",
    "REPORT_NAME_REGEX",
    "load_orchestration_report",
    "resolve_inside",
]
