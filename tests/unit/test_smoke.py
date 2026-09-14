"""Smoke test: the package and its subpackages import, and settings resolve."""

from __future__ import annotations

import importlib

import gridpulse

# Public subpackages expected to exist and be importable.
SUBPACKAGES = (
    "ingestion",
    "validation",
    "warehouse",
    "features",
    "models",
    "risk_engine",
    "optimization",
    "simulation",
    "evaluation",
    "api",
    "dashboard",
    "ai_briefing",
)


def test_all_subpackages_import() -> None:
    for name in SUBPACKAGES:
        module = importlib.import_module(f"gridpulse.{name}")
        assert module.__name__ == f"gridpulse.{name}"


def test_version_exposed() -> None:
    assert isinstance(gridpulse.__version__, str)
    assert gridpulse.__version__


def test_settings_resolve_from_root() -> None:
    from gridpulse.config import get_settings

    settings = get_settings()
    assert settings.project_root.exists()
    assert settings.project_root.name == "GridPulse"
    # Tier layout is centralised and stable.
    assert settings.bronze_dir.name == "bronze"
    assert settings.silver_dir.name == "silver"
    assert settings.gold_dir.name == "gold"
    assert settings.log_level in {"INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"}