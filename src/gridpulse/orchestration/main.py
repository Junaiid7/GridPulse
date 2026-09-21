"""CLI entry point module for GridPulse orchestration.

This module provides the main() function for the console script entry point
defined in pyproject.toml:

    gridpulse-orchestrate = "gridpulse.orchestration.main:main"

Run with:
    gridpulse-orchestrate

Or programmatically:
    python -m gridpulse.orchestration.main
"""

from __future__ import annotations

from gridpulse.orchestration.__main__ import main

__all__ = ["main"]
