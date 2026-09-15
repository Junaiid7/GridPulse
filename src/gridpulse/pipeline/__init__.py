"""Pipeline runner: orchestrate Bronze → Silver → Gold → Features → DQ report.

Phase 4A.  The runner pulls real data through the existing ingestion and
transformation layers, writes deterministic artefacts under ``data/``,
builds a feature dataset, and produces a reproducible data-quality report.

Live execution uses real Open-Meteo (no API key required).  ENTSO-E
datasets are fetched only when ``ENTSOE_API_KEY`` is configured; when the
key is absent every ENTSO-E dataset is recorded as *UNAVAILABLE* and the
run still produces weather artefacts plus the report.
"""

from .runner import run_pipeline

__all__ = ["run_pipeline"]
