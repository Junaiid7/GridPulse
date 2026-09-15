"""Data-quality reporting for GridPulse pipeline runs (Phase 4A).

The :mod:`~gridpulse.reporting.dq_report` module turns a completed pipeline run
into a reproducible, human-readable report covering source / coverage / values /
time / energy / weather / imbalance / provenance sections.
"""

from .dq_report import DataQualityReport

__all__ = ["DataQualityReport"]
