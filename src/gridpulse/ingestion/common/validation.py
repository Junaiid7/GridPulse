"""Ingestion-level validation performed at the Bronze boundary.

These checks are deliberately lightweight: they verify that what the source
promised (timestamps, units, non-negative values, required fields) is what we
stored. The full Great Expectations framework and Silver/Gold transformations
are explicitly out of scope for Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Mapping, Sequence

from .models import TimeSeries

_SEVERITY_ERROR = "error"
_SEVERITY_WARNING = "warning"


@dataclass(frozen=True)
class Issue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[Issue, ...] = ()

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> dict[str, int]:
        return {"errors": len(self.errors), "warnings": len(self.warnings)}

    def __bool__(self) -> bool:
        """Truthy when the report is clean (no errors)."""
        return self.ok


def _err(code: str, message: str) -> Issue:
    return Issue(_SEVERITY_ERROR, code, message)


def _warn(code: str, message: str) -> Issue:
    return Issue(_SEVERITY_WARNING, code, message)


def validate_timestamps(series: TimeSeries) -> ValidationReport:
    """Check ordering, duplicates and gaps against the declared resolution."""
    ts = series.timestamps
    if not ts:
        return ValidationReport()

    issues: list[Issue] = []
    for i, (a, b) in enumerate(zip(ts, ts[1:]), start=1):
        if b < a:
            issues.append(_err("timestamp_order", f"non-monotonic index {i}: {a.isoformat()} -> {b.isoformat()}"))
        elif b == a:
            issues.append(_err("timestamp_duplicate", f"duplicate timestamp {a.isoformat()} at index {i}"))

    if series.resolution_minutes:
        res = timedelta(minutes=series.resolution_minutes)
        res_s = res.total_seconds()
        missing = 0
        for a, b in zip(ts, ts[1:]):
            delta = (b - a).total_seconds()
            if a < b and delta > res_s:
                missing += max(0, int(round(delta / res_s)) - 1)
        if missing:
            issues.append(_warn("timestamp_gap", f"{missing} expected interval(s) missing for {series.entity}"))

    return ValidationReport(tuple(issues))


def validate_non_negative(series: TimeSeries, *, reason: str) -> ValidationReport:
    """Flag values that a source contract says cannot be negative."""
    negative = [(p.timestamp, p.value) for p in series.points if p.value < 0]
    if negative:
        first = negative[0][0].isoformat()
        return ValidationReport(
            (_err("negative_value", f"{len(negative)} negative value(s) in {series.entity} (contract: {reason}); first at {first}"),)
        )
    return ValidationReport()


def validate_required_fields(mapping: Mapping[str, object], required: Iterable[str]) -> ValidationReport:
    missing = [
        name
        for name in required
        if mapping.get(name) in (None, "")
    ]
    if missing:
        return ValidationReport((_err("missing_field", f"missing required field(s): {', '.join(sorted(missing))}"),))
    return ValidationReport()


def validate_units(series: TimeSeries, expected: str) -> ValidationReport:
    if series.unit != expected:
        return ValidationReport((_err("unit_mismatch", f"{series.entity}: expected unit '{expected}', got '{series.unit}'"),))
    return ValidationReport()


def aggregate(reports: Sequence[ValidationReport]) -> ValidationReport:
    """Combine multiple reports into one."""
    issues: list[Issue] = []
    for report in reports:
        issues.extend(report.issues)
    return ValidationReport(tuple(issues))