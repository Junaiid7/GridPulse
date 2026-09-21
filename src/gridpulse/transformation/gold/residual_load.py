"""Residual-load definition and the PSR → category mapping.

``residual_load = load − wind − solar``

Production-type (PSR) aggregation — the exact mapping, consistent with the
repo's own ingestion convention (``ingestion/entsoe/mappings.py`` mirrors
entsoe-py's ``PSRTYPE_MAPPINGS``, and those are the same codes the Phase 2
client sends to the API):

    wind  = B18 (wind-offshore) + B19 (wind-onshore)
    solar = B16 (solar)

The mapping is explicit and parameterisable. It is NOT assumed that every PSR
type is present: a category is summed only over the types that exist for the
hour, and a category that is fully absent is flagged — never silently treated
as zero.

Missing-data policy: if ``load`` is missing the residual is ``None`` (flagged
``load_missing``). If ``wind`` or ``solar`` is missing the residual is still
computed as if that term were zero, BUT the missing term is flagged
(``wind_missing`` / ``solar_missing``) so the value is never taken at face
value. Negative residuals are VALID and never clamped.
"""

from __future__ import annotations

from dataclasses import dataclass

#: PSR codes that make up the "wind" category (wind-offshore + wind-onshore).
WIND_PSR: tuple[str, ...] = ("B18", "B19")
#: PSR codes that make up the "solar" category.
SOLAR_PSR: tuple[str, ...] = ("B16",)

#: Human-readable labels for the codes above (informational only).
PSR_LABELS = {
    "B16": "solar",
    "B18": "wind-offshore",
    "B19": "wind-onshore",
}

_FLAG_LOAD = "load_missing"
_FLAG_WIND = "wind_missing"
_FLAG_SOLAR = "solar_missing"


@dataclass(frozen=True)
class ResidualResult:
    """Outcome of one residual-load computation."""

    residual_mw: float | None
    load_mw: float | None
    wind_mw: float | None
    solar_mw: float | None
    flags: tuple[str, ...] = ()


def compute_residual(
    load_mw: float | None,
    wind_mw: float | None,
    solar_mw: float | None,
) -> ResidualResult:
    """Compute residual load under the documented missing-data policy."""
    if load_mw is None:
        return ResidualResult(None, None, wind_mw, solar_mw, (_FLAG_LOAD,))

    wind = wind_mw if wind_mw is not None else 0.0
    solar = solar_mw if solar_mw is not None else 0.0
    flags: list[str] = []
    if wind_mw is None:
        flags.append(_FLAG_WIND)
    if solar_mw is None:
        flags.append(_FLAG_SOLAR)
    residual = float(load_mw) - float(wind) - float(solar)
    return ResidualResult(residual, float(load_mw), wind_mw, solar_mw, tuple(flags))


def category_hourly(series_by_psr: dict[str, dict], psr_codes: tuple[str, ...]) -> tuple[dict, int]:
    """Sum per-PSR value maps (timestamp → value) into one category map.

    Returns ``(category_map, present_count)`` where ``present_count`` is the
    number of PSR codes with any data. Category values are summed only where a
    timestamp exists in at least one contributing PSR; hours with no
    contribution from any listed PSR are left absent (the caller reports them
    as ``category_missing``).
    """
    category: dict = {}
    present = 0
    for code in psr_codes:
        series = series_by_psr.get(code)
        if not series:
            continue
        present += 1
        for ts, value in series.items():
            category[ts] = category.get(ts, 0.0) + float(value)
    return category, present


__all__ = [
    "WIND_PSR",
    "SOLAR_PSR",
    "PSR_LABELS",
    "ResidualResult",
    "compute_residual",
    "category_hourly",
]