"""Weather features from observed history (as-of safe).

Phase 3 stores *observed* Open-Meteo history only (no forecast-weather API is
ingested yet). For a given cutoff the usable evidence is therefore the most
recent observation *strictly before* the cutoff. This module never peers
ahead: a future observation — including one at the cutoff — is not used.

True numerical-weather-prediction features are a documented future extension
(see ``docs/transform-model.md``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Mapping, Optional

from ..ingestion.common.models import TimeSeries
from ..transformation.aggregation import aggregate_locations, to_hourly
from .asof import history_before


def last_observation_before(series: TimeSeries, as_of: datetime) -> Optional[float]:
    """Value of the most recent observation strictly before ``as_of``."""
    points = history_before(((p.timestamp, p.value) for p in series.points), as_of)
    return points[-1].value if points else None


def hourly_nl(by_location: Mapping[str, TimeSeries]) -> TimeSeries:
    """Aggregate per-location series (equal weight) onto one NL hourly series.

    Location series are first downsampled to UTC hours, then combined by the
    documented equal-weight strategy (deterministic, not a spatial optimum).
    """
    hourly = {name: to_hourly(s) for name, s in by_location.items()}
    return aggregate_locations(hourly)


def weather_features(
    by_variable: Mapping[str, TimeSeries],
    as_of: datetime,
) -> dict[str, Optional[float]]:
    """Per variable: most recent observed value strictly before ``as_of``."""
    return {var: last_observation_before(series, as_of) for var, series in by_variable.items()}


__all__ = ["last_observation_before", "hourly_nl", "weather_features"]