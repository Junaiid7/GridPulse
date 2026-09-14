"""Leakage-safe feature engineering for load forecasting.

Phase 3 implements the core feature builders:

- **asof** — the information-cut guard (``history_before``) used by every
  feature builder to prevent data leakage from future observations.
- **calendar** — deterministic calendar features derived from the target hour's
  Europe/Amsterdam local time (hour, day-of-week, month, is_weekend, etc.).
- **holiday** — a clean ``HolidayCalendar`` interface with a CSV-driven and a
  built-in Netherlands implementation (Gregorian Computus).
- **rolling** — backward-looking rolling statistics (mean, std) over realised
  history, with strict ``as_of`` guard.
- **weather** — weather features from observed Open-Meteo history using the
  ``last_observation_before`` pattern (no forecast-weather API yet).

See ``docs/transform-model.md`` for the full leakage-safety design.
"""
