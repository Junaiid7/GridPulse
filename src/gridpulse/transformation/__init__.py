"""Bronze → Silver → Gold transformation and feature-engineering foundation.

Phase 3. Consumes the UTC-aware :class:`~gridpulse.ingestion.common.models.TimeSeries`
produced by the ingestion layer and produces:

- **Silver**: cleaned, conformed, deterministic CSV tables per (source, entity);
- **Gold**: an hourly Dutch analytical dataset (incl. residual load) ready for
  forecasting features;
- **features**: a leakage-safe feature pipeline for future forecasting models.

Storage is standard-library CSV (no heavy tabular dependencies yet); canonical
timestamps are UTC, with Europe/Amsterdam local fields derived deterministically.
See ``docs/transform-model.md`` for the full design.
"""