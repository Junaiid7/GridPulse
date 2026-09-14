# GridPulse Transformation & Feature-Engineering Design

Phase 3 of the GridPulse pipeline consumes the UTC-aware
`TimeSeries` produced by the Bronze ingestion layer and produces
Silver, Gold and feature tables ready for forecasting.

## Timezone / DST Policy

**Canonical timestamps are aware UTC.** Every stored timestamp is UTC.
Local market time is `Europe/Amsterdam` (CET/UTC+1 in winter,
CEST/UTC+2 in summer). Local values are *derived* from UTC, never
stored as the canonical clock.

Because the mapping is always UTC → local, each UTC instant maps to
exactly one local time — no ambiguous local timestamps can be created.
DST transitions manifest as short/long local days (23 h on the
spring-forward day, 25 h on the fall-back day) and as a per-timestamp
offset that the ISO-8601 `+01:00` / `+02:00` suffix records explicitly.

The `transformation.times` module provides the central helpers:
`AMSTERDAM_TZ`, `local_time()`, `local_date()`, `local_hour()`,
`utc_offset_minutes()`.

## Medallion Layout

### Bronze (ingestion layer)

Raw, immutable inputs as ingested from ENTSO-E and Open-Meteo.
Stored as raw XML/JSON payloads in `data/bronze/`.

### Silver (transformation layer)

Cleaned, conformed source data — one CSV per (source, entity):

- Timestamps forced to aware UTC and sorted
- Duplicate timestamps resolved (keep first, rest flagged `duplicate_dropped`)
- Missing timestamps NOT filled — they stay absent and reported as a warning
- Unit, source and entity carried through verbatim
- Each record gets its deterministic Europe/Amsterdam `timestamp_local`
- Sidecar `.meta.json` records provenance, schema version, and cleaning summary

Path convention: `data/silver/<source>/<entity>/<entity>.csv`

### Gold (transformation layer)

Curated hourly Dutch analytical dataset (`data/silver/nl/hourly.csv`
and `data/gold/nl/hourly.csv`):

- Aligned on the union of UTC hour timestamps from all inputs
- Calendar fields derived in Europe/Amsterdam time
- Residual load computed under the documented missing-data policy
- Per-row provenance: `flags` + `sources`

### Residual Load Definition

```
residual_load = load − wind − solar
```

Production-type (PSR) aggregation (consistent with ENTSO-E conventions):

| Category | PSR codes | Description |
|----------|-----------|-------------|
| Wind | B18 + B19 | wind-offshore + wind-onshore |
| Solar | B16 | solar |

Missing-data policy:
- If `load` is missing → residual is `None` (flagged `load_missing`)
- If `wind` or `solar` is missing → residual computed as if zero, but
  the missing term is flagged (`wind_missing` / `solar_missing`)
- Negative residuals are VALID and never clamped

## Feature Engineering — Leakage Safety

### The Information Cut: `as_of`

Every feature is built relative to a forecast issue time called
`as_of`. This is the time at which the forecast is issued (24 h
ahead of the target hour). The strict rule:

> **For a forecast issued at `as_of`, predictive features may only
> use observations with `timestamp_utc < as_of`.**

Everything below funnels history through `history_before()`, so a
future row — including one at the exact cutoff — can never leak into
a lag or rolling feature.

`target_utc` (the forecast hour) is always after `as_of`; calendar
features of the target hour are derived from the calendar alone and
use no data.

### `history_before()`

The single guard used by every feature builder:

```python
history_before(history, as_of) → list[HistoryPoint]
```

Returns realised observations strictly before `as_of` (UTC). Raises
`ValueError` when `as_of` is naive — a leaky cutoff must never be
silently misinterpreted.

### Calendar Features

Computed from the calendar, never from data:

| Feature | Source | Nullable |
|---------|--------|----------|
| `hour` | Europe/Amsterdam local hour (0–23) | No |
| `day_of_week` | Monday=0 … Sunday=6 | No |
| `day_of_year` | 1–366 | No |
| `month` | 1–12 | No |
| `is_weekend` | Saturday/Sunday | No |
| `is_holiday` | Configured NL public holiday | Optional |

### Holiday Calendar Interface

The `HolidayCalendar` protocol defines:
- `is_holiday(day) → bool`
- `holidays_between(start, end) → Set[date]`
- `as_set() → Set[date]`

Two implementations:
- `CsvHolidayCalendar` — reads ISO dates from a CSV file (data-driven)
- `NetherlandsHolidayCalendar` — minimal built-in for standard Dutch
  public holidays (Gregorian Computus approximation)

### Rolling Features

Backward-looking rolling statistics over realised history. All
windows are computed from `history_before()` and cover
`[as_of - window_hours, as_of)` — they never include the cutoff row.

| Function | Description | Returns `None` when |
|----------|-------------|---------------------|
| `rolling_mean` | Mean of trailing window | Window empty |
| `rolling_std` | Sample std (ddof=1) of trailing window | Window has <2 values |
| `rolling_values` | Raw values in trailing window | (always returns list) |

### Weather Features

Phase 3 stores *observed* Open-Meteo history only (no
forecast-weather API is ingested yet). For a given cutoff the usable
evidence is the most recent observation *strictly before* the cutoff.

| Function | Description |
|----------|-------------|
| `last_observation_before` | Value of most recent observation before `as_of` |
| `hourly_nl` | Aggregate per-location series onto one NL hourly series |
| `weather_features` | Per variable: most recent observed value before `as_of` |

Location aggregation uses the equal-weight strategy (deterministic,
not a spatial optimum): locations are first downsampled to UTC hours,
then combined by equal-weight mean.

## Aggregation

### `to_hourly()`

Downsample any UTC series to a grid of UTC hours by averaging the
minutes present inside each hour. Only present minutes are averaged;
an hour with *no* observations stays absent (missing is preserved,
never invented). Partial hours are reported in metadata.

### `aggregate_locations()`

Equal-weight mean of per-location series aligned on shared
timestamps. For each timestamp present in at least one location the
value is the mean over the locations that have an observation that
hour; locations without an observation are skipped (their absence is
reported in metadata, not invented as zero).

## CSV Storage

Standard-library CSV was chosen deliberately over pandas/pyarrow/Parquet:

- The hourly NL dataset is small (~9 k rows/year per entity)
- The project dependency policy forbids heavy tabular dependencies
  "merely because they are common"
- Parquet/pyarrow can replace this writer in the future warehouse
  phase without changing the public API

Paths are deterministic (fixed basenames — no run-timestamp in
filenames). A sidecar `.meta.json` records provenance, schema version
and row counts.

## Provenance

Every Silver/Gold output carries a sidecar `.meta.json` containing:
- Tier (`silver` / `gold`)
- Source and entity
- Unit
- Timezone policy: `canonical_utc; local derived as Europe/Amsterdam`
- Schema version (`1.0`)
- Generation timestamp
- Notes

## Schema

Both Silver and Gold schemas are defined in `transformation.schema`
as typed `Column` dataclasses with: name, meaning, unit, timezone,
nullable, source, dtype. The lists live in one place so CSV writers,
sidecar metadata and this document cannot drift apart.

### Silver Columns

`timestamp_utc`, `timestamp_local`, `value`, `unit`, `source`,
`entity`, `flags`

### Gold Columns

`timestamp_utc`, `timestamp_local`, `load_mw`, `wind_mw`, `solar_mw`,
`residual_load_mw`, `day_ahead_price_eur_mwh`, `hour`, `day_of_week`,
`day_of_year`, `month`, `is_weekend`, `is_holiday`, `flags`, `sources`
