# Real-Data Run (Phase 4A)

This document records the first **live real-data end-to-end run** of the
GridPulse pipeline (Phase 4A), executed on **2026-09-14** with a **10-day
historical window** `2026-09-04 → 2026-09-14` (UTC, end exclusive).

Everything below was produced by the code path that also runs in the offline
suite (`tests/unit/pipeline/test_runner.py`), so the fixtures and the live run
share one implementation.

## How to reproduce

```bash
# From the repo root (Windows: use .venv\Scripts\python.exe)
.venv/Scripts/python.exe -m gridpulse.pipeline --start 2026-09-04 --end 2026-09-14

# Force a visible failure when ENTSO-E is unavailable (used in CI):
.venv/Scripts/python.exe -m gridpulse.pipeline --start 2026-09-04 --end 2026-09-14 --require-entsoe   # exit 2

# Try to require Gold (also exits non-zero when no load data):
.venv/Scripts/python.exe -m gridpulse.pipeline --start 2026-09-04 --end 2026-09-14 --require-gold     # exit 3
```

CLI flags: `--start`/`--end` (YYYY-MM-DD, end exclusive), `--data-dir`,
`--report-dir`, `--require-entsoe`, `--require-gold`, `--verbose`.
Default window: last 8 days ending yesterday. Defaults read `GRIDPULSE_DATA_DIR`.

## What the run did (honest report)

| Stage | Outcome | Evidence |
|-------|---------|----------|
| Open-Meteo historical weather (6 NL locations × 10 variables) | **60/60 datasets verified live** (HTTP 200, real payloads) | Bronze payloads + manifests below; Silver 240 rows/dataset, 0 errors, 0 warnings |
| ENTSO-E load / generation / day-ahead prices | **0/5 verified — UNAVAILABLE** | No `ENTSOE_API_KEY` configured; clients never constructed; every dataset recorded `unavailable` with reason `ENTSOE_API_KEY not configured` |
| NL imbalance prices (A85) | **UNVERIFIED — no API key** | Cannot instantiate `EntsoeClient`; status is `UNVERIFIED`, never fabricated |
| Gold NL hourly dataset | **NOT PRODUCED** | Requires load data; honest `reason` in report |
| Features (calendar/lags/rolling/weather) | **NOT PRODUCED** | Same blocker as Gold |
| Data-quality report | **Written (JSON + Markdown)** | `data/reports/dq-report_20260904__20260914.{json,md}` |

Data-quality facts from the report:

- **Coverage:** every weather dataset has exactly the expected `240` hourly
  rows (10 days × 24 h), `missing_timestamps = 0`, `duplicate_dropped = 0`,
  `null_values = 0`.
- **Time policy:** all timestamps stored as aware UTC; Europe/Amsterdam local
  derived. Only `+02:00` (CEST) observed in this September window; `0` DST
  transitions (expected — the window does not cross a transition).
- **Values:** e.g. NL-aggregate temperature 10.0–27.1 °C (mean 16.7),
  wind_speed_100m 1.08–13.52 m/s, cloud_cover 0–100 %; every series flagged
  by the correct unit.
- **Provenance:** Bronze manifests carry requested window, retrieval time,
  source URL, identifiers (location/lat/lon), payload SHA-256 and size; a
  per-variable SHA-256 is repeated in the report's PROVENANCE section.

## Artefact tree (live run)

```
data/
  bronze/open-meteo/historical-weather/20260904__20260914/
      payload_*.json          raw Open-Meteo JSON (immutable)
      manifest_*.json         provenance sidecar (sha256, window, url, ids)
  silver/open-meteo/historical-weather-<loc>-<variable>/
      historical-weather-<loc>-<variable>.csv       UTC + Amsterdam local
      historical-weather-<loc>-<variable>.meta.json cleaning summary
  reports/dq-report_20260904__20260914.json         machine-readable DQ report
  reports/dq-report_20260904__20260914.md           human-readable DQ report
```

All of the above is Git-ignored (`.gitignore`), so no downloaded dataset or
run artifact is committed.

## Honest limitations (do not over-read)

1. **ENTSO-E and NL imbalance availability could not be verified** because the
   project has no `ENTSOE_API_KEY`. Endpoint reachability was verified
   separately, but no real series was fetched. This is recorded as
   UNAVAILABLE / UNVERIFIED — **not** as "real data pipeline works".
2. **Gold/features were not produced in this run** for the same reason. The
   offline fixture suite proves the Gold/feature code path works
   (fixture-verified), but a **live** Gold dataset requires load data.
3. Weather variables are **historical observations**, not NWP forecasts. The
   feature builder labels this explicitly; forecasting must not treat them as
   actuals-forecast.
4. Open-Meteo aggregation across 6 locations is an **equal-weight location
   mean** — a documented, deterministic default, not a spatial optimum.

## Next steps (when an ENTSO-E key is available)

1. Set `ENTSOE_API_KEY` in the environment (never in source or `.gitignore`).
2. Re-run: load, generation (B16/B18/B19), day-ahead prices will be fetched
   and Silver-written; Gold + features will be produced; the DQ report will
   switch to `entsoe_datasets_verified > 0` and `gold_produced = true`.
3. Re-run imbalance verification; if A85 return "No matching data found" the
   report will record it as UNAVAILABLE with the exact error — never a
   synthetic penalty (that remains an explicitly-labelled future fallback).