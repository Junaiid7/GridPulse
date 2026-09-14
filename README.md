# GridPulse 2.0

Uncertainty-aware residual-load forecasting and battery storage dispatch
optimisation for the Dutch electricity market.

Status: **Phase 3 — transformation + feature-engineering foundation**.
Phase 2 (ENTSO-E / Open-Meteo ingestion → Bronze) is complete; Phase 3 adds the
Silver/Gold transformation layer (liquid CSV warehouse: cleaning, hourly NL
dataset with residual load) and a leakage-safe feature pipeline (calendar,
holiday, rolling, weather). Forecasting/optimisation (Phase 4+) are not
implemented yet.

## Purpose

GridPulse forecasts *residual load* — net consumer demand minus renewable
generation — for the Dutch market, with calibrated uncertainty estimates, and
uses those probabilistic forecasts to optimise battery storage dispatch.
The pipeline runs from public market and weather data (ENTSO-E, Open-Meteo)
through validation and a layered data warehouse, into forecasting, risk
quantification, dispatch optimisation and simulation, with a FastAPI serving
layer and a Streamlit dashboard.

## Data Sources (Phase 2 — verified)

| Source | Dataset | Endpoint | Resolution | Unit | Timezone | Auth |
|--------|---------|----------|------------|------|----------|------|
| ENTSO-E | Actual total load (A65/A16) | `web-api.tp.entsoe.eu/api` | 15-min / 60-min | MW | UTC | API key |
| ENTSO-E | Day-ahead load forecast (A65/A01) | same | 60-min | MW | UTC | API key |
| ENTSO-E | Actual generation by type (A75) | same | 15-min / 60-min | MW | UTC | API key |
| ENTSO-E | Day-ahead prices (A44) | same | 60-min | EUR/MWh | UTC | API key |
| ENTSO-E | Cross-border physical flows (A11) | same | 15-min / 60-min | MW | UTC | API key |
| ENTSO-E | Imbalance prices (A85) | same | varies | EUR/MWh | UTC | API key |
| Open-Meteo | Historical weather archive | `archive-api.open-meteo.com/v1/archive` | 60-min | see docs | UTC | None (non-commercial) |

Areas: NL (`10YNL----------L`), DE_LU (`10Y1001A1001A82H`), BE (`10YBE----------2`).

See [`docs/data-sources.md`](docs/data-sources.md) for detailed availability
notes, limitations, and the NL imbalance-price status.

## Architecture (target)

Layered pipeline; each layer is an isolated subpackage under `src/gridpulse`:

| Package         | Responsibility                                        |
| --------------- | ----------------------------------------------------- |
| `ingestion`     | Pull raw ENTSO-E / Open-Meteo data                    |
| `validation`    | Data-quality checks (Great Expectations)              |
| `warehouse`     | Bronze / Silver / Gold storage (DuckDB / PostgreSQL)  |
| `features`      | Feature engineering for forecasting                   |
| `models`        | Point + probabilistic load forecasts (LightGBM)       |
| `risk_engine`   | Uncertainty & residual-load risk quantification       |
| `optimization`  | Storage dispatch optimisation (SciPy / PuLP)          |
| `simulation`    | Historical backtesting of forecast-and-dispatch       |
| `evaluation`    | Experiment tracking & evaluation (MLflow)             |
| `api`           | FastAPI serving layer                                 |
| `dashboard`     | Streamlit operational dashboard                       |
| `ai_briefing`   | Automated analyst briefings                           |

Data follows a medallion layout: raw inputs (**bronze**) → cleaned/conformed
(**silver**) → curated analytical tables (**gold**). Datasets are never
committed to Git.

## Repository layout

```
src/gridpulse/                  # Main package (src layout)
  config.py                     # Centralised configuration
  ingestion/
    common/                     # Shared: HTTP, models, validation, storage
    entsoe/                     # ENTSO-E Transparency Platform client + parser
    weather/                    # Open-Meteo Historical Archive client + parser
    fallback/                   # Deliberately-unimplemented NL imbalance fallback
    wiring.py                   # Client construction from config
  transformation/               # Phase 3: Silver/Gold CSV warehouse
    times.py                    # UTC ↔ Europe/Amsterdam DST policy
    schema.py                   # Explicit Silver/Gold column schemas
    csvio.py / provenance.py    # Deterministic CSV + sidecar metadata
    aggregation.py              # Hourly downsampling + location combining
    silver/                     # Cleaning conformed source tables
    gold/                       # Hourly NL dataset + residual load
  features/                     # Phase 3: leakage-safe feature builders
    asof.py                     # The information-cut guard (history_before)
    calendar.py / holiday.py    # Calendar + holiday features
    rolling.py                  # Backward-looking rolling statistics
    weather.py                  # Last-observation-before weather features
tests/
  unit/ingestion/               # Phase 2 mocked tests (no live network)
  unit/transformation/          # Phase 3 Silver/Gold transformation tests
  unit/features/                # Phase 3 feature-engineering tests
  fixtures/                     # Synthetic XML/JSON test fixtures
config/                         # Centralised configuration profiles
docs/                           # Design and operations documentation
docs/transform-model.md         # Phase 3 transformation & leakage design
scripts/                        # Operational/maintenance scripts
notebooks/                      # Exploratory analysis
data/bronze/ data/silver/ data/gold/   # Data tiers (git-ignored)
```

Configuration is centralised in `src/gridpulse/config.py` and overridable via
environment variables (see `.env.example`).

## Setup

Use the existing project virtual environment — do not create a new one.

```bash
cd C:\Users\junai\Desktop\GridPulse
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

This installs the package in editable mode plus `pytest` (the only dev
dependency so far). The runtime has one dependency — `tzdata`, needed for
`zoneinfo.ZoneInfo` on Windows — everything else uses only the Python standard
library (urllib, xml, zipfile, json, logging, csv, zoneinfo).

## Secrets

Set `ENTSOE_API_KEY` in your environment or in a `.env` file (never committed).
Open-Meteo requires no API key for non-commercial use.

```bash
# Windows PowerShell
$env:ENTSOE_API_KEY = "your-key-here"

# Or copy .env.example to .env and fill in the key
```

## Testing

```bash
.venv\Scripts\python.exe -m pytest
```

168 tests covering: HTTP retries/auth/rate-limiting, ENTSO-E parameter
construction/chunking/halving fallback/XML parsing (load, prices, generation,
flows, imbalance ZIP), Open-Meteo parameter construction/JSON parsing, Bronze
storage determinism/manifests, validation logic, offline pipeline fixtures, the
Silver/Gold transformation layer (timezone/DST policy, aggregation, schemas,
CSV storage, residual load, hourly NL dataset) and the leakage-safe feature
builders (as-of cut, calendar, holiday calendars, rolling, weather).

## Roadmap (later phases — not implemented yet)

1. ✅ Data ingestion foundation (ENTSO-E + Open-Meteo → Bronze)
2. 🚧 Silver/Gold warehouse (CSV foundation done; DuckDB/PostgreSQL, Great
   Expectations deferred) + residual-load Gold dataset
3. 🚧 Feature-engineering foundation (calendar/holiday/rolling/weather done);
   LightGBM point/probabilistic residual-load models not yet started
4. Risk engine, storage dispatch optimisation, historical simulation
5. MLflow evaluation, FastAPI serving, Streamlit dashboard, AI briefing
6. Prefect orchestration, Docker, GitHub Actions CI/CD

## Contributing

Keep configuration centralised, never hardcode secrets, and never commit
datasets. Reuse the existing `.venv` for all Python work.