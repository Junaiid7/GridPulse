# GridPulse 2.0

Uncertainty-aware residual-load forecasting and battery storage dispatch
optimisation for the Dutch electricity market.

Status: **Phase 7 — production readiness & serving infrastructure complete**.
Ingestion (ENTSO-E / Open-Meteo), Silver/Gold transformation layer, leakage-safe feature engineering, probabilistic quantile forecasting (LightGBM), battery dispatch optimization (SciPy HiGHS LP), historical backtesting, FastAPI serving layer, Streamlit operational dashboard, structured logging, request correlation IDs, security hardening, and GitHub Actions CI/CD automation are fully implemented and verified with 405 passing tests.

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

## Architecture

Layered pipeline; each layer is an isolated subpackage under `src/gridpulse`:

| Package         | Responsibility                                        |
| --------------- | ----------------------------------------------------- |
| `ingestion`     | Pull raw ENTSO-E / Open-Meteo data                    |
| `validation`    | Schema and data quality checks                        |
| `transformation`| Bronze / Silver / Gold CSV warehouse & DST policy     |
| `features`      | Leakage-safe feature engineering                      |
| `forecast`      | Point & probabilistic residual load models (LightGBM) |
| `optimization`  | Battery storage dispatch & backtesting (SciPy HiGHS)  |
| `api`           | FastAPI serving layer (health, forecast, dispatch)    |
| `dashboard`     | Streamlit operational dashboard & visualizations      |
| `orchestration` | End-to-end pipeline & backtest orchestration          |
| `reporting`     | Data quality and evaluation reports                   |

Data follows a medallion layout: raw inputs (**bronze**) → cleaned/conformed
(**silver**) → curated analytical tables (**gold**). Datasets are never
committed to Git.

## Repository layout

```
src/gridpulse/                  # Main package (src layout)
  config.py                     # Centralised configuration
  ingestion/                    # ENTSO-E and Open-Meteo clients/parsers
  transformation/               # Silver/Gold CSV warehouse & DST policy
  features/                     # Leakage-safe feature builders
  forecast/                     # Probabilistic models (LightGBM), evaluation, persistence
  optimization/                 # Battery dispatch optimization & backtesting
  api/                          # FastAPI serving layer (app, dependencies, schemas)
  dashboard/                    # Streamlit operational dashboard app
  orchestration/                # End-to-end orchestration runner
  reporting/                    # Data quality reports
tests/                          # Comprehensive unit & integration test suite (405 tests)
config/                         # Centralised configuration profiles
docs/                           # Design and operations documentation
scripts/                        # Operational benchmarks & scripts
data/bronze/ data/silver/ data/gold/ data/models/ data/reports/  # Data tiers (git-ignored)
```

Configuration is centralised in `src/gridpulse/config.py` and overridable via
environment variables (see `.env.example`).

## Setup

Use the existing project virtual environment — do not create a new one.

```bash
cd C:\Users\junai\Desktop\GridPulse
.venv\Scripts\python.exe -m pip install -e ".[dev,serving,dashboard]"
```

This installs the package in editable mode along with `pytest`, `ruff`, and optional dependencies for serving (`fastapi`, `uvicorn`) and the dashboard (`streamlit`, `plotly`). Core runtime dependencies include `tzdata`, `lightgbm`, and `scipy`.

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

405 tests covering: HTTP retries/auth/rate-limiting, ENTSO-E parameter
construction/chunking/halving fallback/XML parsing (load, prices, generation,
flows, imbalance ZIP), Open-Meteo parameter construction/JSON parsing, Bronze
storage determinism/manifests, validation logic, offline pipeline fixtures, the
Silver/Gold transformation layer, leakage-safe feature builders, probabilistic residual-load forecasting models, battery dispatch optimization (HiGHS LP), backtesting/evaluation, API serving layers, and dashboard components.

## CI/CD & Quality Automation

GridPulse uses **GitHub Actions** for continuous integration. The CI workflow runs automatically on every push and pull request to `main`.

### Automated Checks
- **Matrix Python Versions**: Tests on Python 3.12 and Python 3.13 (`ubuntu-latest`).
- **Ruff Linting**: Checks code quality and imports (`ruff check .`).
- **Ruff Formatting**: Verifies code formatting compliance (`ruff format --check .`).
- **Test Suite**: Runs the complete pytest suite (`pytest -ra`).

### Running Quality Checks Locally
You can run the exact same checks locally before committing:

```bash
# Install development dependencies
.venv\Scripts\python.exe -m pip install -e ".[dev,serving,dashboard]"

# Run Ruff linter
.venv\Scripts\python.exe -m ruff check .

# Check code formatting
.venv\Scripts\python.exe -m ruff format --check .

# Run test suite
.venv\Scripts\python.exe -m pytest -ra
```

## Roadmap

**Completed:**
- ✅ Data ingestion (ENTSO-E + Open-Meteo → Bronze)
- ✅ Silver/Gold CSV warehouse with residual-load datasets
- ✅ Leakage-safe feature engineering (calendar, holiday, rolling, weather)
- ✅ Probabilistic residual-load forecasting (LightGBM quantile regression)
- ✅ Battery dispatch optimization (SciPy HiGHS LP)
- ✅ Historical backtesting and evaluation
- ✅ FastAPI serving layer (health, forecast, dispatch endpoints)
- ✅ Streamlit operational dashboard
- ✅ End-to-end orchestration
- ✅ GitHub Actions CI/CD (Python 3.12/3.13, Ruff linting/formatting, pytest)

**Future enhancements:**
- DuckDB/PostgreSQL warehouse backend (currently CSV-based)
- Great Expectations data quality framework integration
- MLflow experiment tracking
- Prefect workflow orchestration
- AI-generated analyst briefings

## Contributing

Keep configuration centralised, never hardcode secrets, and never commit
datasets. Reuse the existing `.venv` for all Python work.