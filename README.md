# GridPulse 2.0

Uncertainty-aware residual-load forecasting and battery storage dispatch
optimisation for the Dutch electricity market.

Status: **Phase 7 complete** — ingestion (ENTSO-E / Open-Meteo), Silver/Gold transformation layer, leakage-safe feature engineering, probabilistic quantile forecasting (LightGBM), battery dispatch optimization (SciPy HiGHS LP), historical backtesting, FastAPI serving layer, Streamlit operational dashboard, and GitHub Actions CI are fully implemented and verified with 405 passing tests.

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

## Quick Start

### 1. Clone and Set Up a Virtual Environment

GridPulse requires Python 3.12 or 3.13. We recommend creating a fresh virtual environment for your clone.

**Windows (PowerShell):**
```powershell
cd C:\path\to\your\projects
git clone https://github.com/Junaiid7/GridPulse.git
cd GridPulse
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
```

**macOS / Linux:**
```bash
cd /path/to/your/projects
git clone https://github.com/Junaiid7/GridPulse.git
cd GridPulse
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install GridPulse

Install the package in editable mode along with development tools, the API server, and the dashboard:

```bash
# Activate your virtual environment first (see step 1)
pip install -e ".[dev,serving,dashboard]"
```

This installs:
- **Core dependencies**: `tzdata`, `lightgbm`, `scipy`
- **Dev tools**: `pytest`, `ruff`
- **Serving**: `fastapi`, `uvicorn`
- **Dashboard**: `streamlit`, `plotly`

### 3. Run the Test Suite

Verify your installation by running the full test suite:

```bash
pytest -ra
```

Expected result: **405 tests passing**.

### 4. Run an End-to-End Demo (No API Key Required)

GridPulse includes an offline orchestration workflow that uses deterministic synthetic fixtures. This lets you explore the full pipeline without any external API keys.

```bash
# Run the full end-to-end orchestration on synthetic data
python -m gridpulse.orchestration
```

This will:
1. Generate synthetic ENTSO-E and Open-Meteo data
2. Run the Bronze → Silver → Gold transformation pipeline
3. Train a probabilistic quantile forecasting model (P10/P50/P90)
4. Execute a battery dispatch backtest
5. Write an orchestration report to `data/reports/orchestration_phase5.json`

**Tip:** You can customize the run with CLI flags. See:
```bash
python -m gridpulse.orchestration --help
```

To run the orchestration from a script, you can also use:
```bash
python scripts/phase5_orchestration.py
```

### 5. Start the FastAPI Server

Launch the API server (default: `http://localhost:8000`):

```bash
uvicorn gridpulse.api.app:app --reload --host 0.0.0.0 --port 8000
```

Key endpoints:
- `GET /health` — Liveness and configuration check
- `POST /forecast/predict` — Probabilistic forecast inference (requires a persisted model)
- `POST /optimization/dispatch` — Battery dispatch optimization
- `GET /orchestration/latest` — Latest orchestration report

Interactive API docs: `http://localhost:8000/docs`

### 6. Start the Streamlit Dashboard

Launch the operational dashboard (default: `http://localhost:8501`):

```bash
streamlit run src/gridpulse/dashboard/app.py --server.address=0.0.0.0 --server.port=8501
```

The dashboard provides visualizations of orchestration results, probabilistic forecasts, and dispatch optimization outputs.

## Secrets and Environment Variables

GridPulse reads configuration from environment variables. You can set these in your shell or via a `.env` file.

### Using a `.env` File (Optional)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and fill in your values.
3. Ensure `.env` is never committed (it is already in `.gitignore`).

### ENTSO-E API Key

- Required only for live ENTSO-E data (load, generation, prices, flows, imbalance).
- Not required for offline/fixture workflows or Open-Meteo data.
- Get your key at: https://transparency.entsoe.eu/content/Getting_started/
- Set the environment variable:

**Windows (PowerShell):**
```powershell
$env:ENTSOE_API_KEY = "your-key-here"
```

**macOS / Linux:**
```bash
export ENTSOE_API_KEY="your-key-here"
```

### Other Configuration

See `.env.example` for all supported variables, including:
- `GRIDPULSE_DATA_DIR` — Override the data directory
- `GRIDPULSE_LOG_LEVEL` — Logging verbosity
- `GRIDPULSE_HOLIDAYS_CSV` — Custom holiday calendar

## Data Tiers and Artifacts

GridPulse uses a medallion architecture with the following local directories (all **git-ignored**):

| Directory | Purpose |
|-----------|---------|
| `data/bronze/` | Raw, immutable inputs as ingested |
| `data/silver/` | Cleaned, validated, conformed data |
| `data/gold/` | Curated analytical tables (e.g., features) |
| `data/models/` | Persisted forecasting models |
| `data/reports/` | Data quality and orchestration reports |

These directories are created automatically when needed. Do not commit any contents under `data/` — they are generated artifacts.

## CI & Quality Automation

GridPulse uses **GitHub Actions** for continuous integration. The CI workflow runs automatically on every push and pull request to `main`.

### Automated Checks
- **Matrix Python Versions**: Tests on Python 3.12 and Python 3.13 (`ubuntu-latest`).
- **Ruff Linting**: Checks code quality and imports (`ruff check .`).
- **Ruff Formatting**: Verifies code formatting compliance (`ruff format --check .`).
- **Test Suite**: Runs the complete pytest suite (`pytest -ra`).

### Running Quality Checks Locally

You can run the exact same checks locally before committing:

```bash
# Ensure dev dependencies are installed
pip install -e ".[dev,serving,dashboard]"

# Run Ruff linter
ruff check .

# Check code formatting
ruff format --check .

# Run test suite
pytest -ra
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
- ✅ GitHub Actions CI (Python 3.12/3.13, Ruff linting/formatting, pytest)

**Future enhancements:**
- DuckDB/PostgreSQL warehouse backend (currently CSV-based)
- Great Expectations data quality framework integration
- MLflow experiment tracking
- Prefect workflow orchestration
- AI-generated analyst briefings

## Contributing

- Keep configuration centralised (`src/gridpulse/config.py` and environment variables).
- Never hardcode secrets or API keys in source code.
- Never commit datasets, models, or reports under `data/`.
- Create a fresh virtual environment for your work (do not reuse an existing `.venv` from another developer).
- Run `pytest -ra` and `ruff check .` before submitting pull requests.

## License

This project is licensed for educational and research purposes. See `pyproject.toml` for package metadata.
