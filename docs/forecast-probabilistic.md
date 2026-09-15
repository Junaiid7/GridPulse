# Probabilistic Forecasting (Phase 4C)

Phase 4B delivered *point* forecasts (a single number per 24 h-ahead target).
Phase 4C extends the same leakage-safe, issue-time contract with a **predictive
distribution**: the P10 / P50 / P90 quantiles of the residual load of the
target hour.

Everything here is **FIXTURE-VERIFIED** — it runs on hand-crafted synthetic
electricity + weather fixtures in the offline suite. No live ENTSO-E data is
used or claimed (see `real-data-run.md` for Phase 4A's honest report on live
availability).

## Contract

The probabilistic contract mirrors Phase 4B's issue-time semantics unchanged:

> A forecast **issued at 06:00 UTC** predicts the **residual load** of the hour
> starting **T+24 h**. Predictive features may only use observations
> `timestamp_utc < issue_time` (the `history_before()` cutoff from
> :doc:`transform-model`).

`QuantileForecast` (in ``gridpulse.forecast.probabilistic``) is a frozen
dataclass holding ``issue_time``, ``target_time`` and the three quantiles:

| Field | Meaning |
|-------|---------|
| `p10` | lower predictive quantile (alpha = 0.10) |
| `p50` | median predictive quantile (alpha = 0.50) |
| `p90` | upper predictive quantile (alpha = 0.90) |

Contract guarantees:

- **Ordering is enforced at the boundary.** Constructing a
  `QuantileForecast` with `p10 > p50` or `p50 > p90` raises `ValueError`.
  Invalid triples are detected and **refused, never silently reordered**.
- **Missing predictions stay `None`.** A row the model cannot serve
  (missing features, unfitted model) yields `p10 = p50 = p90 = None` — never a
  fabricated fill.
- **Nominal quantiles** are `(0.10, 0.50, 0.90)` with nominal central-interval
  coverage `NOMINAL_INTERVAL_COVERAGE = 0.80`.

## Quantile-crossing detection and correction

Independent per-quantile tree ensembles can produce a *crossed* triple
(e.g. a fitted P10 exceeding the fitted P50) for a row where the P10 tree is
atypically flat or steep.

- `detect_quantile_crossing(p10, p50, p90)` → `True` when the ordering is
  violated; a `None`-containing triple is *undetermined*, not a crossing.
- `correct_quantiles(p10, p50, p90)` → deterministic ascending re-labelling
  (smallest → p10, middle → p50, largest → p90), returning the triple plus a
  `corrected` flag. It is knowingly crude — extreme quantiles may be permuted
  — so it is applied only after the crossing has been **counted and reported**
  via the model's `crossing_stats()`:
  `{"n_detected", "n_corrected", "n_triples"}`. Crossing is never hidden.

## The quantile model (LightGBM)

`QuantileRegressionModel` (``gridpulse.forecast.models``) uses LightGBM's
**native Booster API** (`lgb.train` / `lgb.Booster`, not the sklearn wrapper) —
scikit-learn is deliberately **not** a dependency.

Three independent boosters are fit with `objective = "quantile"`, one per
alpha 0.10 / 0.50 / 0.90, then combined row-wise into a `QuantileForecast`.

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `boost_from_average=True` | **critical** | Seeds each booster from the alpha-quantile of the training labels — without it, P10 collapses to a constant (extremely poor coverage) |
| `n_estimators=120`, `learning_rate=0.05`, `num_leaves=15`, `min_child_samples=3` | | conservative small-tree defaults for the fixture data size |
| `deterministic=True`, `force_row_wise=True`, `num_threads=1`, `random_state` | | bit-reproducible fits |
| `min_train_rows=5` | | fewer complete rows ⇒ model stays unfitted (all-`None` predictions), never a garbage fit |
| `feature_columns` | | must be a subset of the dataset's leakage-safe `predictor_columns` |

`save()` / `load()` round-trip through JSON with base64-encoded booster model
strings — one entry per quantile.

## Metrics

All in ``gridpulse.forecast.evaluate`` (additive on Phase 4B):

| Metric | Definition | Notes |
|--------|------------|-------|
| `pinball_loss(actuals, quantile_preds, alpha)` | mean of `alpha·(y−q)` when `y≥q`, `(alpha−1)·(y−q)` when `y<q` | lower is better; `None` pairs are skipped |
| `empirical_coverage(actuals, q)` | fraction of actuals `≤` predicted quantile | P10→~0.10, P50→~0.50, P90→~0.90 when calibrated |
| `interval_coverage(actuals, lo, hi)` | fraction `lo ≤ actual ≤ hi` | nominal 0.80 |
| `interval_width_stats(lo, hi)` | mean/median/min/max/std/n of `P90−P10` | sharpness |
| `risk_score(p10, p50, p90)` | `(P90 − P10) / |P50|` | **relative uncertainty indicator**, not a financial risk measure |
| `probabilistic_metrics(...)` | aggregates everything above into one dict | includes mutually-checked n_triples, nominal references |

**Risk score guard:** when any quantile is missing or `|P50| ≤ P50_EPS`
(`1e-9`) the score is `None` — division-by-zero / unstable amplification is
never silently propagated. The score's sign is always positive because the
denominator is `|P50|`.

## Calibration diagnostics

The benchmark reports **empirical coverage per quantile** plus the nominal
references (0.10 / 0.50 / 0.90 and interval-80). The report's calibration note
is deliberately honest:

> Empirical coverage on a deterministic synthetic fixture is **not evidence of
> real-world calibration**; it only demonstrates the measurement machinery.
> Real ENTSO-E data is required before any calibration claim can be made.

## Benchmark A / B / C

`run_probabilistic_benchmark` (``gridpulse.forecast.benchmark``) rests on
Phase 4B's chronological, leakage-safe split (train strictly-before validation
strictly-before test; models fit on **train only**, evaluated on **test**):

| Model | Kind | Point metric used |
|-------|------|-------------------|
| A — seasonal naive 24h | point | MAE/RMSE/MAPE/bias |
| B — ridge linear regression | point | MAE/RMSE/MAPE/bias |
| C — LightGBM quantile regression | probabilistic | P50 as point metrics + full probabilistic metrics + calibration + risk score |

Output object is `ProbabilisticBenchmarkResult` with `to_dict()` (JSON,
`phase = "4C"`), `to_markdown()` (full human report incl. a visible
**DATA STATUS = FIXTURE-VERIFIED** banner) and `write()` (JSON + MD written
to `data/reports/`). Reproducibility metadata records the quantile model's
library version and hyperparameters (no secrets, no API keys).

### Fixture results (reproduce with the script below)

`scripts/phase4c_fixture_benchmark.py` — drives the real pipeline on synthetic
fixtures and writes `data/reports/forecast_benchmark_phase4c.{json,md}`:

```text
DATA STATUS = FIXTURE-VERIFIED
models: seasonal_naive_24h MAE 172.70 · ridge MAE 0.03 · lgbm P50 MAE 9.10
C pinball  q=0.10: 15.87  q=0.50: 4.55  q=0.90: 4.42
C coverage q=0.10: 0.10  q=0.50: 0.40  q=0.90: 0.80   (nominal 0.1/0.5/0.9)
C interval [P10,P90] mean width 201.77 MW (median 280.21, n=10)
C risk score mean 0.097  (near-zero P50: 0)
C crossings 4 detected / 4 corrected of 10 triples
```

P10 and P90 empirical coverage land within a hair of nominal on the fixture;
P50 empirical coverage (0.40) is off because median-fit leaves the late test
rows above the trained P50 range — a known small-data tree effect, shown
honestly rather than tuned away. **None of these numbers is a real-world
performance claim.**

## Leakage gate

The Phase 4C test suite (`tests/unit/forecast/test_probabilistic_leakage.py`)
proves the quantile model honours the same information-cutoff guarantees as
Phase 4B, each as a tamper-invariance proof:

1. **Future target labels** — setting every future `residual_load_mw` to a huge
   value must not move P10/P50/P90.
2. **Future feature values** — perturbing every validation/test row's predictor
   features must not move the fit or the train-window predictions.
3. **Train window only** — `metadata()["n_train_rows"]` equals the number of
   complete rows strictly inside the train window.
4. **Predictor subset** — the model's `feature_columns` are a subset of
   `predictor_columns` (no target column, no `target_utc`/`target_local`).
5. **Same chronological split** — train strictly-before validation
   strictly-before test.

## Reproducibility

Fixed `random_state`, LightGBM `deterministic=True` + `force_row_wise=True` +
`num_threads=1`, and integer-parseable fixtures make the benchmark
bit-reproducible (the test suite asserts two identical seeded runs produce
byte-identical JSON output).

## Honest limitations

1. **FIXTURE-VERIFIED, not operational.** The P10/P50/P90 machinery is proven
   on synthetic data; live ENTSO-E residual-load distributions may look very
   different (regime shifts, ramp shape, price-driven demand response).
2. **Crossing correction is crude.** Ascending re-labelling guarantees
   monotonicity but may permute the extreme quantiles; it is always counted and
   reported via `crossing_stats()`, never hidden.
3. **Weather features are observations only.** Historical Open-Meteo values are
   not NWP forecasts; a real production pipeline needs forecast-weather inputs
   available at issue time.
4. **Small-data tree effects.** Empirical coverage for P50 can drift on small
   windows (see fixture results). This is reported, not tuned away.

## Not implemented (Phase 4C scope boundary)

Battery/asset optimization, dispatch, backtesting, dashboarding, probabilistic
scoring against live data, and calibration tuning on real data are all out of
scope for Phase 4C and are **not** implemented.