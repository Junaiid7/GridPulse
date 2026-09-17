# Battery Dispatch Backtesting (Phase 4D-B)

Phase 4D-A built the dispatch engine and four comparable baseline strategies;
4D-B is the **historical harness** that answers *does uncertainty-aware dispatch
beat point-forecast dispatch in expectation on this synthetic window, and what
does the battery save?* It connects the Phase 4C per-offset P10/P50/P90
forecasts to `run_dispatch` over a chronological held-out window and settles
every schedule against the **realised** gold residual + day-ahead price.

Everything here is **FIXTURE-VERIFIED** — the real pipeline, real models and
real dispatch run on hand-crafted synthetic electricity fixtures (see
`real-data-run.md` for Phase 4A's honest live-data status). The held-out
window is ~9 dispatch days: it is **machinery validation** — it proves the
measurement harness works end-to-end and reproduces — and is not a claim about
real Dutch electricity or strategy superiority. When an `ENTSOE_API_KEY` is
configured, the same code path consumes the real feature/gold tables and the
`data_status` flag flips; until then every result carries `FIXTURE-VERIFIED`
and an explicit statement to that effect.

## Research questions

1. **What does the battery save?** Mean daily settled cost per strategy, and
   per-day savings vs the `no_battery` reference, with a percentile-bootstrap
   CI over paired dispatch days.
2. **Does uncertainty help?** `scenario_lp` (P10/P50/P90, shared decision)
   vs `lp_p50` (P50 only). On this fixture the two optima coincide; on real
   data the spread matters, which is exactly what the harness will measure.
3. **Is the harness honest?** Export-undercut hours, dispatched-but-infeasible
   days and dropped/incomplete profile days are all reported, never hidden.

## The per-offset profile design

The Phase 4B/4C contract issues once daily (default 06:00 UTC) and predicts a
*single* target hour (`issue + 24 h`). Dispatch needs a full 24-hour
P10/P50/P90 profile, so 4D-B builds **24 per-offset forecasting datasets**
(`horizon_hours = 24 + k` for `k = 0..23`), fits **one**
`QuantileRegressionModel` per offset on the **same** train window, and stitches
the per-hour triples into a per-issue-day `ForecastProfile`
(`target = issue + 24h .. issue + 47h`).

- Same `ChronologicalSplit` (default fractions 0.7 / 0.15 / 0.15) for every
  offset; fits see `[train_start, train_end)` only. Nothing about a target
  hour's realised residual is available while fitting or predicting.
- `ForecastProfilesResult.offsets` carries per-offset diagnostics (train/test
  row counts, missing-feature drops, quantile crossing, point and
  probabilistic metrics on each P10/P50/P90 triple) so calibration quality is
  measureable offset by offset and across all offsets.
- A profile is only *dispatchable* when complete (all 24 hours defined).
  Incomplete profiles (off the feature-table edge — typically the final issue
  days) are recorded in `dropped_days`, not silently repaired.

## Price-availability convention (documented assumption)

Day-ahead prices for the operating day are treated as **available at the
06:00 UTC issue instant**. In the real SDAC market the auction clears later on
the prior day; a production deployment would shift the issue hour after the
auction or use a price forecast. The convention is recorded verbatim in
`BacktestResult.info["price_availability_convention"]` and this document
covers the same ground — it is never silent.

## Settlement semantics (honest realised cost)

The dispatch input uses the forecast P10/P50/P90 and the gold day-ahead
prices. Settlement then applies the **realised** gold residual + the same gold
price to the *fixed* schedule:

```
g[t] = realised_residual[t] − discharge[t] + charge[t]
cost  = Σ_t price[t] · g[t]
```

- Realised residual enters **only at settlement** — never into the dispatch
  decision — so the no-export constraint is *forecast-feasibility*.
- When realised residual drops below the forecast the schedule was constrained
  against, `g[t] < 0`: those hours are counted as **export-undercut hours**
  (`n_export_undercut`, the hour indices in `export_undercut_hours`) and the
  honest (possibly export-revenue-discounted) cost is kept. Undercut is
  reported, never hidden.
- `energy_neutrality_delta_mwh` is the net SOC change over the horizon
  (`charge·η_c − discharge/η_d`); strategies that honour
  `terminal_soc = initial_soc` report ≈ 0.
- **Failures are per-day**: a `DispatchInfeasible` on one dispatch day is
  recorded in that strategy's `n_infeasible` and the day is excluded from that
  strategy's paired comparisons; the backtest continues.

## Metrics

- **Forecast** (per offset, and aggregated across all 24): MAE/RMSE of the P50
  (`point_metrics_as_p50`), pinball loss per quantile, empirical coverage,
  80% interval coverage/width, risk score, quantile-crossing counts (Phase 4C
  `evaluate.py`).
- **Dispatch** (per strategy over settled days): `n_days`, `n_infeasible`,
  mean/median/min/max daily cost, mean savings vs `no_battery`,
  total export-undercut hours, throughput (charge/discharge MWh), mean
  neutrality delta.
- **Comparisons** (`no_battery × greedy/lp_p50/scenario_lp`, plus
  `lp_p50 × scenario_lp`): percentile bootstrap CI on
  `mean(cost_A) − mean(cost_B)` over **paired dispatch days** (same day on both
  sides; days where either strategy was infeasible are excluded consistently).
  Positive difference ⇒ A more expensive. Fixed `seed` ⇒ bit-reproducible.

## Leakage rules (enforced and tested)

1. **Train-only fit** — offset models fit on `[train_start, train_end)` only.
2. **Settlement isolation** — the dispatch decision is a function of forecast
   + price alone; realised affects only settlement. Doubling the gold residual
   column leaves throughput unchanged and shifts only settled costs.
3. **Chronological split** — half-open `[start, end)`, strictly increasing,
   shared by all 24 offsets.
4. **As-of features** — predictors come from the issue-time feature row; the
   price convention and settlement are recorded in `info`, never silent.

`tests/unit/optimization/test_backtest_leakage.py` pins each of these.

## Fixture machinery vs real historical evaluation

| | This repo today (`FIXTURE-VERIFIED`) | Real ENTSO-E evaluation (future) |
|---|---|---|
| Feature/gold source | `tests/support/synthetic_electricity.py` | real pipeline tables (when `ENTSOE_API_KEY` set) |
| Held-out window | ~59-day synthetic run, ~9 dispatch days | as long as history allows |
| `data_status` | `"FIXTURE-VERIFIED"` | flips with the live backend |
| Claim made | machinery validation only | strategy evaluation on live data |
| report note | `info["statement"]` says so verbatim | same honest structure |

## Run

```bash
.venv/Scripts/python.exe scripts/phase4d_backtest.py
```

writes `data/reports/dispatch_backtest_phase4db.{json,md}` (the JSON is the
machine-readable form; the Markdown is a human summary). The demo prints the
headline figures, including the DATA STATUS banner and the comparison CIs.

## Repository layout

| File | Role |
|------|------|
| `src/gridpulse/optimization/backtest.py` | per-offset profiles, dispatch assembly, settlement, bootstrap CI, `run_dispatch_backtest`, `BacktestResult` |
| `src/gridpulse/optimization/__init__.py` | public Phase 4D-B exports + entry-point doc |
| `tests/support/backtest_fixture.py` | gold-row readers for the harness |
| `tests/unit/optimization/conftest.py` | shared synthetic pipeline/gold session fixtures |
| `tests/unit/optimization/test_backtest.py` | profile/assembly/settlement/bootstrap/backtest semantics |
| `tests/unit/optimization/test_backtest_leakage.py` | train-only fit, settlement isolation, split, price convention, determinism |
| `scripts/phase4d_backtest.py` | offline fixture demo → `data/reports/` |