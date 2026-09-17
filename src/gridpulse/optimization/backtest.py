"""Backtesting and historical evaluation for battery dispatch (Phase 4D-B).

Connects the Phase 4C probabilistic forecast layer to the Phase 4D-A dispatch
engine in a common, leak-safe, historical harness:

- **Per-offset forecast profile**. The Phase 4B/4C contract issues once daily
  at ``issue_hour_utc`` (default 06:00 UTC) and predicts a *single* target hour
  (``target = issue + 24h``). Dispatch needs a full 24-hour
  P10/P50/P90 profile, so this module builds **24 per-offset forecasting
  datasets** (``horizon_hours = 24 + k`` for ``k = 0..23``), fits one
  ``QuantileRegressionModel`` per offset on the **same** train window, and
  stitches the per-hour triples into a per-issue-day :class:`ForecastProfile`.
- **Dispatch + settlement**. Each complete profile becomes a validated
  ``DispatchInput`` (prices from the gold table, scenarios from the forecast).
  Every strategy's schedule is then *settled* against the **realised** gold
  residual (plus the same gold price). Realised residual only ever enters at
  settlement — never into the dispatch decision — so the no-export constraint
  is forecast-feasibility, and any hour where realised load drops below the
  forecast is reported honestly as an export-undercut hour.

DATA STATUS
-----------
This phase is **FIXTURE-VERIFIED**: it runs the real pipeline + real models +
real dispatch on the hand-crafted synthetic window
(``tests/support/synthetic_electricity.py``). The ~9-day held-out window is
*machinery validation only* — it proves the measurement harness, and is not a
claim about real Dutch electricity. When an ``ENTSOE_API_KEY`` is configured,
the same code path consumes the real feature/gold tables and the flag flips;
until then ``data_status`` stays ``"FIXTURE-VERIFIED"``.

Conventions (documented, never silent)
--------------------------------------
- **Price availability**: day-ahead prices for the operating day are treated
  as available at the 06:00 UTC issue instant. In the real SDAC market the
  auction clears later on the prior day; a production deployment would shift
  the issue hour after the auction or use a price forecast. The convention is
  recorded in ``BacktestResult.info`` and ``docs/backtest.md``.
- **Settlement**: realised gold residual + the gold price, applied only after
  the schedule is fixed. Realised grid demand ``g[t] = realised[t] - dis[t]
  + ch[t]`` may go negative when realised falls below the forecast the
  schedule was constrained against; those hours are counted as
  ``export_undercut_hours`` and the honest realised cost is kept.
- **Failures are per-day**: a :class:`DispatchInfeasible` on one dispatch day
  is recorded in ``dropped_days``/per-strategy ``n_infeasible`` and the
  backtest continues.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..forecast.benchmark import DEFAULT_LR_FEATURES
from ..forecast.contract import build_forecasting_dataset
from ..forecast.evaluate import compute_point_metrics, probabilistic_metrics
from ..forecast.models import QuantileRegressionModel
from ..forecast.split import ChronologicalSplit, chronological_split_by_fraction
from .battery import BatteryConfig
from .contract import DispatchInfeasible, DispatchInput
from .strategies import STRATEGIES, run_dispatch

#: Default candidate predictors for the per-offset models (mirrors the Phase
#: 4B/4C ridge / quantile benchmark set; only columns actually present in the
#: feature table are used).
DEFAULT_FEATURES = DEFAULT_LR_FEATURES

#: Which of the real dispatch horizons correspond to a 24 h day-ahead lookahead.
PROFILE_HOURS = tuple(range(24))

#: Settlement / price tolerance below which a value is treated as zero-safe.
COST_TOL = 1e-9

#: Truthful label for every result this module returns (see module docstring).
DATA_STATUS = "FIXTURE-VERIFIED"

PRICE_AVAILABILITY_CONVENTION = (
    "day-ahead prices for the operating day are treated as available at the "
    "issue instant (06:00 UTC). This is a documented convention, not a market "
    "realism claim: the real SDAC auction clears later on the prior day, and a "
    "production deployment would shift the issue hour after the auction or use "
    "a price forecast."
)

ASOF_POLICY = (
    "per-offset forecast issued once daily at the issue hour; predictors from "
    "the issue-time feature row (strictly before issue); label = realised "
    "residual of the target hour; price = gold day-ahead price for the "
    "operating day (documented convention); settlement = realised gold "
    "residual + price, applied only after the schedule is fixed."
)


# ============================================================================
# Forecast profile
# ============================================================================
@dataclass(frozen=True)
class ForecastProfile:
    """One 24-hour P10/P50/P90 residual-load profile for a dispatch day.

    ``target_times`` are the ``issue + 24h .. issue + 47h`` UTC hours.
    ``p10_mw``/``p50_mw``/``p90_mw`` hold one value per target hour, or
    ``None`` when that hour has no forecast (off the feature-table edge or a
    model that refused to fit). ``n_defined`` counts hours with a complete
    triple; a profile is only *dispatchable* when complete (all 24 defined).
    """

    issue_time: datetime
    target_times: tuple
    p10_mw: tuple
    p50_mw: tuple
    p90_mw: tuple
    n_defined: int = 0

    @property
    def complete(self) -> bool:
        return self.n_defined == len(self.target_times)

    def to_dict(self) -> dict:
        return {
            "issue_time": self.issue_time.isoformat(),
            "target_times": [t.isoformat() for t in self.target_times],
            "p10_mw": list(self.p10_mw),
            "p50_mw": list(self.p50_mw),
            "p90_mw": list(self.p90_mw),
            "n_defined": self.n_defined,
        }


@dataclass(frozen=True)
class ForecastProfilesResult:
    """Outcome of :func:`build_forecast_profiles`.

    ``profiles`` maps each test-window issue time (aware UTC) to its
    :class:`ForecastProfile`. ``offsets`` holds per-offset fit/predict
    diagnostics (train/test row counts, crossing, point + probabilistic
    metrics on the P10/P50/P90 triples). ``split`` is the single shared
    ``ChronologicalSplit`` used by all 24 offsets.

    ``aligned`` carries the raw per-offset aligned vectors
    ``(actuals, p10, p50, p90)`` so the top-level backtest can aggregate the
    probabilistic view across all offsets. It is *not* part of the serialized
    report (the compact ``offsets`` entries are); it exists for computation
    only.
    """

    profiles: Mapping
    offsets: list
    aligned: list
    split: ChronologicalSplit
    feature_columns: list
    issue_hour_utc: int


def build_forecast_profiles(
    feature_rows,
    *,
    split: Optional[ChronologicalSplit] = None,
    issue_hour_utc: int = 6,
    seed: int = 0,
    quantile_model_kwargs: Optional[dict] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> ForecastProfilesResult:
    """Build a per-issue-day 24-hour forecast profile over the test window.

    One ``QuantileRegressionModel`` per target-hour offset ``k = 0..23``
    (``horizon_hours = 24 + k``), all fit on the **same** train window and
    evaluated on the **same** test window, per the documented price/issue
    convention. Offsets inherit the caller-provided ``split`` verbatim when one
    is given, else a deterministic chronological fraction split computed from
    the issue timestamps of the base (offset-0) dataset.

    Parameters
    ----------
    feature_rows:
        Pipeline feature-table rows (``features_rows(run)`` output).
    split:
        Optional ``ChronologicalSplit``; honoured verbatim. Otherwise
        ``chronological_split_by_fraction`` on the base dataset's issue times.
    issue_hour_utc:
        One daily issue at this UTC clock hour (must match 4B/4C semantics).
    seed:
        Seed shared by LightGBM (deterministic) and the bootstrap below.
    quantile_model_kwargs:
        Passed to :class:`~gridpulse.forecast.models.QuantileRegressionModel`.
    feature_columns:
        Predictor columns; defaults to ``DEFAULT_FEATURES`` present in the
        feature table (mirrors the 4B/4C benchmark).

    Returns
    -------
    :class:`ForecastProfilesResult`.
    """
    base = build_forecasting_dataset(
        feature_rows, issue_hour_utc=issue_hour_utc, horizon_hours=24, naive_lag_hours=24
    )
    if not base.rows:
        raise ValueError("forecasting dataset is empty; cannot build dispatch profiles")

    if split is None:
        split = chronological_split_by_fraction(base.issue_times)

    present = set(base.predictor_columns)
    if feature_columns is not None:
        cols = [c for c in feature_columns if c in present]
    else:
        cols = [c for c in DEFAULT_FEATURES if c in present]
    if not cols:
        raise ValueError("no default/requested features present in the forecasting dataset")

    by_issue: dict = {}
    offsets: list[dict] = []
    aligned: list = []
    for k in PROFILE_HOURS:
        ds = build_forecasting_dataset(
            feature_rows, issue_hour_utc=issue_hour_utc,
            horizon_hours=24 + k, naive_lag_hours=24,
        )
        model = QuantileRegressionModel(
            feature_columns=list(cols), random_state=seed,
            **(quantile_model_kwargs or {}),
        )
        model.fit(ds, start=split.train_start, end=split.train_end)
        fcs = model.predict(ds, start=split.test_start, end=split.test_end)
        rows = [r for r in ds.rows if split.test_start <= r.issue_time < split.test_end]

        actuals = [r.target_mw for r in rows]
        p10 = [f.p10 for f in fcs]
        p50 = [f.p50 for f in fcs]
        p90 = [f.p90 for f in fcs]
        point = compute_point_metrics(actuals, p50)
        prob = probabilistic_metrics(actuals, p10, p50, p90)

        offsets.append(
            {
                "offset": k,
                "horizon_hours": 24 + k,
                "n_test_rows": len(rows),
                "n_aligned": prob["n_triples"],
                "fitted": model.metadata()["fitted"],
                "n_train_rows": model.metadata()["n_train_rows"],
                "n_rows_dropped_missing_features": model.metadata().get(
                    "n_rows_dropped_missing_features", 0
                ),
                "crossing": model.crossing_stats(),
                "point_metrics_as_p50": point,
                "probabilistic_metrics": prob,
            }
        )
        # Retain only the aligned (all-defined) rows for the cross-offset view.
        aligned_actuals: list[float] = []
        aligned_p10: list[float] = []
        aligned_p50: list[float] = []
        aligned_p90: list[float] = []
        for a, lo, mid, hi in zip(actuals, p10, p50, p90):
            if None in (a, lo, mid, hi):
                continue
            aligned_actuals.append(float(a))
            aligned_p10.append(float(lo))
            aligned_p50.append(float(mid))
            aligned_p90.append(float(hi))
        aligned.append((aligned_actuals, aligned_p10, aligned_p50, aligned_p90))

        for r, f in zip(rows, fcs):
            by_issue.setdefault(r.issue_time, {})[k] = f

    profiles = {}
    for issue, per_offset in by_issue.items():
        target_times = tuple(issue + timedelta(hours=24 + k) for k in PROFILE_HOURS)
        p10 = []
        p50 = []
        p90 = []
        n_defined = 0
        for k in PROFILE_HOURS:
            fc = per_offset.get(k)
            if fc is not None and fc.p50 is not None:
                p10.append(fc.p10)
                p50.append(fc.p50)
                p90.append(fc.p90)
                n_defined += 1
            else:
                p10.append(None)
                p50.append(None)
                p90.append(None)
        profiles[issue] = ForecastProfile(
            issue_time=issue,
            target_times=target_times,
            p10_mw=tuple(p10),
            p50_mw=tuple(p50),
            p90_mw=tuple(p90),
            n_defined=n_defined,
        )

    return ForecastProfilesResult(
        profiles=profiles,
        offsets=offsets,
        aligned=aligned,
        split=split,
        feature_columns=cols,
        issue_hour_utc=issue_hour_utc,
    )


# ============================================================================
# Dispatch input assembly + settlement
# ============================================================================
def assemble_dispatch_input(
    profile: ForecastProfile,
    price_eur_mwh: Sequence[float],
    *,
    battery: Optional[BatteryConfig] = None,
) -> DispatchInput:
    """Build a validated :class:`DispatchInput` from a complete profile.

    The P50 forecast becomes ``residual_load_mw`` and the P10/P50/P90 profiles
    become ``scenario_residual_load_mw``, so ordering and
    ``scenario["p50"] == residual_load_mw`` hold by construction (and are
    re-validated by :class:`DispatchInput`). Prices are the gold day-ahead
    prices for the operating day. Raises ``ValueError`` on an incomplete
    profile or a price length mismatch — never silently fills.
    """
    if not profile.complete:
        raise ValueError(
            f"cannot build a DispatchInput from an incomplete profile "
            f"(n_defined={profile.n_defined} of {len(profile.target_times)})"
        )
    prices = [float(v) for v in price_eur_mwh]
    if len(prices) != len(profile.target_times):
        raise ValueError(
            f"need one price per target hour ({len(profile.target_times)}); "
            f"got {len(prices)}"
        )
    return DispatchInput(
        issue_time=profile.issue_time,
        target_times=profile.target_times,
        price_eur_mwh=prices,
        residual_load_mw=tuple(float(v) for v in profile.p50_mw),
        scenario_residual_load_mw={
            "p10": tuple(float(v) for v in profile.p10_mw),
            "p50": tuple(float(v) for v in profile.p50_mw),
            "p90": tuple(float(v) for v in profile.p90_mw),
        },
        battery=battery if battery is not None else BatteryConfig(),
    )


def settle_day(
    result, realised_residual_mw: Sequence[float], price_eur_mwh: Sequence[float]
) -> dict:
    """Settle a dispatch schedule against the *realised* gold residual + price.

    The schedule (``charge_mw``/``discharge_mw``) is fixed and never changed
    here — this is the honest cost a market participant would pay. Returns::

        cost_eur                 realised cost  sum_t price[t]*g[t]
        grid_demand_realised      per-hour realised net grid demand
        export_undercut_hours     hours where realised g[t] < 0 (would-be export)
        n_export_undercut         count of those hours
        throughput_charge_mwh     total charged energy
        throughput_discharge_mwh  total discharged grid-energy
        energy_neutrality_delta   net SOC change over the horizon (MWh), ~0 when
                                  terminal_soc == initial_soc
    """
    n = len(result.target_times)
    if len(realised_residual_mw) != n or len(price_eur_mwh) != n:
        raise ValueError("realised residual and prices must match the horizon length")
    realised = [float(v) for v in realised_residual_mw]
    prices = [float(v) for v in price_eur_mwh]

    grid = [realised[t] - result.discharge_mw[t] + result.charge_mw[t] for t in range(n)]
    cost = sum(prices[t] * grid[t] for t in range(n))
    undercut = [t for t in range(n) if grid[t] < -COST_TOL]
    ch_total = sum(result.charge_mw)
    dis_total = sum(result.discharge_mw)
    neutral = (
        ch_total * result.battery.charge_efficiency
        - dis_total / result.battery.discharge_efficiency
    )
    return {
        "cost_eur": cost,
        "grid_demand_realised": grid,
        "export_undercut_hours": undercut,
        "n_export_undercut": len(undercut),
        "throughput_charge_mwh": ch_total,
        "throughput_discharge_mwh": dis_total,
        "energy_neutrality_delta_mwh": neutral,
    }


# ============================================================================
# Bootstrap CI on mean daily cost difference (paired dispatch days)
# ============================================================================
def bootstrap_cost_difference_ci(
    daily_a: Sequence[Optional[float]],
    daily_b: Sequence[Optional[float]],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci_level: float = 0.95,
) -> dict:
    """Percentile bootstrap CI for ``mean(cost_A) - mean(cost_B)``.

    The two lists are **paired dispatch days** (same index = same day). Pairs
    where either side is ``None`` (a strategy was infeasible that day) are
    excluded consistently. A positive ``difference`` means strategy A is *more
    expensive* on average. Deterministic for a fixed ``seed``.

    Returns ``difference``, ``ci_low``, ``ci_high``, ``n_boot``,
    ``ci_level``, ``n_samples`` (number of paired days used).
    """
    if len(daily_a) != len(daily_b):
        raise ValueError("daily cost lists must be parallel (one entry per dispatch day)")
    pairs = [
        (float(a), float(b))
        for a, b in zip(daily_a, daily_b)
        if a is not None and b is not None
    ]
    n = len(pairs)
    empty = {
        "difference": None, "ci_low": None, "ci_high": None,
        "n_boot": n_boot, "ci_level": ci_level, "n_samples": 0,
    }
    if n == 0:
        return empty
    point = sum(a - b for a, b in pairs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choices(pairs, k=n)
        boots.append(sum(a - b for a, b in sample) / n)
    boots.sort()
    alpha = (1 - ci_level) / 2
    lo_idx = max(0, int(math.floor((n_boot - 1) * alpha)))
    hi_idx = min(n_boot - 1, int(math.floor((n_boot - 1) * (1 - alpha))))
    return {
        "difference": point,
        "ci_low": boots[lo_idx],
        "ci_high": boots[hi_idx],
        "n_boot": n_boot,
        "ci_level": ci_level,
        "n_samples": n,
    }


# ============================================================================
# Top-level backtest
# ============================================================================
def run_dispatch_backtest(
    feature_rows,
    gold_rows,
    *,
    split: Optional[ChronologicalSplit] = None,
    seed: int = 0,
    battery: Optional[BatteryConfig] = None,
    strategies: Optional[Sequence[str]] = None,
    issue_hour_utc: int = 6,
    quantile_model_kwargs: Optional[dict] = None,
    n_boot: int = 2000,
) -> "BacktestResult":
    """Run the full Phase 4D-B backtest over the test window.

    ``feature_rows`` drive the per-offset forecast profiles; ``gold_rows``
    (gold ``hourly.csv`` dicts, one per hour with ``timestamp_utc``,
    ``residual_load_mw``, ``day_ahead_price_eur_mwh``) provide the realised
    prices + residual for dispatch and settlement. Every complete profile day
    is dispatched under each strategy and settled against the realised data.

    Strategies are the Phase 4D-A set by default. Pairwise mean-cost
    comparisons are computed for ``(no_battery, X)`` for each ``X`` plus
    ``(lp_p50, scenario_lp)``; all share one ``seed`` and one ``n_boot``.
    """
    battery = battery if battery is not None else BatteryConfig()
    strat_names = list(strategies) if strategies is not None else [
        "no_battery", "greedy_arbitrage", "lp_p50", "scenario_lp",
    ]
    unknown = [s for s in strat_names if s not in STRATEGIES]
    if unknown:
        raise ValueError(f"unknown strategy(s) {unknown}; expected one of {sorted(STRATEGIES)}")

    profiles_result = build_forecast_profiles(
        feature_rows,
        split=split,
        issue_hour_utc=issue_hour_utc,
        seed=seed,
        quantile_model_kwargs=quantile_model_kwargs,
    )
    split = profiles_result.split

    gold_by_ts = {}
    for row in gold_rows:
        ts = _parse_gold_timestamp(row.get("timestamp_utc"))
        if ts is not None:
            gold_by_ts[ts] = row

    days: list[dict] = []
    dropped: dict[str, str] = {}
    for issue, profile in sorted(profiles_result.profiles.items()):
        if not profile.complete:
            dropped[issue.isoformat()] = (
                f"incomplete forecast profile ({profile.n_defined} of 24 hours)"
            )
            continue
        prices = []
        realised = []
        ok = True
        for ts in profile.target_times:
            row = gold_by_ts.get(ts)
            price = _parse_gold_float(row.get("day_ahead_price_eur_mwh")) if row else None
            resid = _parse_gold_float(row.get("residual_load_mw")) if row else None
            if price is None or resid is None:
                ok = False
                break
            prices.append(price)
            realised.append(resid)
        if not ok:
            dropped[issue.isoformat()] = "gold price/residual missing for the dispatch horizon"
            continue

        inputs = assemble_dispatch_input(profile, prices, battery=battery)
        results = {}
        for strat in strat_names:
            try:
                results[strat] = settle_day(run_dispatch(inputs, strategy=strat), realised, prices)
            except DispatchInfeasible as exc:
                results[strat] = {"infeasible": True, "message": str(exc)}
        days.append(
            {
                "issue_time": issue,
                "target_times": profile.target_times,
                "price_eur_mwh": prices,
                "realised_residual_mw": realised,
                "results": results,
            }
        )

    has_no_battery = "no_battery" in strat_names
    # --- per-strategy aggregates over the settled days ----------------------
    strategy_stats = []
    for strat in strat_names:
        costs: list[float] = []
        savings: list[float] = []
        n_infeasible = 0
        n_export = 0
        throughput_ch = 0.0
        throughput_dis = 0.0
        neutral_sum = 0.0
        for day in days:
            res = day["results"][strat]
            if res.get("infeasible"):
                n_infeasible += 1
                continue
            costs.append(res["cost_eur"])
            if has_no_battery:
                no_bat = day["results"]["no_battery"]
                if not no_bat.get("infeasible"):
                    savings.append(no_bat["cost_eur"] - res["cost_eur"])
            n_export += res["n_export_undercut"]
            throughput_ch += res["throughput_charge_mwh"]
            throughput_dis += res["throughput_discharge_mwh"]
            neutral_sum += res["energy_neutrality_delta_mwh"]
        n_days = len(costs)
        strategy_stats.append(
            {
                "strategy": strat,
                "n_days": n_days,
                "n_infeasible": n_infeasible,
                "daily_costs_eur": costs,
                "mean_daily_cost_eur": _mean(costs),
                "median_daily_cost_eur": _median(costs),
                "min_daily_cost_eur": min(costs) if costs else None,
                "max_daily_cost_eur": max(costs) if costs else None,
                "mean_daily_savings_vs_no_battery_eur": _mean(savings),
                "total_export_undercut_hours": n_export,
                "total_throughput_charge_mwh": throughput_ch,
                "total_throughput_discharge_mwh": throughput_dis,
                "mean_energy_neutrality_delta_mwh": (
                    neutral_sum / n_days if n_days else None
                ),
            }
        )

    # --- pairwise comparisons (paired days, bootstrap CI) -------------------
    comparisons: list[dict] = []
    pairs = [("no_battery", "greedy_arbitrage"), ("no_battery", "lp_p50"),
             ("no_battery", "scenario_lp"), ("lp_p50", "scenario_lp")]
    pairs = [(a, b) for a, b in pairs if a in strat_names and b in strat_names]
    for a, b in pairs:
        ca: list[Optional[float]] = []
        cb: list[Optional[float]] = []
        for day in days:
            ra = day["results"][a]
            rb = day["results"][b]
            if ra.get("infeasible") or rb.get("infeasible"):
                ca.append(None)
                cb.append(None)
                continue
            ca.append(ra["cost_eur"])
            cb.append(rb["cost_eur"])
        ci = bootstrap_cost_difference_ci(ca, cb, n_boot=n_boot, seed=seed)
        comparisons.append(
            {
                "A": a,
                "B": b,
                "direction": f"mean_daily_cost({a}) - mean_daily_cost({b})",
                "difference": ci["difference"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "ci_level": ci["ci_level"],
                "n_boot": ci["n_boot"],
                "n_days": ci["n_samples"],
                "method": "percentile bootstrap on paired dispatch days",
                "note": (
                    "positive difference => strategy A ('"
                    + a + "') is more expensive on average. An interval "
                    "excluding 0 is statistically suggestive only on this "
                    "fixture-verified, short-window evaluation."
                ),
            }
        )

    # --- forecast summary across offsets (headline calibration/risk) --------
    forecast_summary = _aggregate_offsets(profiles_result.aligned)

    n_complete = sum(1 for p in profiles_result.profiles.values() if p.complete)
    info = {
        "data_status": DATA_STATUS,
        "target_column": "residual_load_mw",
        "horizon_hours": 24,
        "issue_hour_utc": issue_hour_utc,
        "issue_cadence": "once_daily",
        "asof_policy": ASOF_POLICY,
        "price_availability_convention": PRICE_AVAILABILITY_CONVENTION,
        "n_profiles": len(profiles_result.profiles),
        "n_profiles_complete": n_complete,
        "n_dispatch_days": len(days),
        "n_dropped_days": len(dropped),
        "battery": battery.to_dict(),
        "strategies": strat_names,
        "feature_columns": profiles_result.feature_columns,
        "forecast_summary": forecast_summary,
        "export_undercut_note": (
            "settlement uses realised gold residual; hours where realised grid "
            "demand goes negative (below the forecast the schedule was "
            "constrained against) are counted as export-undercut, never hidden."
        ),
        "statement": (
            "UNVERIFIED for live electricity: there is no ENTSO-E API key; "
            "the feature/gold tables behind this backtest were generated from "
            "hand-crafted synthetic fixtures exercising the real pipeline, and "
            "the held-out window (~9 dispatch days) is machinery validation "
            "only. No claim about real Dutch electricity is made."
        ),
    }

    reproducibility = {
        "seed": seed,
        "n_boot": n_boot,
        "gridpulse_version": _gridpulse_version(),
        "python_version": _python_version(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    return BacktestResult(
        phase="4D-B",
        data_status=DATA_STATUS,
        info=info,
        forecast_offsets=profiles_result.offsets,
        strategies=strategy_stats,
        comparisons=comparisons,
        split=split.to_dict(),
        dropped_days=dropped,
        reproducibility=reproducibility,
    )


def _aggregate_offsets(aligned_offsets: Sequence[tuple]) -> dict:
    """Concatenate every offset's aligned triples into one probabilistic view."""
    actuals: list[float] = []
    p10: list[float] = []
    p50: list[float] = []
    p90: list[float] = []
    n_pairs = 0
    for off_actuals, off_p10, off_p50, off_p90 in aligned_offsets:
        n_pairs += len(off_actuals)
        actuals.extend(off_actuals)
        p10.extend(off_p10)
        p50.extend(off_p50)
        p90.extend(off_p90)
    if not actuals:
        return {"n_aligned_pairs": 0, "probabilistic_metrics": {}}
    return {
        "n_aligned_pairs": n_pairs,
        "probabilistic_metrics": probabilistic_metrics(actuals, p10, p50, p90),
    }


# ============================================================================
# BacktestResult
# ============================================================================
@dataclass(frozen=True)
class BacktestResult:
    """Structured Phase 4D-B output (JSON + Markdown renderable)."""

    phase: str
    data_status: str
    info: Mapping
    forecast_offsets: list
    strategies: list
    comparisons: list
    split: Mapping
    dropped_days: Mapping
    reproducibility: Mapping

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "data_status": self.data_status,
            "info": self.info,
            "forecast_offsets": self.forecast_offsets,
            "strategies": self.strategies,
            "comparisons": self.comparisons,
            "split": self.split,
            "dropped_days": self.dropped_days,
            "reproducibility": self.reproducibility,
        }

    def to_markdown(self) -> str:
        l = []
        l.append("# Battery Dispatch Backtest (Phase 4D-B)")
        l.append("")
        l.append(f"**DATA STATUS = `{self.data_status}`**")
        l.append("")
        info = self.info
        l.append(
            f"- Target: `{info['target_column']}` · horizon {info['horizon_hours']} h · "
            f"issue once daily at {info['issue_hour_utc']}:00 UTC"
        )
        l.append(
            f"- Dispatch days evaluated: `{info['n_dispatch_days']}` "
            f"(`{info['n_profiles_complete']}` complete profiles of "
            f"`{info['n_profiles']}`; `{info['n_dropped_days']}` dropped days)")
        l.append(f"- As-of policy: {info['asof_policy']}")
        l.append(f"- Price convention: {info['price_availability_convention']}")
        l.append("")
        l.append(f"- Statement: {info['statement']}")
        l.append("")
        l.append("## Forecast evaluation (per target-hour offset, P10/P50/P90)")
        l.append("")
        l.append("| offset | n_aligned | MAE (P50) | pinball@0.5 | int.80 cov | width mean | crossing det |")
        l.append("|---|---|---|---|---|---|---|")
        for off in self.forecast_offsets:
            point = off["point_metrics_as_p50"]
            prob = off["probabilistic_metrics"]
            w = prob["interval_width"]
            x = off["crossing"]
            l.append(
                f"| {off['offset']} | {off['n_aligned']} | {_fmt(point['mae'])} | "
                f"{_fmt(prob['pinball']['0.50'])} | {_fmt(prob['interval_coverage'])} | "
                f"{_fmt(w['mean'])} | {x['n_detected']} |"
            )
        fs = info["forecast_summary"]
        fsp = fs.get("probabilistic_metrics") or {}
        if fsp:
            l.append("")
            l.append("### Aggregated probabilistic summary (all offsets)")
            l.append("")
            l.append(
                f"- pinball P10/P50/P90: {_fmt(fsp['pinball']['0.10'])} / "
                f"{_fmt(fsp['pinball']['0.50'])} / {_fmt(fsp['pinball']['0.90'])}"
            )
            l.append(
                f"- empirical coverage: {_fmt(fsp['empirical_coverage']['0.10'])} / "
                f"{_fmt(fsp['empirical_coverage']['0.50'])} / "
                f"{_fmt(fsp['empirical_coverage']['0.90'])} · "
                f"80% interval coverage {_fmt(fsp['interval_coverage'])}"
            )
            rs = fsp["risk_score"]
            l.append(
                f"- risk score mean `{_fmt(rs['mean'])}` · median `{_fmt(rs['median'])}` · "
                f"near-zero P50 `{rs['n_near_zero_p50']}`"
            )
        l.append("")
        l.append("## Dispatch evaluation (realised settlement, per strategy)")
        l.append("")
        l.append("| strategy | n days | mean cost (EUR) | median | savings vs no_bat (EUR) | export-undercut h | neutral Δ (MWh) |")
        l.append("|---|---|---|---|---|---|---|")
        for s in self.strategies:
            l.append(
                f"| {s['strategy']} | {s['n_days']} | {_fmt(s['mean_daily_cost_eur'])} | "
                f"{_fmt(s['median_daily_cost_eur'])} | "
                f"{_fmt(s['mean_daily_savings_vs_no_battery_eur'])} | "
                f"{s['total_export_undercut_hours']} | "
                f"{_fmt(s['mean_energy_neutrality_delta_mwh'])} |"
            )
        l.append("")
        l.append("## Pairwise cost comparisons (bootstrap CI, paired days)")
        l.append("")
        l.append("| A | B | mean Δ (A−B) EUR | 95% CI | n days |")
        l.append("|---|---|---|---|---|")
        for c in self.comparisons:
            l.append(
                f"| {c['A']} | {c['B']} | {_fmt(c['difference'])} | "
                f"[{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}] | {c['n_days']} |"
            )
        l.append("")
        l.append(
            f"- {self.comparisons[0]['note']}" if self.comparisons else ""
        )
        l.append("")
        if self.dropped_days:
            l.append("## Dropped days")
            l.append("")
            for ts, reason in sorted(self.dropped_days.items()):
                l.append(f"- `{ts}`: {reason}")
            l.append("")
        l.append("## Split (chronological, shared by all offsets)")
        l.append("")
        l.append(f"- {_fmt_split(self.split)}")
        l.append("")
        l.append("## Reproducibility")
        l.append("")
        r = self.reproducibility
        l.append(
            f"- seed `{r['seed']}`, n_boot `{r['n_boot']}`, gridpulse "
            f"`{r['gridpulse_version']}`, python `{r['python_version']}`, "
            f"generated `{r['generated_at_utc']}` UTC"
        )
        l.append("")
        return "\n".join(l)

    def write(self, out_dir, *, stem: str = "dispatch_backtest_phase4db") -> list:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{stem}.json"
        md_path = out_dir / f"{stem}.md"
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        md_path.write_text(self.to_markdown() + "\n", encoding="utf-8", newline="\n")
        return [json_path, md_path]


# ============================================================================
# small helpers
# ============================================================================
def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _parse_gold_timestamp(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        from ..ingestion.common.models import ensure_utc

        return ensure_utc(datetime.fromisoformat(str(value)))
    except (TypeError, ValueError):
        return None


def _parse_gold_float(value) -> Optional[float]:
    if value is None or str(value) == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gridpulse_version() -> str:
    try:
        from .. import __version__  # type: ignore[attr-defined]

        return str(__version__)
    except Exception:
        return "unknown"


def _python_version() -> str:
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_split(split: Mapping) -> str:
    return (
        f"train [{split['train_start']}, {split['train_end']}) · "
        f"validation [{split['validation_start']}, {split['validation_end']}) · "
        f"test [{split['test_start']}, {split['test_end']})"
    )


__all__ = [
    "DATA_STATUS",
    "PRICE_AVAILABILITY_CONVENTION",
    "ASOF_POLICY",
    "DEFAULT_FEATURES",
    "ForecastProfile",
    "ForecastProfilesResult",
    "build_forecast_profiles",
    "assemble_dispatch_input",
    "settle_day",
    "bootstrap_cost_difference_ci",
    "run_dispatch_backtest",
    "BacktestResult",
]