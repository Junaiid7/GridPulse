"""Synthetic dispatch input fixtures for Phase 4D-A testing.

Builds :class:`DispatchInput` instances from the synthetic electricity fixture
(``tests/support/synthetic_electricity.py``), reusing its price/residual data
while clearly labeling the P10/P90 spread as synthetic.

DATA STATUS: FIXTURE-VERIFIED (synthetic inputs only, no real ENTSO-E data).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from gridpulse.optimization import BatteryConfig, DispatchInput

from .synthetic_electricity import build_synthetic_window

UTC = timezone.utc


def _ensure_utc(dt: datetime) -> datetime:
    """Normalise to aware UTC or reject naive datetimes."""
    if dt.tzinfo is None:
        raise ValueError(f"datetime must be timezone-aware UTC; got naive {dt!r}")
    return dt.astimezone(UTC)


def synthetic_dispatch_input(
    when: datetime | None = None,
    battery: BatteryConfig | None = None,
) -> DispatchInput:
    """Build a :class:`DispatchInput` from the synthetic electricity fixture.

    Parameters
    ----------
    when:
        Start of the 24-hour dispatch horizon (aware UTC). Defaults to
        2024-02-01 00:00 UTC.
    battery:
        Battery configuration. Defaults to the generic 50 MWh / 10 MW battery.

    Returns
    -------
    A validated :class:`DispatchInput` with:

    - ``issue_time``: day-before 06:00 UTC (when a day-ahead forecast would
      have been issued).
    - ``target_times``: 24 hourly timestamps from ``when`` to ``when+23h``.
    - ``price_eur_mwh``: synthetic day-ahead prices from the fixture
      (``day_ahead_price_eur_mwh``).
    - ``residual_load_mw``: P50 / system residual load from the fixture.
    - ``scenario_residual_load_mw``: synthetic P10/P50/P90 spread (P50 ±10%
      synthetic uncertainty band, **not** from a real forecast model).

    DATA STATUS: FIXTURE-VERIFIED (synthetic inputs).
    """
    if when is None:
        when = datetime(2024, 2, 1, 0, 0, tzinfo=UTC)  # aware UTC default
    when = _ensure_utc(when)
    if battery is None:
        battery = BatteryConfig()

    fixture = build_synthetic_window(when)
    target_times = [when + timedelta(hours=h) for h in range(24)]
    issue_time = when - timedelta(hours=18)  # day-before 06:00 UTC

    prices = []
    residual = []
    for t in target_times:
        signals = fixture.signals_at(t)
        prices.append(signals["day_ahead_price_eur_mwh"])
        residual.append(signals["residual_load_mw"])

    # Synthetic P10/P90 spread: P50 ±10% (labeled synthetic).
    p50 = tuple(residual)
    p10 = tuple(r * 0.90 for r in residual)
    p90 = tuple(r * 1.10 for r in residual)

    return DispatchInput(
        issue_time=issue_time,
        target_times=tuple(target_times),
        price_eur_mwh=tuple(prices),
        residual_load_mw=p50,
        scenario_residual_load_mw={"p10": p10, "p50": p50, "p90": p90},
        battery=battery,
    )


def tiny_pattern(hours: int = 24) -> dict:
    """Hand-checkable low→high price ramp + constant residual.

    Returns a dict with:

    - ``issue_time``: 2024-01-01 00:00 UTC.
    - ``target_times``: ``hours`` hourly timestamps from 2024-01-01 06:00 UTC.
    - ``price_eur_mwh``: linear ramp from 10.0 to 10.0 + hours - 1 EUR/MWh.
    - ``residual_load_mw``: constant 100.0 MW (no net demand variation, pure
      price arbitrage signal).

    Useful for behavioral tests: optimal strategy should charge in early cheap
    hours, discharge in later expensive hours, no scenario complexity.
    """
    issue_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    target_times = tuple(
        datetime(2024, 1, 1, 6, 0, tzinfo=UTC) + timedelta(hours=h)
        for h in range(hours)
    )
    price_eur_mwh = tuple(10.0 + float(h) for h in range(hours))
    residual_load_mw = tuple(100.0 for _ in range(hours))
    return {
        "issue_time": issue_time,
        "target_times": target_times,
        "price_eur_mwh": price_eur_mwh,
        "residual_load_mw": residual_load_mw,
    }


__all__ = ["synthetic_dispatch_input", "tiny_pattern"]
