"""Dispatch input and result contract tests (Phase 4D-A)."""

from datetime import datetime, timedelta

import pytest

from gridpulse.optimization import (
    BatteryConfig,
    DispatchInput,
    DispatchResult,
)


def _make_valid_input(**overrides):
    """Helper: create a valid DispatchInput with defaults."""
    base = {
        "issue_time": datetime(2024, 1, 1, 6, 0, tzinfo=None),
        "target_times": tuple(
            datetime(2024, 1, 1, 0, 0, tzinfo=None) + timedelta(hours=h)
            for h in range(24)
        ),
        "price_eur_mwh": tuple(50.0 for _ in range(24)),
        "residual_load_mw": tuple(100.0 for _ in range(24)),
    }
    base.update(overrides)
    return DispatchInput(**base)


def test_dispatch_input_requires_24_hour_horizon():
    """target_times must contain exactly 24 hourly timestamps."""
    with pytest.raises(ValueError, match="must contain exactly 24 hourly timestamps"):
        _make_valid_input(
            target_times=tuple(
                datetime(2024, 1, 1, 0, 0, tzinfo=None) + timedelta(hours=h)
                for h in range(23)
            )
        )

    with pytest.raises(ValueError, match="must contain exactly 24 hourly timestamps"):
        _make_valid_input(
            target_times=tuple(
                datetime(2024, 1, 1, 0, 0, tzinfo=None) + timedelta(hours=h)
                for h in range(25)
            )
        )


def test_dispatch_input_rejects_non_hourly_timestamps():
    """target timestamps must be on the hour (minute==0)."""
    times = list(
        datetime(2024, 1, 1, 0, 0, tzinfo=None) + timedelta(hours=h) for h in range(24)
    )
    times[5] = datetime(2024, 1, 1, 5, 30, tzinfo=None)  # off-hour
    with pytest.raises(ValueError, match="must be on the hour"):
        _make_valid_input(target_times=tuple(times))


def test_dispatch_input_rejects_non_strictly_increasing():
    """target_times must be strictly increasing (no duplicates, no reversals)."""
    times = list(
        datetime(2024, 1, 1, 0, 0, tzinfo=None) + timedelta(hours=h) for h in range(24)
    )
    times[10] = times[9]  # duplicate
    with pytest.raises(ValueError, match="must be strictly increasing"):
        _make_valid_input(target_times=tuple(times))


def test_dispatch_input_rejects_non_finite_prices():
    """price_eur_mwh must contain only finite values."""
    prices = [50.0] * 24
    prices[7] = float("nan")
    with pytest.raises(
        ValueError, match="price_eur_mwh must contain only finite values"
    ):
        _make_valid_input(price_eur_mwh=tuple(prices))


def test_dispatch_input_rejects_negative_prices():
    """price_eur_mwh must be non-negative."""
    prices = [50.0] * 24
    prices[12] = -10.0
    with pytest.raises(ValueError, match="price_eur_mwh must be >= 0"):
        _make_valid_input(price_eur_mwh=tuple(prices))


def test_dispatch_input_rejects_non_finite_residual_load():
    """residual_load_mw must contain only finite values."""
    loads = [100.0] * 24
    loads[3] = float("inf")
    with pytest.raises(
        ValueError, match="residual_load_mw must contain only finite values"
    ):
        _make_valid_input(residual_load_mw=tuple(loads))


def test_dispatch_input_scenario_ordering_enforced():
    """P10 <= P50 <= P90 ordering is enforced at the boundary."""
    base = _make_valid_input()
    p50 = base.residual_load_mw
    p10 = tuple(r * 0.9 for r in p50)
    p90 = tuple(r * 1.1 for r in p50)

    # Valid ordering.
    DispatchInput(
        issue_time=base.issue_time,
        target_times=base.target_times,
        price_eur_mwh=base.price_eur_mwh,
        residual_load_mw=p50,
        scenario_residual_load_mw={"p10": p10, "p50": p50, "p90": p90},
    )

    # Invalid: p10 > p50 at hour 0.
    bad_p10 = list(p10)
    bad_p10[0] = p50[0] + 10.0
    with pytest.raises(ValueError, match="scenario ordering violated"):
        DispatchInput(
            issue_time=base.issue_time,
            target_times=base.target_times,
            price_eur_mwh=base.price_eur_mwh,
            residual_load_mw=p50,
            scenario_residual_load_mw={"p10": tuple(bad_p10), "p50": p50, "p90": p90},
        )


def test_dispatch_input_scenario_all_or_none():
    """scenario_residual_load_mw must provide all three keys or none."""
    base = _make_valid_input()

    # Missing key.
    with pytest.raises(ValueError, match="must provide exactly the keys"):
        DispatchInput(
            issue_time=base.issue_time,
            target_times=base.target_times,
            price_eur_mwh=base.price_eur_mwh,
            residual_load_mw=base.residual_load_mw,
            scenario_residual_load_mw={"p10": base.residual_load_mw},
        )


def test_dispatch_input_scenario_p50_must_match_residual():
    """scenario['p50'] must equal residual_load_mw."""
    base = _make_valid_input()
    p50 = base.residual_load_mw
    p10 = tuple(r * 0.9 for r in p50)
    p90 = tuple(r * 1.1 for r in p50)
    bad_p50 = tuple(r + 1.0 for r in p50)

    with pytest.raises(
        ValueError, match="scenario_residual_load_mw\\['p50'\\] must equal"
    ):
        DispatchInput(
            issue_time=base.issue_time,
            target_times=base.target_times,
            price_eur_mwh=base.price_eur_mwh,
            residual_load_mw=p50,
            scenario_residual_load_mw={"p10": p10, "p50": bad_p50, "p90": p90},
        )


def test_dispatch_input_terminal_soc_bounds():
    """terminal_soc must lie in [soc_min, soc_max]."""
    bat = BatteryConfig(soc_min=0.10, soc_max=0.90)
    base = _make_valid_input(battery=bat)

    # Valid.
    DispatchInput(**{**base.__dict__, "terminal_soc": 0.50})

    # Out of bounds.
    with pytest.raises(ValueError, match="terminal_soc .* must lie in"):
        DispatchInput(**{**base.__dict__, "terminal_soc": 0.95})


def test_dispatch_result_exposes_all_phase4db_fields():
    """DispatchResult carries all fields needed for Phase 4D-B backtesting."""
    result = DispatchResult(
        strategy="test",
        issue_time=datetime(2024, 1, 1, 6, 0, tzinfo=None),
        target_times=tuple(
            datetime(2024, 1, 1, 0, 0, tzinfo=None) + timedelta(hours=h)
            for h in range(24)
        ),
        charge_mw=tuple(0.0 for _ in range(24)),
        discharge_mw=tuple(0.0 for _ in range(24)),
        soc_mwh=tuple(25.0 for _ in range(24)),
        grid_demand_mw=tuple(100.0 for _ in range(24)),
        residual_load_mw=tuple(100.0 for _ in range(24)),
        scenario_residual_load_mw={},
        scenario_costs_eur={"weighted_total": 120000.0},
        simulated_cost_eur=120000.0,
        battery=BatteryConfig(),
        status="optimal",
        solver="test",
        message=None,
    )

    d = result.to_dict()
    assert "strategy" in d
    assert "issue_time" in d
    assert "target_times" in d
    assert "charge_mw" in d
    assert "discharge_mw" in d
    assert "soc_mwh" in d
    assert "grid_demand_mw" in d
    assert "residual_load_mw" in d
    assert "scenario_residual_load_mw" in d
    assert "scenario_costs_eur" in d
    assert "simulated_cost_eur" in d
    assert "battery" in d
    assert "status" in d
    assert "solver" in d
    assert "message" in d
    assert d["data_status"] == "FIXTURE-VERIFIED"
