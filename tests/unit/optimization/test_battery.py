"""Battery configuration and state model tests (Phase 4D-A)."""

import math

import pytest

from gridpulse.optimization import BatteryConfig, BatteryModel


def test_battery_config_defaults():
    """BatteryConfig applies task-spec defaults correctly."""
    bat = BatteryConfig()
    assert bat.capacity_mwh == 50.0
    assert bat.max_charge_power_mw == 10.0
    assert bat.max_discharge_power_mw == 10.0
    assert bat.round_trip_efficiency == 0.90
    assert bat.soc_min == 0.10
    assert bat.soc_max == 0.90
    assert bat.initial_soc == 0.50
    # Even split of round-trip loss: both = sqrt(0.90) ≈ 0.9487
    assert bat.charge_efficiency == pytest.approx(math.sqrt(0.90))
    assert bat.discharge_efficiency == pytest.approx(math.sqrt(0.90))
    assert bat.charge_efficiency * bat.discharge_efficiency == pytest.approx(0.90)


def test_battery_config_efficiency_product_validation():
    """charge_efficiency * discharge_efficiency must equal round_trip_efficiency."""
    # Valid: product matches within tolerance.
    bat = BatteryConfig(
        round_trip_efficiency=0.90,
        charge_efficiency=0.95,
        discharge_efficiency=0.90 / 0.95,
    )
    assert bat.charge_efficiency * bat.discharge_efficiency == pytest.approx(0.90)

    # Invalid: product does not match.
    with pytest.raises(
        ValueError, match="charge_efficiency \\* discharge_efficiency must equal"
    ):
        BatteryConfig(
            round_trip_efficiency=0.90,
            charge_efficiency=0.95,
            discharge_efficiency=0.85,  # 0.95 * 0.85 = 0.8075 ≠ 0.90
        )


def test_battery_config_soc_bounds_validation():
    """SOC min/max/initial must satisfy 0 <= min <= initial <= max <= 1."""
    # Valid.
    BatteryConfig(soc_min=0.20, soc_max=0.80, initial_soc=0.50)

    # soc_min > soc_max.
    with pytest.raises(ValueError, match="soc_min .* must be <= soc_max"):
        BatteryConfig(soc_min=0.90, soc_max=0.10)

    # initial_soc out of bounds.
    with pytest.raises(ValueError, match="initial_soc .* must lie in"):
        BatteryConfig(soc_min=0.10, soc_max=0.90, initial_soc=0.95)

    # Non-finite soc_min.
    with pytest.raises(ValueError, match="soc_min must be finite"):
        BatteryConfig(soc_min=float("nan"))


def test_battery_config_power_and_capacity_validation():
    """Capacity and powers must be > 0 and finite."""
    with pytest.raises(ValueError, match="capacity_mwh must be > 0"):
        BatteryConfig(capacity_mwh=0.0)

    with pytest.raises(ValueError, match="max_charge_power_mw must be > 0"):
        BatteryConfig(max_charge_power_mw=-1.0)

    with pytest.raises(ValueError, match="max_discharge_power_mw must be > 0"):
        BatteryConfig(max_discharge_power_mw=0.0)


def test_battery_config_fraction_to_mwh_helpers():
    """Helper properties convert SOC fractions to MWh correctly."""
    bat = BatteryConfig(capacity_mwh=50.0, soc_min=0.10, soc_max=0.90, initial_soc=0.50)
    assert bat.soc_min_mwh == pytest.approx(5.0)
    assert bat.soc_max_mwh == pytest.approx(45.0)
    assert bat.initial_soc_mwh == pytest.approx(25.0)


def test_battery_model_soc_transition_equation():
    """BatteryModel.state_vector implements the exact SOC transition equation."""
    bat = BatteryConfig(
        capacity_mwh=100.0,
        round_trip_efficiency=0.90,
        initial_soc=0.50,  # 50 MWh
    )
    model = BatteryModel(bat)
    eta_c = bat.charge_efficiency
    eta_d = bat.discharge_efficiency

    # Hour 0: charge 10 MW.
    # soc[1] = 50 + 10*eta_c - 0/eta_d = 50 + 10*sqrt(0.9) ≈ 59.487 MWh.
    # Hour 1: discharge 5 MW.
    # soc[2] = 59.487 + 0 - 5/eta_d = 59.487 - 5/sqrt(0.9) ≈ 54.217 MWh.
    charge = [10.0, 0.0]
    discharge = [0.0, 5.0]
    soc = model.state_vector(charge, discharge, start_soc_mwh=50.0)

    expected_1 = 50.0 + 10.0 * eta_c
    expected_2 = expected_1 - 5.0 / eta_d
    assert soc == pytest.approx([expected_1, expected_2])


def test_battery_model_charge_adds_discharge_removes():
    """Charge increases SOC, discharge decreases SOC (sign convention)."""
    bat = BatteryConfig(capacity_mwh=100.0, initial_soc=0.50)
    model = BatteryModel(bat)
    start = bat.initial_soc_mwh  # 50 MWh

    # Charge only.
    soc_charge = model.state_vector([10.0], [0.0], start_soc_mwh=start)
    assert soc_charge[0] > start

    # Discharge only.
    soc_discharge = model.state_vector([0.0], [10.0], start_soc_mwh=start)
    assert soc_discharge[0] < start


def test_battery_model_state_vector_length_mismatch():
    """state_vector raises ValueError on charge/discharge length mismatch."""
    bat = BatteryConfig()
    model = BatteryModel(bat)
    with pytest.raises(
        ValueError, match="charge and discharge vectors must be the same length"
    ):
        model.state_vector([10.0, 5.0], [3.0])
