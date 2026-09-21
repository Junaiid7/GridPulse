"""Unit tests for the residual-load definition and PSR aggregation."""

from __future__ import annotations

from gridpulse.transformation.gold.residual_load import (
    SOLAR_PSR,
    WIND_PSR,
    category_hourly,
    compute_residual,
)


def test_residual_definition_wind_and_solar() -> None:
    r = compute_residual(load_mw=100.0, wind_mw=25.0, solar_mw=15.0)
    assert r.residual_mw == 60.0
    assert r.load_mw == 100.0
    assert r.flags == ()


def test_residual_without_wind_or_solar() -> None:
    r = compute_residual(load_mw=100.0, wind_mw=None, solar_mw=None)
    assert r.residual_mw == 100.0
    assert r.flags == ("wind_missing", "solar_missing")


def test_residual_missing_wind_only() -> None:
    r = compute_residual(load_mw=100.0, wind_mw=None, solar_mw=10.0)
    assert r.residual_mw == 90.0
    assert r.flags == ("wind_missing",)


def test_residual_missing_solar_only() -> None:
    r = compute_residual(load_mw=100.0, wind_mw=20.0, solar_mw=None)
    assert r.residual_mw == 80.0
    assert r.flags == ("solar_missing",)


def test_residual_missing_load_is_none() -> None:
    r = compute_residual(load_mw=None, wind_mw=5.0, solar_mw=5.0)
    assert r.residual_mw is None
    assert r.flags == ("load_missing",)


def test_negative_residual_is_valid() -> None:
    # Solar + wind exceeding load is valid (net export); never clamped.
    r = compute_residual(load_mw=40.0, wind_mw=30.0, solar_mw=60.0)
    assert r.residual_mw == -50.0


def test_psr_mapping_constants() -> None:
    assert WIND_PSR == ("B18", "B19")
    assert SOLAR_PSR == ("B16",)


def test_category_hourly_sums_present_psrs() -> None:
    series = {
        "B18": {"h12": 10.0, "h13": 20.0},
        "B19": {"h12": 5.0, "h14": 2.0},
    }
    cat, present = category_hourly(series, ("B18", "B19"))
    assert present == 2
    assert cat == {"h12": 15.0, "h13": 20.0, "h14": 2.0}


def test_category_hourly_missing_psr_skipped() -> None:
    cat, present = category_hourly({"B18": {"h12": 7.0}}, ("B18", "B19"))
    assert present == 1
    assert cat == {"h12": 7.0}


def test_category_hourly_empty() -> None:
    cat, present = category_hourly({}, ("B18",))
    assert present == 0
    assert cat == {}
