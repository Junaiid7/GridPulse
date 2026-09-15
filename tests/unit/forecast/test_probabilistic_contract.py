"""Phase 4C probabilistic contract: ordering enforcement, crossing,
risk score, missing-quantile handling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from gridpulse.forecast.probabilistic import (
    P50_EPS,
    QUANTILES,
    QuantileForecast,
    QuantileForecasts,
    correct_quantiles,
    detect_quantile_crossing,
    risk_score,
)

UTC = timezone.utc
ISSUE = datetime(2024, 2, 1, 6, 0, tzinfo=UTC)
TARGET = ISSUE + timedelta(hours=24)


def test_valid_triple_constructs():
    q = QuantileForecast(issue_time=ISSUE, target_time=TARGET, p10=100.0, p50=110.0, p90=130.0)
    assert q.has_quantiles
    assert q.width == pytest.approx(30.0)
    assert q.risk_score is not None


def test_quantile_ordering_is_enforced_not_silently_reordered():
    # p10 > p50 -> rejected with a clear message (never silently reordered).
    with pytest.raises(ValueError, match="quantile ordering violated"):
        QuantileForecast(issue_time=ISSUE, target_time=TARGET, p10=120.0, p50=110.0, p90=130.0)
    # p50 > p90 -> rejected too.
    with pytest.raises(ValueError, match="quantile ordering violated"):
        QuantileForecast(issue_time=ISSUE, target_time=TARGET, p10=100.0, p50=130.0, p90=120.0)
    # Equal quantiles are allowed (degenerate, but ordered).
    q = QuantileForecast(issue_time=ISSUE, target_time=TARGET, p10=110.0, p50=110.0, p90=110.0)
    assert q.has_quantiles


def test_alpha_low_high_are_checked():
    with pytest.raises(ValueError, match="alpha_low"):
        QuantileForecast(
            issue_time=ISSUE, target_time=TARGET,
            p10=1.0, p50=2.0, p90=3.0,
            alpha_low=0.9, alpha_high=0.1,
        )


def test_missing_quantiles_are_undefined_not_fabricated():
    q = QuantileForecast(issue_time=ISSUE, target_time=TARGET)
    assert not q.has_quantiles
    assert q.p10 is None and q.p50 is None and q.p90 is None
    assert q.width is None
    assert q.risk_score is None


def test_quantile_forecasts_container_length():
    qs = QuantileForecasts(
        rows=[
            QuantileForecast(issue_time=ISSUE, target_time=TARGET, p10=1, p50=2, p90=3),
            QuantileForecast(issue_time=TARGET, target_time=TARGET + timedelta(hours=24)),
        ]
    )
    assert len(qs) == 2
    assert qs.quantiles == QUANTILES


def test_detect_quantile_crossing():
    assert not detect_quantile_crossing(1.0, 2.0, 3.0)
    assert detect_quantile_crossing(3.0, 2.0, 4.0)          # p10 > p50
    assert detect_quantile_crossing(1.0, 4.0, 3.0)          # p50 > p90
    assert detect_quantile_crossing(3.0, 2.0, 1.0)          # fully reversed
    # Undefined triples are not a crossing.
    assert not detect_quantile_crossing(None, 2.0, 3.0)
    assert not detect_quantile_crossing(1.0, None, 3.0)


def test_correct_quantiles_sorts_ascending():
    lo, mid, hi, corrected = correct_quantiles(3.0, 1.0, 2.0)
    assert (lo, mid, hi) == (1.0, 2.0, 3.0)
    assert corrected is True
    # Already ordered -> unchanged.
    assert correct_quantiles(1.0, 2.0, 3.0) == (1.0, 2.0, 3.0, False)
    # Partial ties keep 3 distinct outputs after sort.
    assert correct_quantiles(2.0, 1.0, 2.0)[:3] == (1.0, 2.0, 2.0)


def test_correct_quantiles_leaves_undefined_unchanged():
    assert correct_quantiles(None, 2.0, 3.0) == (None, 2.0, 3.0, False)
    assert correct_quantiles(1.0, None, 3.0) == (1.0, None, 3.0, False)


def test_correct_quantiles_guarantees_ordering_on_all_inputs():
    import itertools

    for triple in itertools.product([0.0, 1.0, 5.0, 10.0], repeat=3):
        lo, mid, hi, _ = correct_quantiles(*triple)
        assert lo <= mid <= hi


def test_risk_score_formula_and_positive_sign():
    # (P90-P10) / |P50|
    assert risk_score(100.0, 200.0, 300.0) == pytest.approx(1.0)
    assert risk_score(80.0, 200.0, 120.0) == pytest.approx(0.2)
    # For an ordered triple (p10<=p50<=p90) the interval spread is non-negative,
    # so an all-negative target still yields a positive relative indicator
    # thanks to |P50| in the denominator.
    assert risk_score(-300.0, -200.0, -100.0) == pytest.approx(1.0)


@pytest.mark.parametrize("p50", [0.0, 1e-12, -1e-12])
def test_risk_score_guards_near_zero_p50(p50):
    # Division-by-zero / unstable amplification is never silently propagated.
    assert risk_score(100.0, p50, 200.0) is None


def test_risk_score_missing_quantile_is_none():
    assert risk_score(None, 200.0, 300.0) is None
    assert risk_score(100.0, None, 300.0) is None
    assert risk_score(100.0, 200.0, None) is None


def test_p50_eps_threshold_consistency():
    # At exactly the epsilon threshold, |p50| < eps -> None.
    assert abs(P50_EPS) < abs(1e-8)
    assert risk_score(100.0, P50_EPS, 200.0) is None