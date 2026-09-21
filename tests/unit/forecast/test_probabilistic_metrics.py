"""Phase 4C probabilistic metrics: pinball loss, empirical coverage,
interval coverage, interval width / sharpness, aggregate metrics."""

from __future__ import annotations

import pytest

from gridpulse.forecast.evaluate import (
    empirical_coverage,
    interval_coverage,
    interval_width_stats,
    pinball_loss,
    probabilistic_metrics,
)


def test_pinball_loss_is_exact():
    # Sample: y=0, q=1, alpha=0.5 -> |0-1| = 1  (d<0 => d*(alpha-1) = -1*-0.5 = 0.5)
    #         y=2, q=1, alpha=0.5 -> (2-1)*0.5 = 0.5
    assert pinball_loss([0.0, 2.0], [1.0, 1.0], 0.5) == pytest.approx(0.5)
    # Asymmetric quantile: y=5, q=3, alpha=0.9 -> (5-3)*0.9 = 1.8
    assert pinball_loss([5.0], [3.0], 0.9) == pytest.approx(1.8)
    # y=3, q=5, alpha=0.9 -> (3-5)*(0.9-1) = 0.2
    assert pinball_loss([3.0], [5.0], 0.9) == pytest.approx(0.2)


def test_pinball_never_negative_lower_is_better():
    import random

    rng = random.Random(0)
    ys = [rng.uniform(0, 100) for _ in range(50)]
    for alpha in (0.1, 0.5, 0.9):
        for q in (10.0, 50.0):
            loss = pinball_loss(ys, [q] * len(ys), alpha)
            assert loss is not None and loss >= 0.0


def test_pinball_skips_none_entries():
    assert pinball_loss([1.0, None, 3.0], [2.0, 2.0, None], 0.5) == pytest.approx(
        pinball_loss([1.0, 3.0], [2.0, 2.0], 0.5)
    )
    assert pinball_loss([None, None], [1.0, 1.0], 0.5) is None


def test_pinball_length_mismatch_rejected():
    with pytest.raises(ValueError, match="length mismatch"):
        pinball_loss([1.0, 2.0], [1.0], 0.5)


def test_empirical_coverage_basic():
    actuals = [9.0, 11.0, 12.0, 13.0, 14.0]
    # q=10 counts actuals <= 10 -> 1/5 = 0.20
    assert empirical_coverage(actuals, [10.0] * 5) == pytest.approx(0.20)
    # q=13 counts actuals <= 13 -> 4/5 = 0.80
    assert empirical_coverage(actuals, [13.0] * 5) == pytest.approx(0.80)
    assert empirical_coverage(
        [1.0, None],
        [2.0, 2.0],
    ) == pytest.approx(1.0)
    assert empirical_coverage([None, None], [1.0, 1.0]) is None


def test_interval_coverage_basic():
    actuals = [5.0, 8.0, 10.0, 12.0, 15.0]
    lo, hi = 4.0, 11.0  # inside: 5, 8, 10 -> 3/5
    assert interval_coverage(actuals, [lo] * 5, [hi] * 5) == pytest.approx(0.60)
    assert interval_coverage([1.0, None], [0.0, 0.0], [2.0, 2.0]) == pytest.approx(1.0)
    assert interval_coverage([None], [0.0], [2.0]) is None


def test_interval_width_stats_exact():
    lo = [0.0, 0.0, 0.0]
    hi = [2.0, 4.0, 6.0]  # widths 2, 4, 6
    stats = interval_width_stats(lo, hi)
    assert stats["mean"] == pytest.approx(4.0)
    assert stats["median"] == pytest.approx(4.0)
    assert stats["min"] == pytest.approx(2.0)
    assert stats["max"] == pytest.approx(6.0)
    assert stats["n"] == 3
    # population std of [2,4,6] = sqrt(8/3)
    assert stats["std"] == pytest.approx((8.0 / 3.0) ** 0.5)


def test_interval_width_stats_empty_and_even():
    empty = interval_width_stats([None, None], [1.0, 1.0])
    assert empty["n"] == 0 and empty["mean"] is None and empty["median"] is None
    even = interval_width_stats([0.0, 0.0, 0.0, 0.0], [0.0, 10.0, 20.0, 30.0])
    assert even["median"] == pytest.approx(15.0)


def test_probabilistic_metrics_aggregate():
    actuals = [10.0, 20.0, 30.0, 40.0, 50.0]
    p10 = [9.0, 19.0, 29.0, 39.0, 49.0]  # always below -> cov 0
    p50 = [10.0, 20.0, 30.0, 40.0, 50.0]  # at/equal -> cov 1.0
    p90 = [95.0, 95.0, 95.0, 95.0, 95.0]  # always above -> cov 1.0
    m = probabilistic_metrics(actuals, p10, p50, p90)
    assert m["n_triples"] == 5
    assert m["empirical_coverage"]["0.10"] == pytest.approx(0.0)
    assert m["empirical_coverage"]["0.50"] == pytest.approx(1.0)
    assert m["empirical_coverage"]["0.90"] == pytest.approx(1.0)
    assert m["interval_coverage"] == pytest.approx(1.0)  # p10<=a<=p90 everywhere
    assert m["nominal_quantiles"] == {"0.10": 0.1, "0.50": 0.5, "0.90": 0.9}
    assert m["nominal_interval_coverage"] == 0.80
    assert m["interval_width"]["n"] == 5
    assert m["pinball"]["0.50"] is not None


def test_probabilistic_metrics_skips_any_missing_triple():
    actuals = [10.0, 20.0, 30.0]
    p10 = [9.0, None, 29.0]
    p50 = [10.5, 20.0, 30.0]
    p90 = [90.0, 90.0, None]
    m = probabilistic_metrics(actuals, p10, p50, p90)
    # Only row 0 fully valid: a=10, lo=9, mid=10.5, hi=90.
    assert m["n_triples"] == 1
    assert m["empirical_coverage"]["0.10"] == pytest.approx(0.0)  # 10 <= 9? no
    assert m["empirical_coverage"]["0.50"] == pytest.approx(1.0)  # 10 <= 10.5? yes
    assert m["empirical_coverage"]["0.90"] == pytest.approx(1.0)  # 10 <= 90? yes
    assert m["interval_coverage"] == pytest.approx(1.0)  # 9<=10<=90


def test_probabilistic_metrics_all_missing_returns_empty_keys():
    m = probabilistic_metrics([None], [None], [None], [None])
    assert m["n_triples"] == 0
    assert m["pinball"]["0.10"] is None
    assert m["empirical_coverage"]["0.50"] is None
    assert m["interval_coverage"] is None
    assert m["interval_width"]["n"] == 0
    assert m["risk_score"]["mean"] is None


def test_probabilistic_metrics_length_mismatch_rejected():
    with pytest.raises(ValueError, match="same length"):
        probabilistic_metrics([1.0, 2.0], [1.0], [1.0], [1.0])


def test_probabilistic_metrics_risk_score_and_near_zero():
    actuals = [10.0, 20.0]
    p10 = [9.0, 0.0]
    p50 = [10.0, 0.0]  # second P50 near zero -> excluded from score
    p90 = [11.0, 0.0]
    m = probabilistic_metrics(actuals, p10, p50, p90)
    rs = m["risk_score"]
    assert rs["n_p50_valid"] == 2
    assert rs["n_near_zero_p50"] == 1
    assert rs["mean"] == pytest.approx((11.0 - 9.0) / 10.0)  # only first triple
    assert m["n_triples"] == 2
