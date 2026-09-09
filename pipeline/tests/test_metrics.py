"""Checks for the G14 statistics, against values computed by hand."""

import numpy as np
import pytest

from energy_gap.metrics import cvrmse, nmbe, r2, resid_stats

Y = np.array([10.0, 20.0, 30.0, 40.0])
YHAT = np.array([12.0, 18.0, 33.0, 39.0])


def test_cvrmse_matches_hand_calculation():
    # sqrt(18/4) = 2.12132, over mean 25, times 100
    assert cvrmse(Y, YHAT) == pytest.approx(8.48528, abs=1e-4)


def test_nmbe_matches_hand_calculation():
    # residuals sum to -2; 100 * -2 / (4 * 25)
    assert nmbe(Y, YHAT) == pytest.approx(-2.0, abs=1e-9)


def test_nmbe_keeps_its_sign():
    """Over-prediction must read negative, under-prediction positive."""
    over = np.array([11.0, 21.0, 31.0])
    under = np.array([9.0, 19.0, 29.0])
    truth = np.array([10.0, 20.0, 30.0])
    assert nmbe(truth, over) < 0
    assert nmbe(truth, under) > 0


def test_degrees_of_freedom_inflate_the_error():
    """Spending parameters shrinks the denominator, so the statistic rises."""
    assert cvrmse(Y, YHAT, n_params=2) > cvrmse(Y, YHAT, n_params=0)


def test_perfect_prediction_is_zero_error():
    assert cvrmse(Y, Y) == pytest.approx(0.0)
    assert nmbe(Y, Y) == pytest.approx(0.0)
    assert r2(Y, Y) == pytest.approx(1.0)


def test_r2_of_the_mean_is_zero():
    """Predicting the mean every time explains none of the variance."""
    flat = np.full_like(Y, Y.mean())
    assert r2(Y, flat) == pytest.approx(0.0)


def test_resid_stats_detects_autocorrelation():
    """Alternating residuals are strongly negatively autocorrelated."""
    rng = np.random.default_rng(0)
    truth = rng.normal(100, 10, 500)

    alternating = truth + np.tile([5.0, -5.0], 250)
    _, rho_alt = resid_stats(truth, alternating)
    assert rho_alt < -0.9

    independent = truth + rng.normal(0, 5, 500)
    _, rho_ind = resid_stats(truth, independent)
    assert abs(rho_ind) < 0.2


def test_resid_sigma_recovers_known_spread():
    rng = np.random.default_rng(1)
    truth = np.full(5000, 100.0)
    noisy = truth + rng.normal(0, 7.0, 5000)
    sigma, _ = resid_stats(truth, noisy)
    assert sigma == pytest.approx(7.0, rel=0.05)