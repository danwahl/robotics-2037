"""Tests for METR horizon model."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from metr import METRFit, fit_metr

# --- Fixtures ---


@pytest.fixture
def sample_df():
    """Synthetic METR data with known parameters.

    h50 = exp(0.1 * t), h80 = exp(0.1 * t - 1) = h50 / e
    So h80/h50 = 1/e ≈ 0.368, and k = log(0.25) / log(0.368) ≈ 1.39
    """
    base = date(2020, 1, 1)
    days_per_month = 30.44
    t_months = [0, 6, 12, 18, 24]

    dates = []
    for t in t_months:
        days = int(t * days_per_month)
        dates.append(date.fromordinal(base.toordinal() + days))

    h50 = [np.exp(0.1 * t) for t in t_months]
    h80 = [np.exp(0.1 * t - 1) for t in t_months]  # h80 = h50 / e

    return pd.DataFrame(
        {
            "model": [f"model_{i}" for i in range(len(t_months))],
            "release_date": dates,
            "p50_horizon": h50,
            "p80_horizon": h80,
        }
    )


@pytest.fixture
def sample_fit(sample_df):
    """Fit from synthetic data."""
    return fit_metr(sample_df)


# --- Tests for fit_metr ---


def test_fit_metr_slope_50(sample_fit):
    """Fit recovers correct slope for h50."""
    assert sample_fit.slope_50 == pytest.approx(0.1, rel=0.05)


def test_fit_metr_slope_80(sample_fit):
    """Fit recovers correct slope for h80 (same as h50)."""
    assert sample_fit.slope_80 == pytest.approx(0.1, rel=0.05)


def test_fit_metr_intercept_50(sample_fit):
    """Fit recovers correct intercept for h50."""
    assert sample_fit.intercept_50 == pytest.approx(0.0, abs=0.1)


def test_fit_metr_intercept_80(sample_fit):
    """Fit recovers correct intercept for h80 (shifted by -1)."""
    assert sample_fit.intercept_80 == pytest.approx(-1.0, abs=0.1)


def test_fit_metr_r_squared(sample_fit):
    """Perfect exponential data should give R² ≈ 1."""
    assert sample_fit.r_squared_50 > 0.99
    assert sample_fit.r_squared_80 > 0.99


def test_fit_metr_base_date(sample_fit):
    """Base date should be earliest date in data."""
    assert sample_fit.base_date == date(2020, 1, 1)


def test_fit_metr_cov_matrix_shape(sample_fit):
    """Covariance matrix should be 4x4."""
    assert sample_fit.cov_matrix.shape == (4, 4)


def test_fit_metr_cov_matrix_symmetric(sample_fit):
    """Covariance matrix should be symmetric."""
    assert np.allclose(sample_fit.cov_matrix, sample_fit.cov_matrix.T)


# --- Tests for METRFit.sample_horizons ---


def test_sample_horizons_h50_at_t0(sample_fit):
    """At t=0, h50 should be exp(intercept_50) ≈ 1."""
    np.random.seed(42)
    h50, h80 = sample_fit.sample_horizons(t_months=0, n=1000)
    assert np.median(h50) == pytest.approx(1.0, rel=0.1)


def test_sample_horizons_h80_at_t0(sample_fit):
    """At t=0, h80 should be exp(intercept_80) ≈ 1/e ≈ 0.368."""
    np.random.seed(42)
    h50, h80 = sample_fit.sample_horizons(t_months=0, n=1000)
    assert np.median(h80) == pytest.approx(1 / np.e, rel=0.1)


def test_sample_horizons_grow_over_time(sample_fit):
    """Horizons should grow over time."""
    np.random.seed(42)
    h50_0, h80_0 = sample_fit.sample_horizons(t_months=0, n=1000)
    h50_12, h80_12 = sample_fit.sample_horizons(t_months=12, n=1000)
    assert np.median(h50_12) > np.median(h50_0)
    assert np.median(h80_12) > np.median(h80_0)


# --- Tests for METRFit.sample_k ---


def test_sample_k_value(sample_fit):
    """k should be approximately log(0.25) / log(1/e) ≈ 1.39."""
    np.random.seed(42)
    k = sample_fit.sample_k(t_months=0, n=1000)
    expected_k = np.log(0.25) / np.log(1 / np.e)  # ≈ 1.39
    assert np.median(k) == pytest.approx(expected_k, rel=0.1)


def test_sample_k_consistent_over_time(sample_fit):
    """k should be roughly constant if h50 and h80 have same slope."""
    np.random.seed(42)
    k_0 = sample_fit.sample_k(t_months=0, n=1000)
    k_12 = sample_fit.sample_k(t_months=12, n=1000)
    assert np.median(k_0) == pytest.approx(np.median(k_12), rel=0.2)


# --- Tests for METRFit.success_probability ---


def test_success_probability_at_h50(sample_fit):
    """Success probability at task = h50 should be ~50%."""
    np.random.seed(42)
    h50, _ = sample_fit.sample_horizons(t_months=12, n=1)
    task_min = h50[0]
    probs = sample_fit.success_probability(task_min, t_months=12, n=1000)
    assert np.median(probs) == pytest.approx(0.5, abs=0.1)


def test_success_probability_at_h80(sample_fit):
    """Success probability at task = h80 should be ~80%."""
    np.random.seed(42)
    _, h80 = sample_fit.sample_horizons(t_months=12, n=1)
    task_min = h80[0]
    probs = sample_fit.success_probability(task_min, t_months=12, n=1000)
    assert np.median(probs) == pytest.approx(0.8, abs=0.1)


def test_success_probability_short_task():
    """Very short tasks should have high success probability."""
    fit = METRFit(
        intercept_50=0.0,
        slope_50=0.1,
        intercept_80=-1.0,
        slope_80=0.1,
        cov_matrix=np.eye(4) * 0.001,  # tiny uncertainty
        base_date=date(2020, 1, 1),
        r_squared_50=0.99,
        r_squared_80=0.99,
        residual_std_50=0.01,
        residual_std_80=0.01,
    )
    np.random.seed(42)
    # At t=12, h50 ≈ exp(1.2) ≈ 3.3 min. A 0.1 min task should be very easy.
    probs = fit.success_probability(task_min=0.1, t_months=12, n=1000)
    assert np.median(probs) > 0.95


def test_success_probability_long_task():
    """Very long tasks should have low success probability."""
    fit = METRFit(
        intercept_50=0.0,
        slope_50=0.1,
        intercept_80=-1.0,
        slope_80=0.1,
        cov_matrix=np.eye(4) * 0.001,  # tiny uncertainty
        base_date=date(2020, 1, 1),
        r_squared_50=0.99,
        r_squared_80=0.99,
        residual_std_50=0.01,
        residual_std_80=0.01,
    )
    np.random.seed(42)
    # At t=12, h50 ≈ 3.3 min. A 1000 min task should be very hard.
    probs = fit.success_probability(task_min=1000, t_months=12, n=1000)
    assert np.median(probs) < 0.05


def test_success_probability_improves_over_time():
    """Same task should have higher success probability later."""
    fit = METRFit(
        intercept_50=0.0,
        slope_50=0.1,
        intercept_80=-1.0,
        slope_80=0.1,
        cov_matrix=np.eye(4) * 0.001,
        base_date=date(2020, 1, 1),
        r_squared_50=0.99,
        r_squared_80=0.99,
        residual_std_50=0.01,
        residual_std_80=0.01,
    )
    np.random.seed(42)
    probs_early = fit.success_probability(task_min=10, t_months=0, n=1000)
    probs_late = fit.success_probability(task_min=10, t_months=24, n=1000)
    assert np.median(probs_late) > np.median(probs_early)
