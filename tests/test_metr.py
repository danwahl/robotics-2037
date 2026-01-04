"""Tests for METR horizon model."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import squigglepy as sq

from metr import METRFit, fit_metr

# --- Fixtures ---


@pytest.fixture
def sample_df():
    """Synthetic METR data with known parameters."""
    # True params: intercept=0, slope=0.1 (log scale)
    # So horizon = exp(0.1 * t), doubling time = ln(2)/0.1 ≈ 6.93 months
    t_months = [0, 6, 12, 18, 24]
    horizons = [np.exp(0.1 * t) for t in t_months]

    return pd.DataFrame(
        {
            "model": [f"model_{i}" for i in range(len(t_months))],
            "release_date": [
                date(2020, 1 + t // 12 * 12, 1 + (t % 12)) for t in [0, 6, 12, 18, 24]
            ],
            "p50_horizon": horizons,
            "p50_ci_low": [h * 0.8 for h in horizons],
            "p50_ci_high": [h * 1.2 for h in horizons],
            "p80_horizon": [h * 0.5 for h in horizons],
            "p80_ci_low": [h * 0.4 for h in horizons],
            "p80_ci_high": [h * 0.6 for h in horizons],
        }
    )


@pytest.fixture
def sample_df_exact():
    """Synthetic data with exact dates for precise t calculation."""
    days_per_month = 30.44
    t_months = [0, 6, 12, 18, 24]

    dates = [date(2020, 1, 1)]
    for t in t_months[1:]:
        days = int(t * days_per_month)
        d = date(2020, 1, 1)
        dates.append(date.fromordinal(d.toordinal() + days))

    horizons = [np.exp(0.1 * t) for t in t_months]

    return pd.DataFrame(
        {
            "model": [f"model_{i}" for i in range(len(t_months))],
            "release_date": dates,
            "p50_horizon": horizons,
            "p50_ci_low": [h * 0.8 for h in horizons],
            "p50_ci_high": [h * 1.2 for h in horizons],
            "p80_horizon": [h * 0.5 for h in horizons],
            "p80_ci_low": [h * 0.4 for h in horizons],
            "p80_ci_high": [h * 0.6 for h in horizons],
        }
    )


# --- Tests for fit_metr ---


def test_fit_metr_slope(sample_df_exact):
    """Fit recovers approximately correct slope."""
    fit = fit_metr(sample_df_exact)
    assert fit.slope == pytest.approx(0.1, rel=0.05)


def test_fit_metr_intercept(sample_df_exact):
    """Fit recovers approximately correct intercept."""
    fit = fit_metr(sample_df_exact)
    assert fit.intercept == pytest.approx(0.0, abs=0.1)


def test_fit_metr_r_squared(sample_df_exact):
    """Perfect exponential data should give R² ≈ 1."""
    fit = fit_metr(sample_df_exact)
    assert fit.r_squared > 0.99


def test_fit_metr_doubling_time(sample_df_exact):
    """Doubling time should be ln(2)/slope ≈ 6.93 months."""
    fit = fit_metr(sample_df_exact)
    expected = np.log(2) / 0.1
    assert fit.doubling_time_months == pytest.approx(expected, rel=0.05)


def test_fit_metr_base_date(sample_df_exact):
    """Base date should be earliest date in data."""
    fit = fit_metr(sample_df_exact)
    assert fit.base_date == date(2020, 1, 1)


def test_fit_metr_residual_std(sample_df_exact):
    """Perfect exponential data should have near-zero residual std."""
    fit = fit_metr(sample_df_exact)
    assert fit.residual_std < 0.01


def test_fit_metr_horizon_type_p50(sample_df_exact):
    """Default horizon type should be p50."""
    fit = fit_metr(sample_df_exact)
    assert fit.horizon_type == "p50"


def test_fit_metr_horizon_type_p80(sample_df_exact):
    """Can fit p80 horizon."""
    fit = fit_metr(sample_df_exact, horizon="p80")
    assert fit.horizon_type == "p80"


def test_fit_metr_invalid_horizon(sample_df_exact):
    """Invalid horizon type should raise ValueError."""
    with pytest.raises(ValueError):
        fit_metr(sample_df_exact, horizon="p90")


# --- Tests for METRFit.sample_params ---


def test_sample_params_correlation():
    """Sampled params should have approximately correct correlation."""
    sq.set_seed(42)
    fit = METRFit(
        intercept=0.0,
        intercept_se=1.0,
        slope=0.1,
        slope_se=0.01,
        correlation=-0.9,
        base_date=date(2020, 1, 1),
        r_squared=0.99,
        residual_std=0.1,
        horizon_type="p50",
    )
    intercepts, slopes = fit.sample_params(10000)
    empirical_corr = np.corrcoef(intercepts, slopes)[0, 1]
    assert empirical_corr == pytest.approx(-0.9, abs=0.05)


def test_sample_params_means():
    """Sampled params should have approximately correct means."""
    sq.set_seed(42)
    fit = METRFit(
        intercept=-3.0,
        intercept_se=0.5,
        slope=0.1,
        slope_se=0.01,
        correlation=-0.9,
        base_date=date(2020, 1, 1),
        r_squared=0.99,
        residual_std=0.1,
        horizon_type="p50",
    )
    intercepts, slopes = fit.sample_params(10000)
    assert np.mean(intercepts) == pytest.approx(-3.0, abs=0.05)
    assert np.mean(slopes) == pytest.approx(0.1, abs=0.005)


# --- Tests for METRFit.horizon_at ---


def test_horizon_at_t0():
    """At t=0, horizon should be exp(intercept)."""
    sq.set_seed(42)
    fit = METRFit(
        intercept=2.0,
        intercept_se=0.01,  # small SE for tight test
        slope=0.1,
        slope_se=0.001,
        correlation=0.0,
        base_date=date(2020, 1, 1),
        r_squared=0.99,
        residual_std=0.01,
        horizon_type="p50",
    )
    h = fit.horizon_at(t_months=0, n=1000)
    expected = np.exp(2.0)
    assert np.median(h) == pytest.approx(expected, rel=0.05)


def test_horizon_at_grows():
    """Horizon should grow with time."""
    sq.set_seed(42)
    fit = METRFit(
        intercept=0.0,
        intercept_se=0.1,
        slope=0.1,
        slope_se=0.01,
        correlation=-0.9,
        base_date=date(2020, 1, 1),
        r_squared=0.99,
        residual_std=0.1,
        horizon_type="p50",
    )
    h0 = fit.horizon_at(t_months=0, n=1000)
    h12 = fit.horizon_at(t_months=12, n=1000)
    assert np.median(h12) > np.median(h0)


def test_horizon_at_date():
    """horizon_at_date should give same result as horizon_at with correct t."""
    sq.set_seed(42)
    fit = METRFit(
        intercept=0.0,
        intercept_se=0.1,
        slope=0.1,
        slope_se=0.01,
        correlation=0.0,
        base_date=date(2020, 1, 1),
        r_squared=0.99,
        residual_std=0.1,
        horizon_type="p50",
    )

    # 1 year = ~12 months
    sq.set_seed(42)
    h_by_date = fit.horizon_at_date(date(2021, 1, 1), n=1000)

    t_months = (date(2021, 1, 1) - date(2020, 1, 1)).days / 30.44
    sq.set_seed(42)
    h_by_months = fit.horizon_at(t_months, n=1000)

    assert np.median(h_by_date) == pytest.approx(np.median(h_by_months), rel=0.01)


def test_prediction_interval_wider_than_confidence():
    """Prediction interval should be wider than confidence interval."""
    sq.set_seed(42)
    fit = METRFit(
        intercept=0.0,
        intercept_se=0.1,
        slope=0.1,
        slope_se=0.01,
        correlation=0.0,
        base_date=date(2020, 1, 1),
        r_squared=0.99,
        residual_std=0.5,  # substantial residual variance
        horizon_type="p50",
    )

    n = 5000
    h_ci = fit.horizon_at(t_months=12, n=n)
    h_pred = fit.horizon_at_prediction(t_months=12, n=n)

    ci_width = np.percentile(h_ci, 95) - np.percentile(h_ci, 5)
    pred_width = np.percentile(h_pred, 95) - np.percentile(h_pred, 5)

    assert pred_width > ci_width
