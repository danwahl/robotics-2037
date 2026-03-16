"""Tests for resource constraint models."""

import numpy as np
import pytest

from constraints import (
    AlgorithmicEfficiency,
    ComputeConstraint,
    DataConstraint,
    EnergyConstraint,
    constrained_h50,
    resource_growth_rates,
)

# --- EnergyConstraint tests ---


def test_energy_starts_at_current():
    e = EnergyConstraint()
    assert e.power_at(0) == pytest.approx(e.current_gw, rel=0.01)


def test_energy_approaches_max():
    e = EnergyConstraint()
    assert e.power_at(50) == pytest.approx(e.max_gw, rel=0.01)


def test_energy_growth_factor_starts_at_1():
    e = EnergyConstraint()
    assert e.growth_factor(0) == pytest.approx(1.0, rel=0.01)


# --- ComputeConstraint tests ---


def test_compute_starts_at_1():
    c = ComputeConstraint()
    assert c.growth_factor(0) == pytest.approx(1.0, rel=0.01)


def test_compute_grows():
    c = ComputeConstraint()
    assert c.growth_factor(1) > c.growth_factor(0)


def test_compute_capped_by_energy():
    c = ComputeConstraint()
    e = EnergyConstraint(max_gw=0.2)
    uncapped = c.growth_factor(10, energy=None)
    capped = c.growth_factor(10, energy=e)
    assert capped < uncapped


# --- AlgorithmicEfficiency tests ---


def test_algo_starts_at_1():
    a = AlgorithmicEfficiency()
    assert a.growth_factor(0) == pytest.approx(1.0, rel=0.01)


def test_algo_diminishing_returns():
    a = AlgorithmicEfficiency()
    r1 = a.growth_factor(2) / a.growth_factor(1)
    r2 = a.growth_factor(6) / a.growth_factor(5)
    assert r2 < r1


def test_algo_bounded():
    a = AlgorithmicEfficiency()
    asymptote = np.exp(a.initial_rate / a.decay)
    assert a.growth_factor(100) == pytest.approx(asymptote, rel=0.01)


# --- DataConstraint tests ---


def test_data_starts_at_1():
    d = DataConstraint()
    assert d.growth_factor(0, 1.0) == pytest.approx(1.0, rel=0.01)


def test_data_grows():
    d = DataConstraint()
    assert d.growth_factor(5, 10.0) > d.growth_factor(0, 1.0)


def test_synth_scales_with_compute():
    d = DataConstraint()
    assert d.growth_factor(5, 100.0) > d.growth_factor(5, 1.0)


# --- resource_growth_rates tests ---


def test_growth_rates_shape():
    t = np.linspace(0, 10, 100)
    rates = resource_growth_rates(t)
    assert rates["compute"].shape == (99,)
    assert rates["data"].shape == (99,)
    assert rates["binding"].shape == (99,)


def test_growth_rates_positive_early():
    """Resources should be growing early on."""
    t = np.linspace(0, 2, 50)
    rates = resource_growth_rates(t)
    assert np.all(rates["binding"] > 0)


def test_binding_rate_decreases_over_time():
    """Binding growth rate should eventually slow as asymptotes hit."""
    t = np.linspace(0, 20, 200)
    rates = resource_growth_rates(t)
    early = np.mean(rates["binding"][:20])
    late = np.mean(rates["binding"][-20:])
    assert late < early


# --- constrained_h50 tests ---


def test_constrained_h50_starts_at_initial():
    t = np.linspace(0, 10, 100)
    h50 = constrained_h50(t, h50_start=100.0, metr_doubling_months=4.2,
                          required_resource_doubling_months=6.0)
    assert h50[0] == pytest.approx(100.0, rel=0.01)


def test_constrained_h50_grows():
    t = np.linspace(0, 5, 100)
    h50 = constrained_h50(t, h50_start=100.0, metr_doubling_months=4.2,
                          required_resource_doubling_months=6.0)
    assert h50[-1] > h50[0]


def test_constrained_h50_finite():
    t = np.linspace(0, 20, 200)
    h50 = constrained_h50(t, h50_start=100.0, metr_doubling_months=4.2,
                          required_resource_doubling_months=6.0)
    assert np.all(np.isfinite(h50))


def test_constrained_slower_than_unconstrained():
    """Constrained h50 should grow slower than pure exponential."""
    t = np.linspace(0, 15, 200)
    h50_c = constrained_h50(t, h50_start=100.0, metr_doubling_months=4.2,
                            required_resource_doubling_months=6.0)
    # Unconstrained
    g = np.log(2) / (4.2 / 12.0)
    h50_u = 100.0 * np.exp(g * t)
    # At large t, constrained should be below unconstrained
    assert h50_c[-1] < h50_u[-1]


def test_tight_energy_slows_growth():
    """Tighter energy cap should produce lower h50."""
    t = np.linspace(0, 10, 100)
    e_loose = EnergyConstraint(max_gw=20.0)
    e_tight = EnergyConstraint(max_gw=0.5)
    h50_loose = constrained_h50(t, 100.0, 4.2, 6.0, energy=e_loose)
    h50_tight = constrained_h50(t, 100.0, 4.2, 6.0, energy=e_tight)
    assert h50_tight[-1] < h50_loose[-1]
