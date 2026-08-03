"""Tests for physical AI capability model."""

import numpy as np
import pytest

from physical import (
    FleetConstraint,
    HardwareCapability,
    PhysicalHorizon,
    physical_speedup,
    sample_physical_speedup,
)

# --- FleetConstraint tests ---


def test_fleet_starts_at_current():
    f = FleetConstraint()
    assert f.fleet_at(0) == pytest.approx(f.current_fleet, rel=0.01)


def test_fleet_approaches_max():
    f = FleetConstraint()
    assert f.fleet_at(600) == pytest.approx(f.max_fleet, rel=0.01)


def test_fleet_growth_factor_starts_at_1():
    f = FleetConstraint()
    assert f.growth_factor(0) == pytest.approx(1.0, rel=0.01)


# --- PhysicalHorizon tests ---


def test_structured_horizon_greater_than_unstructured():
    ph = PhysicalHorizon()
    for t in [0, 12, 24, 60]:
        h_s = ph.horizon_at(t, "structured")
        h_u = ph.horizon_at(t, "unstructured")
        assert h_s > h_u


def test_horizon_grows_over_time():
    ph = PhysicalHorizon()
    for env in ["structured", "unstructured"]:
        h_0 = ph.horizon_at(0, env)
        h_12 = ph.horizon_at(12, env)
        h_24 = ph.horizon_at(24, env)
        assert h_12 > h_0
        assert h_24 > h_12


def test_sw_coupling_accelerates_horizon():
    ph = PhysicalHorizon()
    for env in ["structured", "unstructured"]:
        h_no_sw = ph.horizon_at(24, env, sw_speedup=1.0)
        h_with_sw = ph.horizon_at(24, env, sw_speedup=5.0)
        assert h_with_sw > h_no_sw


def test_horizon_at_t0_matches_initial():
    ph = PhysicalHorizon()
    assert ph.horizon_at(0, "structured") == pytest.approx(60.0, rel=0.02)
    assert ph.horizon_at(0, "unstructured") == pytest.approx(4.0, rel=0.02)


def test_success_probability_at_h50():
    ph = PhysicalHorizon()
    h50 = ph.horizon_at(12, "structured")
    p = ph.success_probability(h50, 12, "structured")
    assert p == pytest.approx(0.5, abs=0.01)


def test_success_probability_short_task_high():
    ph = PhysicalHorizon()
    p = ph.success_probability(0.1, 12, "structured")
    assert p > 0.95


def test_success_probability_long_task_low():
    ph = PhysicalHorizon()
    p = ph.success_probability(100000, 12, "unstructured")
    assert p < 0.05


# --- Entropic ceiling tests ---


def test_horizon_capped_by_entropic_ceiling():
    """Horizon should not exceed entropic ceiling even at large t."""
    ph = PhysicalHorizon()
    for env in ["structured", "unstructured"]:
        h_late = ph.horizon_at(600, env, sw_speedup=100.0)
        ceiling = ph._entropic_ceiling(env)
        assert h_late <= ceiling * 1.001


# --- Fleet data throttle tests ---


def test_fleet_throttle_slows_growth():
    """Tiny fleet should grow slower than large fleet."""
    ph_small = PhysicalHorizon(fleet=FleetConstraint(max_fleet=100))
    ph_large = PhysicalHorizon(fleet=FleetConstraint(max_fleet=10_000_000))
    for env in ["structured", "unstructured"]:
        h_small = ph_small.horizon_at(60, env)
        h_large = ph_large.horizon_at(60, env)
        assert h_large > h_small


# --- HardwareCapability tests ---


def test_hardware_feasibility_bounded():
    hw = HardwareCapability()
    for t in [0, 12, 60, 120, 600]:
        for env in ["structured", "unstructured"]:
            f = hw.feasibility_at(t, env)
            assert 0 <= f <= 1


def test_hardware_feasibility_non_decreasing():
    hw = HardwareCapability()
    for env in ["structured", "unstructured"]:
        prev = 0
        for t in range(0, 121, 12):
            f = hw.feasibility_at(t, env)
            assert f >= prev - 1e-10
            prev = f


def test_structured_feasibility_greater_than_unstructured():
    hw = HardwareCapability()
    for t in [0, 12, 24, 60]:
        f_s = hw.feasibility_at(t, "structured")
        f_u = hw.feasibility_at(t, "unstructured")
        assert f_s > f_u


def test_hardware_at_t0_matches_initial():
    hw = HardwareCapability()
    assert hw.feasibility_at(0, "structured") == pytest.approx(0.75, rel=0.01)
    assert hw.feasibility_at(0, "unstructured") == pytest.approx(0.20, rel=0.01)


def test_sw_accelerates_hardware():
    hw = HardwareCapability()
    for env in ["structured", "unstructured"]:
        f_no_sw = hw.feasibility_at(60, env, sw_speedup=1.0)
        f_with_sw = hw.feasibility_at(60, env, sw_speedup=10.0)
        assert f_with_sw > f_no_sw


# --- physical_speedup tests ---


def test_physical_speedup_bounded():
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=10000)
    for t in [0, 12, 24, 60, 120]:
        for env in ["structured", "unstructured"]:
            s = physical_speedup(
                t, sw_speedup=2.0, env_type=env, task_durations=tasks
            )
            assert s >= 1.0
            assert np.isfinite(s)


def test_structured_speedup_greater_than_unstructured():
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=10000)
    s_s = physical_speedup(
        24, sw_speedup=2.0, env_type="structured", task_durations=tasks
    )
    s_u = physical_speedup(
        24, sw_speedup=2.0, env_type="unstructured", task_durations=tasks
    )
    assert s_s > s_u


def test_unstructured_speedup_near_1_at_t0():
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=10000)
    s = physical_speedup(
        0, sw_speedup=1.0, env_type="unstructured", task_durations=tasks
    )
    assert s < 1.2


def test_speedup_increases_with_sw():
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=10000)
    s_low = physical_speedup(
        24, sw_speedup=1.0, env_type="structured", task_durations=tasks
    )
    s_high = physical_speedup(
        24, sw_speedup=5.0, env_type="structured", task_durations=tasks
    )
    assert s_high > s_low


# --- Monte Carlo sampling tests ---


def test_sample_physical_speedup_shape():
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=5000)
    sw_samples = np.array([1.5, 2.0, 3.0, 5.0, 10.0])
    result = sample_physical_speedup(24, sw_samples, "structured", tasks)
    assert result.shape == (5,)
    assert np.all(result >= 1.0)
    assert np.all(np.isfinite(result))


def test_sample_speedup_increases_with_sw_samples():
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=5000)
    sw_samples = np.array([1.0, 5.0, 10.0])
    result = sample_physical_speedup(24, sw_samples, "structured", tasks)
    assert result[1] > result[0]
    assert result[2] > result[1]


# --- Trajectory and calibration tests ---


def test_fleet_throttle_is_one_at_t0():
    ph = PhysicalHorizon()
    g0 = ph.fleet.growth_rate(0.0)
    assert np.isclose(min(1.0, g0 / g0), 1.0)
    # Throttle declines as fleet growth saturates
    assert ph.fleet.growth_rate(120.0) < g0


def test_initial_growth_matches_base_doubling():
    """With throttle=1 and no SW boost, h50 should double at the base rate."""
    ph = PhysicalHorizon()
    h0 = ph.horizon_at(0, "structured", 1.0)
    h1 = ph.horizon_at(ph.doubling_months_structured, "structured", 1.0)
    assert np.isclose(h1 / h0, 2.0, rtol=0.05)


def test_trajectory_below_constant_endpoint():
    """A ramping SW trajectory should yield lower h50 than applying the
    endpoint speedup as a constant over all of history."""
    ph = PhysicalHorizon()
    sw_end = 100.0
    ramp = lambda u: sw_end ** (u / 60.0)  # noqa: E731
    h_ramp = ph.horizon_at(60, "unstructured", ramp)
    h_const = ph.horizon_at(60, "unstructured", sw_end)
    assert h_ramp < h_const


def test_feasibility_trajectory_below_constant():
    hw = HardwareCapability()
    sw_end = 100.0
    ramp = lambda u: sw_end ** (u / 60.0)  # noqa: E731
    f_ramp = hw.feasibility_at(60, "unstructured", ramp)
    f_const = hw.feasibility_at(60, "unstructured", sw_end)
    assert f_ramp < f_const


def test_feasibility_constant_sw_matches_closed_form():
    """With constant sw, the integrated logit equals the closed-form logistic."""
    hw = HardwareCapability()
    sw = 5.0
    t = 36.0
    rate = hw.base_rate_annual + hw.sw_design_coupling * np.log(sw)
    logit0 = np.log(hw.initial_unstructured / (1 - hw.initial_unstructured))
    expected = 1 / (1 + np.exp(-(logit0 + rate * t / 12.0)))
    assert np.isclose(hw.feasibility_at(t, "unstructured", sw), expected, rtol=1e-6)


def test_sample_speedup_with_trajectory():
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=5000)
    sw_samples = np.array([1.0, 5.0, 10.0])
    traj = lambda u: 10.0 ** (u / 24.0)  # noqa: E731
    result = sample_physical_speedup(
        24, sw_samples, "structured", tasks, sw_trajectory=traj
    )
    assert result.shape == (3,)
    assert np.all(result >= 1.0)
    # Still monotone in the sampled endpoint speedup
    assert result[2] > result[1] > result[0]
