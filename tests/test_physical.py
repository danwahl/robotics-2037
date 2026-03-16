"""Tests for physical AI capability model."""

import numpy as np
import pytest

from physical import (
    HardwareCapability,
    PhysicalHorizon,
    physical_speedup,
    sample_physical_speedup,
)

# --- PhysicalHorizon tests ---


def test_structured_horizon_greater_than_unstructured():
    """Structured environments should have higher h50 at all times."""
    ph = PhysicalHorizon()
    for t in [0, 12, 24, 60]:
        h_s = ph.horizon_at(t, "structured")
        h_u = ph.horizon_at(t, "unstructured")
        assert h_s > h_u


def test_horizon_grows_over_time():
    """Horizons should increase over time."""
    ph = PhysicalHorizon()
    for env in ["structured", "unstructured"]:
        h_0 = ph.horizon_at(0, env)
        h_12 = ph.horizon_at(12, env)
        h_24 = ph.horizon_at(24, env)
        assert h_12 > h_0
        assert h_24 > h_12


def test_sw_coupling_accelerates_horizon():
    """Higher software speedup should increase physical horizon."""
    ph = PhysicalHorizon()
    for env in ["structured", "unstructured"]:
        h_no_sw = ph.horizon_at(24, env, sw_speedup=1.0)
        h_with_sw = ph.horizon_at(24, env, sw_speedup=5.0)
        assert h_with_sw > h_no_sw


def test_horizon_at_t0_matches_initial():
    """At t=0, horizon should equal initial value."""
    ph = PhysicalHorizon()
    assert ph.horizon_at(0, "structured") == pytest.approx(60.0, rel=0.01)
    assert ph.horizon_at(0, "unstructured") == pytest.approx(4.0, rel=0.01)


def test_success_probability_at_h50():
    """Success probability at task = h50 should be ~50%."""
    ph = PhysicalHorizon()
    h50 = ph.horizon_at(12, "structured")
    p = ph.success_probability(h50, 12, "structured")
    assert p == pytest.approx(0.5, abs=0.01)


def test_success_probability_short_task_high():
    """Very short tasks should have high success probability."""
    ph = PhysicalHorizon()
    p = ph.success_probability(0.1, 12, "structured")
    assert p > 0.95


def test_success_probability_long_task_low():
    """Very long tasks should have low success probability."""
    ph = PhysicalHorizon()
    p = ph.success_probability(100000, 12, "unstructured")
    assert p < 0.05


# --- HardwareCapability tests ---


def test_hardware_feasibility_bounded():
    """Feasibility should always be in [0, 1]."""
    hw = HardwareCapability()
    for t in [0, 12, 60, 120, 600]:
        for env in ["structured", "unstructured"]:
            f = hw.feasibility_at(t, env)
            assert 0 <= f <= 1


def test_hardware_feasibility_non_decreasing():
    """Feasibility should not decrease over time."""
    hw = HardwareCapability()
    for env in ["structured", "unstructured"]:
        prev = 0
        for t in range(0, 121, 12):
            f = hw.feasibility_at(t, env)
            assert f >= prev - 1e-10
            prev = f


def test_structured_feasibility_greater_than_unstructured():
    """Structured should have higher feasibility at all times."""
    hw = HardwareCapability()
    for t in [0, 12, 24, 60]:
        f_s = hw.feasibility_at(t, "structured")
        f_u = hw.feasibility_at(t, "unstructured")
        assert f_s > f_u


def test_hardware_at_t0_matches_initial():
    """At t=0, feasibility should equal initial value."""
    hw = HardwareCapability()
    assert hw.feasibility_at(0, "structured") == pytest.approx(0.75, rel=0.01)
    assert hw.feasibility_at(0, "unstructured") == pytest.approx(0.20, rel=0.01)


def test_sw_accelerates_hardware():
    """Software speedup should improve hardware feasibility."""
    hw = HardwareCapability()
    for env in ["structured", "unstructured"]:
        f_no_sw = hw.feasibility_at(60, env, sw_speedup=1.0)
        f_with_sw = hw.feasibility_at(60, env, sw_speedup=10.0)
        assert f_with_sw > f_no_sw


# --- physical_speedup tests ---


def test_physical_speedup_bounded():
    """Speedup should be finite and >= 1 at all time points."""
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=10000)
    for t in [0, 12, 24, 60, 120]:
        for env in ["structured", "unstructured"]:
            s = physical_speedup(t, sw_speedup=2.0, env_type=env, task_durations=tasks)
            assert s >= 1.0
            assert np.isfinite(s)


def test_structured_speedup_greater_than_unstructured():
    """Structured environments should show higher speedup."""
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=10000)
    s_s = physical_speedup(24, sw_speedup=2.0, env_type="structured", task_durations=tasks)
    s_u = physical_speedup(24, sw_speedup=2.0, env_type="unstructured", task_durations=tasks)
    assert s_s > s_u


def test_unstructured_speedup_near_1_at_t0():
    """At t=0 with no SW speedup, unstructured speedup should be modest."""
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=10000)
    s = physical_speedup(0, sw_speedup=1.0, env_type="unstructured", task_durations=tasks)
    assert s < 1.2  # very modest automation at t=0


def test_speedup_increases_with_sw():
    """Higher software speedup should increase physical speedup."""
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=10000)
    s_low = physical_speedup(
        24, sw_speedup=1.0, env_type="structured", task_durations=tasks
    )
    s_high = physical_speedup(
        24, sw_speedup=5.0, env_type="structured", task_durations=tasks
    )
    assert s_high > s_low


# --- Ceiling tests ---


def test_horizon_saturates_at_ceiling():
    """Horizon should approach ceiling at large t."""
    ph = PhysicalHorizon()
    for env in ["structured", "unstructured"]:
        h_late = ph.horizon_at(600, env, sw_speedup=1.0)
        ceiling = ph._ceiling(env)
        assert h_late == pytest.approx(ceiling, rel=0.01)


def test_ceiling_matches_initial_at_t0():
    """Ceiling shouldn't change t=0 values significantly."""
    ph = PhysicalHorizon()
    assert ph.horizon_at(0, "structured") == pytest.approx(60.0, rel=0.02)
    assert ph.horizon_at(0, "unstructured") == pytest.approx(4.0, rel=0.02)


# --- Monte Carlo sampling tests ---


def test_sample_physical_speedup_shape():
    """Sampled speedups should have correct shape."""
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=5000)
    sw_samples = np.array([1.5, 2.0, 3.0, 5.0, 10.0])
    result = sample_physical_speedup(
        24, sw_samples, "structured", tasks
    )
    assert result.shape == (5,)
    assert np.all(result >= 1.0)
    assert np.all(np.isfinite(result))


def test_sample_speedup_increases_with_sw_samples():
    """Higher SW speedup samples should give higher physical speedup."""
    np.random.seed(42)
    tasks = np.random.lognormal(mean=np.log(30), sigma=1.5, size=5000)
    sw_samples = np.array([1.0, 5.0, 10.0])
    result = sample_physical_speedup(
        24, sw_samples, "structured", tasks
    )
    assert result[1] > result[0]
    assert result[2] > result[1]
