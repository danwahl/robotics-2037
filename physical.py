"""Physical AI capability model: hardware feasibility and physical task horizons.

Extends the METR software horizon model to physical/robotic tasks, segmented
by environment type (structured vs unstructured).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PhysicalHorizon:
    """Physical AI task horizon by environment type.

    Models the h50 (50% success) horizon for physical tasks, analogous to
    METR's software task horizon. Starts lower than METR, with a slower
    doubling time that accelerates via software speedup coupling.

    Args:
        initial_h50_structured: Starting h50 for structured envs (minutes).
        initial_h50_unstructured: Starting h50 for unstructured envs (minutes).
        doubling_months_structured: Base doubling time for structured (months).
        doubling_months_unstructured: Base doubling time for unstructured (months).
        h80_h50_ratio: Ratio of h80 to h50 (controls sigmoid steepness).
        sw_coupling: How much SW speedup accelerates the doubling rate (0-1).
        ceiling_structured: Logistic ceiling for structured h50 (minutes).
        ceiling_unstructured: Logistic ceiling for unstructured h50 (minutes).
    """

    initial_h50_structured: float = 60.0
    initial_h50_unstructured: float = 4.0
    doubling_months_structured: float = 8.0
    doubling_months_unstructured: float = 14.0
    h80_h50_ratio: float = 0.2
    sw_coupling: float = 0.3
    ceiling_structured: float = 60 * 24 * 60.0  # 60 days
    ceiling_unstructured: float = 18 * 60.0  # 18 hours

    def _base_slope(self, env_type: str) -> float:
        """Log-linear slope (per month) for a given environment."""
        if env_type == "structured":
            return np.log(2) / self.doubling_months_structured
        else:
            return np.log(2) / self.doubling_months_unstructured

    def _initial_h50(self, env_type: str) -> float:
        if env_type == "structured":
            return self.initial_h50_structured
        else:
            return self.initial_h50_unstructured

    def _ceiling(self, env_type: str) -> float:
        if env_type == "structured":
            return self.ceiling_structured
        else:
            return self.ceiling_unstructured

    def horizon_at(
        self,
        t_months: float,
        env_type: str = "unstructured",
        sw_speedup: float = 1.0,
    ) -> float:
        """Compute h50 at time t, boosted by software speedup.

        Software coupling accelerates the *rate* of physical AI improvement:
        effective_slope = base_slope * (1 + sw_coupling * log(sw_speedup))

        A logistic ceiling caps h50 at a maximum value per environment.

        Args:
            t_months: Months from base date (Jan 2026).
            env_type: "structured" or "unstructured".
            sw_speedup: Software speedup factor at time t (1.0 = no speedup).

        Returns:
            h50 in minutes.
        """
        base_slope = self._base_slope(env_type)
        # SW coupling accelerates the doubling rate itself
        effective_slope = base_slope * (
            1 + self.sw_coupling * np.log(max(sw_speedup, 1.0))
        )

        h50_0 = self._initial_h50(env_type)
        L = self._ceiling(env_type)

        # Logistic: L / (1 + exp(-k*(t - t0)))
        # Constrained to match h50_0 at t=0 and initial slope
        k = effective_slope
        t0 = np.log(max(L / h50_0 - 1, 1e-10)) / k
        return L / (1 + np.exp(-k * (t_months - t0)))

    def success_probability(
        self,
        task_min: float,
        t_months: float,
        env_type: str = "unstructured",
        sw_speedup: float = 1.0,
    ) -> float:
        """Success probability for a physical task of given duration.

        Uses the same sigmoid as METRFit: P = 1 / (1 + (d/h50)^k)
        where k = log(0.25) / log(h80/h50).

        Args:
            task_min: Task duration in minutes.
            t_months: Months from base date.
            env_type: "structured" or "unstructured".
            sw_speedup: Software speedup factor at time t.

        Returns:
            Success probability [0, 1].
        """
        h50 = self.horizon_at(t_months, env_type, sw_speedup)
        h80 = h50 * self.h80_h50_ratio
        k = np.log(0.25) / np.log(h80 / h50)

        log_ratio = k * np.log(task_min / h50)
        log_ratio = np.clip(log_ratio, -50, 50)
        return 1 / (1 + np.exp(log_ratio))


@dataclass
class HardwareCapability:
    """Fraction of physical tasks that are hardware-feasible.

    Models the physical capability of robots (dexterity, mobility, endurance,
    perception, strength) as a single aggregate fraction that improves over
    time via logistic growth.

    Args:
        initial_structured: Starting feasibility for structured environments.
        initial_unstructured: Starting feasibility for unstructured envs.
        base_rate_annual: Baseline annual improvement rate.
        sw_design_coupling: How much log(sw_speedup) boosts improvement.
    """

    initial_structured: float = 0.75
    initial_unstructured: float = 0.20
    base_rate_annual: float = 0.06
    sw_design_coupling: float = 0.05

    def feasibility_at(
        self,
        t_months: float,
        env_type: str = "unstructured",
        sw_speedup: float = 1.0,
    ) -> float:
        """Fraction of physical tasks that are hardware-feasible at time t.

        Args:
            t_months: Months from base date (Jan 2026).
            env_type: "structured" or "unstructured".
            sw_speedup: Software speedup factor at time t.

        Returns:
            Feasibility fraction [0, 1].
        """
        if env_type == "structured":
            initial = self.initial_structured
        else:
            initial = self.initial_unstructured

        t_years = t_months / 12.0
        sw_boost = self.sw_design_coupling * np.log(max(sw_speedup, 1.0))
        annual_rate = self.base_rate_annual + sw_boost

        # Logistic growth toward 1.0
        odds_0 = initial / (1 - initial)
        return 1 / (1 + (1 / odds_0) * np.exp(-annual_rate * t_years))


def physical_automation_fraction(
    task_durations: np.ndarray,
    h50: float,
    h80_h50_ratio: float,
    hw_feasibility: float,
) -> float:
    """Compute probability-weighted automation fraction for physical tasks.

    Args:
        task_durations: Array of task durations in minutes.
        h50: Physical AI h50 horizon in minutes.
        h80_h50_ratio: Ratio of h80 to h50.
        hw_feasibility: Fraction of tasks that are hardware-feasible.

    Returns:
        Effective automation fraction (volume-weighted).
    """
    h80 = h50 * h80_h50_ratio
    k = np.log(0.25) / np.log(h80 / h50)

    log_ratio = k * np.log(task_durations / h50)
    log_ratio = np.clip(log_ratio, -50, 50)
    probs = 1 / (1 + np.exp(log_ratio))

    # Hardware feasibility gates all tasks equally
    effective_probs = hw_feasibility * probs
    automated_volume = np.sum(task_durations * effective_probs)
    total_volume = np.sum(task_durations)

    return automated_volume / total_volume


def sample_physical_speedup(
    t_months: float,
    sw_speedups: np.ndarray,
    env_type: str,
    task_durations: np.ndarray,
    horizon: PhysicalHorizon | None = None,
    hardware: HardwareCapability | None = None,
) -> np.ndarray:
    """Sample physical speedup with uncertainty from SW speedup samples.

    Args:
        t_months: Months from base date (Jan 2026).
        sw_speedups: Array of SW speedup samples (n_samples,).
        env_type: "structured" or "unstructured".
        task_durations: Array of sampled physical task durations (minutes).
        horizon: PhysicalHorizon config (uses defaults if None).
        hardware: HardwareCapability config (uses defaults if None).

    Returns:
        Array of speedup samples (n_samples,).
    """
    if horizon is None:
        horizon = PhysicalHorizon()
    if hardware is None:
        hardware = HardwareCapability()

    speedups = np.empty(len(sw_speedups))
    for i, sw in enumerate(sw_speedups):
        h50 = horizon.horizon_at(t_months, env_type, sw)
        hw_frac = hardware.feasibility_at(t_months, env_type, sw)
        frac = physical_automation_fraction(
            task_durations, h50, horizon.h80_h50_ratio, hw_frac
        )
        speedups[i] = 1 / (1 - frac)
    return speedups


def physical_speedup(
    t_months: float,
    sw_speedup: float,
    env_type: str,
    task_durations: np.ndarray,
    horizon: PhysicalHorizon | None = None,
    hardware: HardwareCapability | None = None,
) -> float:
    """Compute physical task speedup at time t (deterministic).

    Args:
        t_months: Months from base date (Jan 2026).
        sw_speedup: Software speedup factor at time t.
        env_type: "structured" or "unstructured".
        task_durations: Array of sampled physical task durations (minutes).
        horizon: PhysicalHorizon config (uses defaults if None).
        hardware: HardwareCapability config (uses defaults if None).

    Returns:
        Speedup factor (1.0 = no speedup).
    """
    if horizon is None:
        horizon = PhysicalHorizon()
    if hardware is None:
        hardware = HardwareCapability()

    h50 = horizon.horizon_at(t_months, env_type, sw_speedup)
    hw_frac = hardware.feasibility_at(t_months, env_type, sw_speedup)

    frac = physical_automation_fraction(
        task_durations, h50, horizon.h80_h50_ratio, hw_frac
    )

    return 1 / (1 - frac)
