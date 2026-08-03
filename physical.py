"""Physical AI capability model: hardware feasibility and physical task horizons.

Extends the METR software horizon model to physical/robotic tasks, segmented
by environment type (structured vs unstructured).

Three independent limits on physical h50:
1. SW cognitive throttle (inherited via sw_coupling)
2. Physical data wall (fleet-limited real-time data collection)
3. Entropic ceiling (hard cap from environmental properties)
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

SwTrajectory = Callable[[float], float] | float


def _sw_at(sw: SwTrajectory, t_months: float) -> float:
    """Evaluate a speedup trajectory (callable) or constant at time t."""
    return sw(t_months) if callable(sw) else sw


@dataclass
class FleetConstraint:
    """Robot fleet size for physical training data collection.

    Robots collect data in real-time (~3 hrs/robot/day). Fleet grows
    logistically from ~2K to ~2M, doubling every ~18 months.

    Args:
        current_fleet: Data-collecting robots in 2026.
        max_fleet: Logistic asymptote.
        doubling_months: Fleet doubling time.
        hours_per_robot_day: Useful training data per robot per day.
    """

    current_fleet: float = 2000
    max_fleet: float = 2_000_000
    doubling_months: float = 18.0
    hours_per_robot_day: float = 3.0

    def fleet_at(self, t_months: float) -> float:
        """Fleet size at time t (months from Jan 2026)."""
        rate = np.log(2) / self.doubling_months
        odds_0 = self.current_fleet / (self.max_fleet - self.current_fleet)
        return self.max_fleet / (
            1 + (1 / odds_0) * np.exp(-rate * t_months)
        )

    def growth_factor(self, t_months: float) -> float:
        """Cumulative fleet data growth relative to t=0."""
        return self.fleet_at(t_months) / self.current_fleet

    def growth_rate(self, t_months: float) -> float:
        """Instantaneous log-growth rate d/dt ln(fleet) at time t.

        Closed form for the logistic: rate * (1 - fleet/max).
        """
        rate = np.log(2) / self.doubling_months
        return rate * (1 - self.fleet_at(t_months) / self.max_fleet)


@dataclass
class PhysicalHorizon:
    """Physical AI task horizon by environment type.

    Uses resource-gap throttling (like the SW constraint model):
    physical h50 grows at base rate, throttled by fleet data availability
    and boosted by SW coupling, capped by entropic ceilings.

    Args:
        initial_h50_structured: Starting h50 for structured envs (minutes).
        initial_h50_unstructured: Starting h50 for unstructured envs (minutes).
        doubling_months_structured: Base doubling time for structured (months).
        doubling_months_unstructured: Base doubling time for unstructured (months).
        h80_h50_ratio: Ratio of h80 to h50 (controls sigmoid steepness).
        sw_coupling: How much SW speedup accelerates the doubling rate.
        entropic_ceiling_structured: Hard cap from environment (minutes).
        entropic_ceiling_unstructured: Hard cap from environment (minutes).
        fleet: Fleet constraint for physical data wall.
    """

    initial_h50_structured: float = 60.0
    initial_h50_unstructured: float = 4.0
    doubling_months_structured: float = 8.0
    doubling_months_unstructured: float = 14.0
    h80_h50_ratio: float = 0.2
    sw_coupling: float = 0.3
    entropic_ceiling_structured: float = 60 * 24 * 60.0  # 60 days
    entropic_ceiling_unstructured: float = 18 * 60.0  # 18 hours
    fleet: FleetConstraint | None = None

    def __post_init__(self):
        if self.fleet is None:
            self.fleet = FleetConstraint()

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

    def _entropic_ceiling(self, env_type: str) -> float:
        if env_type == "structured":
            return self.entropic_ceiling_structured
        else:
            return self.entropic_ceiling_unstructured

    def horizon_at(
        self,
        t_months: float,
        env_type: str = "unstructured",
        sw_speedup: SwTrajectory = 1.0,
    ) -> float:
        """Compute h50 at time t with fleet data throttle + entropic ceiling.

        Growth rate = base_rate * fleet_throttle * (1 + sw_coupling * log(sw))
        Then capped by entropic ceiling.

        The fleet throttle is calibrated so throttle=1 at t=0: the base
        doubling times are the currently observed rates, which slow as
        fleet data growth falls below its current rate.

        Args:
            t_months: Months from base date (Jan 2026).
            env_type: "structured" or "unstructured".
            sw_speedup: Software speedup over time — a callable mapping
                months to speedup, or a constant. The trajectory is
                integrated, so past growth uses past (smaller) speedups.

        Returns:
            h50 in minutes.
        """
        if t_months <= 0:
            return self._initial_h50(env_type)

        # Integrate growth rate over time with fleet throttle
        n_steps = max(int(t_months), 12)
        dt = t_months / n_steps
        log_h50 = np.log(self._initial_h50(env_type))
        base_slope = self._base_slope(env_type)

        # Required fleet data rate: calibrated so throttle=1 at present
        g_required = self.fleet.growth_rate(0.0)

        for i in range(n_steps):
            t_i = (i + 0.5) * dt

            throttle = min(1.0, self.fleet.growth_rate(t_i) / g_required)

            # SW coupling boosts rate
            sw = _sw_at(sw_speedup, t_i)
            sw_boost = 1 + self.sw_coupling * np.log(max(sw, 1.0))

            # Effective growth rate this step
            g_eff = base_slope * throttle * sw_boost
            log_h50 += g_eff * dt

        h50 = np.exp(log_h50)

        # Apply entropic ceiling
        return min(h50, self._entropic_ceiling(env_type))

    def success_probability(
        self,
        task_min: float,
        t_months: float,
        env_type: str = "unstructured",
        sw_speedup: SwTrajectory = 1.0,
    ) -> float:
        """Success probability for a physical task of given duration.

        Uses the same sigmoid as METRFit: P = 1 / (1 + (d/h50)^k)
        where k = log(0.25) / log(h80/h50).
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
    """

    initial_structured: float = 0.75
    initial_unstructured: float = 0.20
    base_rate_annual: float = 0.06
    sw_design_coupling: float = 0.05

    def feasibility_at(
        self,
        t_months: float,
        env_type: str = "unstructured",
        sw_speedup: SwTrajectory = 1.0,
    ) -> float:
        """Fraction of physical tasks that are hardware-feasible at time t.

        Logistic growth with a time-varying rate: the SW design boost is
        integrated over the trajectory, so past growth uses past speedups.
        Solves dF/dt = r(t) F (1-F) via logit(F(t)) = logit(F0) + int r dt.
        """
        if env_type == "structured":
            initial = self.initial_structured
        else:
            initial = self.initial_unstructured

        logit = np.log(initial / (1 - initial))

        if t_months > 0:
            n_steps = max(int(t_months), 12)
            dt = t_months / n_steps
            for i in range(n_steps):
                t_i = (i + 0.5) * dt
                sw = _sw_at(sw_speedup, t_i)
                sw_boost = self.sw_design_coupling * np.log(max(sw, 1.0))
                annual_rate = self.base_rate_annual + sw_boost
                logit += annual_rate * dt / 12.0

        return 1 / (1 + np.exp(-logit))


def physical_automation_fraction(
    task_durations: np.ndarray,
    h50: float,
    h80_h50_ratio: float,
    hw_feasibility: float,
) -> float:
    """Compute probability-weighted automation fraction for physical tasks."""
    h80 = h50 * h80_h50_ratio
    k = np.log(0.25) / np.log(h80 / h50)

    log_ratio = k * np.log(task_durations / h50)
    log_ratio = np.clip(log_ratio, -50, 50)
    probs = 1 / (1 + np.exp(log_ratio))

    effective_probs = hw_feasibility * probs
    automated_volume = np.sum(task_durations * effective_probs)
    total_volume = np.sum(task_durations)

    return automated_volume / total_volume


def _scaled_trajectory(
    trajectory: Callable[[float], float], t_months: float, sw_target: float
) -> SwTrajectory:
    """Scale a median SW trajectory so it passes through sw_target at t.

    Scales in log space: traj_i(u) = traj(u)^alpha with
    alpha = log(sw_target) / log(traj(t)), preserving the ramp shape.
    """
    sw_med = max(trajectory(t_months), 1.0)
    if sw_med <= 1.0 or sw_target <= 1.0:
        return max(sw_target, 1.0)
    alpha = np.log(sw_target) / np.log(sw_med)
    return lambda u: max(trajectory(u), 1.0) ** alpha


def sample_physical_speedup(
    t_months: float,
    sw_speedups: np.ndarray,
    env_type: str,
    task_durations: np.ndarray,
    horizon: PhysicalHorizon | None = None,
    hardware: HardwareCapability | None = None,
    sw_trajectory: Callable[[float], float] | None = None,
) -> np.ndarray:
    """Sample physical speedup with uncertainty from SW speedup samples.

    Args:
        sw_speedups: Sampled SW speedup values at time t.
        sw_trajectory: Median SW speedup path (months -> speedup). Each
            sample's path is the trajectory log-scaled to pass through
            that sample's speedup at t. If None, each sample's speedup
            is treated as constant over history (overstates past boost).
    """
    if horizon is None:
        horizon = PhysicalHorizon()
    if hardware is None:
        hardware = HardwareCapability()

    speedups = np.empty(len(sw_speedups))
    for i, sw in enumerate(sw_speedups):
        if sw_trajectory is not None:
            sw_path: SwTrajectory = _scaled_trajectory(sw_trajectory, t_months, sw)
        else:
            sw_path = sw
        h50 = horizon.horizon_at(t_months, env_type, sw_path)
        hw_frac = hardware.feasibility_at(t_months, env_type, sw_path)
        frac = physical_automation_fraction(
            task_durations, h50, horizon.h80_h50_ratio, hw_frac
        )
        speedups[i] = 1 / (1 - frac)
    return speedups


def physical_speedup(
    t_months: float,
    sw_speedup: SwTrajectory,
    env_type: str,
    task_durations: np.ndarray,
    horizon: PhysicalHorizon | None = None,
    hardware: HardwareCapability | None = None,
) -> float:
    """Compute physical task speedup at time t (deterministic)."""
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
