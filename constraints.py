"""Resource constraints on AI capability growth.

Models four factors that limit how fast METR h50 can grow:
energy, compute availability, algorithmic efficiency, and data.

Approach: rather than deriving h50 from resources, we throttle the
observed METR growth rate when resource growth can't keep pace.
h50 grows at the unconstrained METR rate until the binding resource's
growth rate falls below what's required, then h50 growth slows
proportionally.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class EnergyConstraint:
    """Maximum power available for AI training/inference.

    Logistic growth from current cluster power toward grid-scale limit.
    """

    current_gw: float = 0.1  # ~100 MW clusters (2024)
    max_gw: float = 7.0  # grid-scale limit
    growth_rate: float = 0.6  # annual logistic rate

    def power_at(self, t_years: float) -> float:
        """Max cluster power in GW at time t (years from 2024)."""
        odds_0 = self.current_gw / (self.max_gw - self.current_gw)
        return self.max_gw / (
            1 + (1 / odds_0) * np.exp(-self.growth_rate * t_years)
        )

    def growth_factor(self, t_years: float) -> float:
        """Cumulative energy growth relative to t=0."""
        return self.power_at(t_years) / self.current_gw


@dataclass
class ComputeConstraint:
    """GPU supply and hardware efficiency, capped by energy."""

    gpu_growth_rate: float = np.log(4)  # 4x/year GPU supply
    hw_efficiency_rate: float = np.log(1.4)  # 40%/year FLOP/watt

    def growth_factor(
        self, t_years: float, energy: EnergyConstraint | None = None
    ) -> float:
        """Cumulative compute growth relative to t=0."""
        gpu_factor = np.exp(self.gpu_growth_rate * t_years)
        if energy is not None:
            energy_factor = energy.growth_factor(t_years)
            efficiency_factor = np.exp(self.hw_efficiency_rate * t_years)
            energy_limited = energy_factor * efficiency_factor
            return min(gpu_factor, energy_limited)
        return gpu_factor


@dataclass
class AlgorithmicEfficiency:
    """Algorithmic progress multiplier with diminishing returns."""

    initial_rate: float = np.log(3)  # 3x/year initially
    decay: float = 0.25  # halves every ~2.8 years

    def growth_factor(self, t_years: float) -> float:
        """Cumulative algorithmic efficiency multiplier."""
        if self.decay < 1e-10:
            return np.exp(self.initial_rate * t_years)
        exponent = (self.initial_rate / self.decay) * (
            1 - np.exp(-self.decay * t_years)
        )
        return np.exp(exponent)


@dataclass
class DataConstraint:
    """Training data availability: human (logistic cap) + synthetic."""

    human_data_max: float = 3.0  # max human data (multiple of current)
    human_growth_rate: float = 0.3  # annual logistic rate
    synth_efficiency: float = 0.1  # synthetic data worth 10% of real

    def growth_factor(self, t_years: float, effective_compute: float) -> float:
        """Cumulative data growth relative to t=0."""
        odds_0 = 1.0 / (self.human_data_max - 1.0)
        human = self.human_data_max / (
            1 + (1 / odds_0) * np.exp(-self.human_growth_rate * t_years)
        )
        synth = self.synth_efficiency * effective_compute
        baseline = 1.0 + self.synth_efficiency
        return (human + synth) / baseline


def resource_growth_rates(
    t_years: np.ndarray,
    energy: EnergyConstraint | None = None,
    compute: ComputeConstraint | None = None,
    algo: AlgorithmicEfficiency | None = None,
    data: DataConstraint | None = None,
) -> dict[str, np.ndarray]:
    """Compute instantaneous growth rates for each resource over time.

    Returns dict with keys: "compute", "data", "binding", and the
    instantaneous log-growth rate for each at every time step.
    """
    if energy is None:
        energy = EnergyConstraint()
    if compute is None:
        compute = ComputeConstraint()
    if algo is None:
        algo = AlgorithmicEfficiency()
    if data is None:
        data = DataConstraint()

    dt = np.diff(t_years)

    # Effective compute at each time
    eff_compute = np.array([
        compute.growth_factor(t, energy) * algo.growth_factor(t)
        for t in t_years
    ])

    # Data at each time
    eff_data = np.array([
        data.growth_factor(t, ec)
        for t, ec in zip(t_years, eff_compute)
    ])

    # Instantaneous log-growth rates (d/dt ln(resource))
    compute_rates = np.diff(np.log(eff_compute)) / dt
    data_rates = np.diff(np.log(eff_data)) / dt

    # Binding rate = min of the two at each step
    binding_rates = np.minimum(compute_rates, data_rates)
    binding_names = np.where(
        compute_rates <= data_rates, "compute", "data"
    )

    return {
        "compute": compute_rates,
        "data": data_rates,
        "binding": binding_rates,
        "binding_name": binding_names,
        "t_mid": (t_years[:-1] + t_years[1:]) / 2,
    }


def constrained_h50(
    t_years: np.ndarray,
    h50_start: float,
    metr_doubling_months: float,
    required_resource_doubling_months: float,
    energy: EnergyConstraint | None = None,
    compute: ComputeConstraint | None = None,
    algo: AlgorithmicEfficiency | None = None,
    data: DataConstraint | None = None,
) -> np.ndarray:
    """Compute h50 trajectory throttled by resource constraints.

    h50 grows at the METR rate as long as resources keep up.
    When resource growth falls below the required rate, h50 growth
    slows proportionally.

    Args:
        t_years: Time array (years from 2024).
        h50_start: h50 at t=0 (minutes).
        metr_doubling_months: Unconstrained METR h50 doubling time.
        required_resource_doubling_months: How fast resources must grow
            to sustain the METR rate (estimated from historical period).
        energy, compute, algo, data: Constraint configs.

    Returns:
        Array of h50 values (minutes) at each time step.
    """
    rates = resource_growth_rates(t_years, energy, compute, algo, data)

    # Unconstrained METR growth rate (per year)
    g_metr = np.log(2) / (metr_doubling_months / 12.0)

    # Required resource growth rate (per year)
    g_required = np.log(2) / (required_resource_doubling_months / 12.0)

    # Throttle: actual h50 growth = metr rate * min(1, actual/required)
    throttle = np.minimum(1.0, rates["binding"] / g_required)
    g_actual = g_metr * throttle

    # Integrate to get h50 trajectory
    dt = np.diff(t_years)
    log_h50 = np.zeros(len(t_years))
    log_h50[0] = np.log(h50_start)
    for i in range(len(dt)):
        log_h50[i + 1] = log_h50[i] + g_actual[i] * dt[i]

    return np.exp(log_h50)


def sample_ceilings(
    t_months_from_base: np.ndarray,
    h50_now: float,
    metr_doubling_months: float,
    required_resource_doubling_months: float,
    n_samples: int,
    t_now_years: float = 2.1,
) -> np.ndarray:
    """Sample time-varying h50 ceilings with probabilistic constraints.

    Samples uncertainty in: energy max, data max, synth efficiency,
    algo decay rate. Returns ceiling array (n_times, n_samples).

    Args:
        t_months_from_base: Time array in months from METR base date.
        h50_now: Current best observed h50 (minutes).
        metr_doubling_months: Unconstrained METR doubling time.
        required_resource_doubling_months: Calibrated required rate.
        n_samples: Number of Monte Carlo samples.
        t_now_years: Current time in years from 2024.

    Returns:
        Array of shape (len(t_months_from_base), n_samples) with
        ceiling values in minutes for each (time, sample).
    """
    rng = np.random.default_rng()

    # Time grid for constraint model (years from 2024)
    t_years = np.linspace(t_now_years, 14, 300)

    ceilings = np.zeros((len(t_months_from_base), n_samples))

    for s in range(n_samples):
        # Sample constraint parameters
        e = EnergyConstraint(
            max_gw=rng.lognormal(np.log(7.0), 0.3),
        )
        a = AlgorithmicEfficiency(
            decay=rng.lognormal(np.log(0.25), 0.3),
        )
        d = DataConstraint(
            human_data_max=rng.lognormal(np.log(3.0), 0.3),
            synth_efficiency=rng.lognormal(np.log(0.1), 0.5),
        )

        # Compute constrained trajectory for this sample
        h50_traj = constrained_h50(
            t_years, h50_now, metr_doubling_months,
            required_resource_doubling_months,
            e, ComputeConstraint(), a, d,
        )

        # Interpolate to the requested time grid
        for i, t_m in enumerate(t_months_from_base):
            # Convert months-from-METR-base to years-from-2024
            # METR base date ~= Mar 2023, so offset ~= -0.75 years
            t_yr = t_m / 12.0 - 0.75
            if t_yr <= t_now_years:
                # Before now: ceiling is effectively infinite
                ceilings[i, s] = 1e10
            else:
                idx = np.searchsorted(t_years, t_yr)
                idx = min(idx, len(h50_traj) - 1)
                ceilings[i, s] = h50_traj[idx]

    return ceilings
