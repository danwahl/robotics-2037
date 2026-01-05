"""METR horizon model: fit and sample from AI task horizon data."""

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import yaml


def load_metr_data(path: str) -> pd.DataFrame:
    """Load benchmark_results.yaml into a DataFrame."""
    with open(path) as f:
        data = yaml.safe_load(f)

    rows = []
    for model_name, model_data in data["results"].items():
        release = model_data["release_date"]
        if isinstance(release, str):
            release = datetime.strptime(release, "%Y-%m-%d").date()

        metrics = model_data["metrics"]
        rows.append(
            {
                "model": model_name,
                "release_date": release,
                "p50_horizon": metrics["p50_horizon_length"]["estimate"],
                "p80_horizon": metrics["p80_horizon_length"]["estimate"],
            }
        )

    return pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)


@dataclass
class METRFit:
    """
    Fitted model for AI task success probability over time.

    Models two log-linear trends:
        log(h50) = intercept_50 + slope_50 * t
        log(h80) = intercept_80 + slope_80 * t

    If ceiling is passed to sample_horizons, uses logistic instead:
        h50 = ceiling / (1 + exp(-k*(t - t0)))
        h80 = ceiling * ratio / (1 + exp(-k*(t - t0)))
        where ratio = exp(intercept_80 - intercept_50) preserves the p80/p50 relationship

    Success probability at task duration d:
        P(success) = 1 / (1 + (d / h50)^k)
        where k = log(0.25) / log(h80 / h50)
    """

    intercept_50: float
    slope_50: float
    intercept_80: float
    slope_80: float
    cov_matrix: np.ndarray  # 4x4: [intercept_50, slope_50, intercept_80, slope_80]
    base_date: date
    r_squared_50: float
    r_squared_80: float
    residual_std_50: float
    residual_std_80: float

    @property
    def h80_h50_ratio(self) -> float:
        """Ratio of h80 to h50 at t=0 (preserved in logistic ceiling)."""
        return np.exp(self.intercept_80 - self.intercept_50)

    @property
    def doubling_time_months(self) -> float:
        """Doubling time for h50 (and h80, assuming same slope)."""
        return np.log(2) / self.slope_50

    def sample_params(self, n: int) -> np.ndarray:
        """Sample n sets of [intercept_50, slope_50, intercept_80, slope_80]."""
        mean = [self.intercept_50, self.slope_50, self.intercept_80, self.slope_80]
        return np.random.multivariate_normal(mean, self.cov_matrix, size=n)

    def sample_horizons(
        self, t_months: float, n: int = 1000, ceiling: float | np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample h50 and h80 at time t (in months from base_date).

        Args:
            t_months: Time in months from base_date
            n: Number of samples
            ceiling: Optional p50 ceiling in minutes. Can be:
                     - None: exponential (unbounded)
                     - float: fixed ceiling for all samples
                     - np.ndarray of length n: per-sample ceilings (for uncertainty)
                     p80 ceiling is derived to maintain the same ratio as exponential fit.
        """
        params = self.sample_params(n)

        # Handle ceiling array
        if ceiling is not None and not isinstance(ceiling, np.ndarray):
            ceiling = np.full(n, ceiling)

        if ceiling is None:
            # Exponential: h = exp(intercept + slope * t)
            log_h50 = params[:, 0] + params[:, 1] * t_months
            log_h80 = params[:, 2] + params[:, 3] * t_months
            return np.exp(log_h50), np.exp(log_h80)
        else:
            # Logistic: h50 = L / (1 + exp(-k*(t - t0)))
            # k = slope, t0 = ln(L/exp(intercept) - 1) / k
            L_50 = ceiling
            L_80 = ceiling * self.h80_h50_ratio  # maintain ratio

            k_50 = params[:, 1]
            t0_50 = np.log(np.maximum(L_50 / np.exp(params[:, 0]) - 1, 1e-10)) / k_50
            h50 = L_50 / (1 + np.exp(-k_50 * (t_months - t0_50)))

            k_80 = params[:, 3]
            t0_80 = np.log(np.maximum(L_80 / np.exp(params[:, 2]) - 1, 1e-10)) / k_80
            h80 = L_80 / (1 + np.exp(-k_80 * (t_months - t0_80)))

            return h50, h80

    def sample_k(
        self, t_months: float, n: int = 1000, ceiling: float | np.ndarray | None = None
    ) -> np.ndarray:
        """Sample sigmoid steepness k at time t."""
        h50, h80 = self.sample_horizons(t_months, n, ceiling=ceiling)
        return np.log(0.25) / np.log(h80 / h50)

    def success_probability(
        self,
        task_min: float,
        t_months: float,
        n: int = 1000,
        ceiling: float | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Sample success probability for a task of given duration at time t.

        Returns array of n probability samples.
        """
        h50, h80 = self.sample_horizons(t_months, n, ceiling=ceiling)
        k = np.log(0.25) / np.log(h80 / h50)

        # Numerically stable: compute in log space to avoid overflow
        log_ratio = k * np.log(task_min / h50)
        # Clip to avoid overflow in exp
        log_ratio = np.clip(log_ratio, -50, 50)
        return 1 / (1 + np.exp(log_ratio))

    def success_probability_at_date(
        self, task_min: float, d: date, n: int = 1000, ceiling: float | np.ndarray | None = None
    ) -> np.ndarray:
        """Sample success probability at a given date."""
        t_months = (d - self.base_date).days / 30.44
        return self.success_probability(task_min, t_months, n, ceiling=ceiling)


def fit_metr(df: pd.DataFrame) -> METRFit:
    """Fit log-linear models for h50 and h80 trends."""
    base_date = df["release_date"].min()
    t = np.array([(d - base_date).days / 30.44 for d in df["release_date"]])

    y_50 = np.log(df["p50_horizon"].values)
    y_80 = np.log(df["p80_horizon"].values)

    n_obs = len(t)

    # Fit h50: log(h50) = intercept_50 + slope_50 * t
    X = np.column_stack([np.ones(n_obs), t])
    beta_50, *_ = np.linalg.lstsq(X, y_50, rcond=None)
    residuals_50 = y_50 - X @ beta_50

    # Fit h80: log(h80) = intercept_80 + slope_80 * t
    beta_80, *_ = np.linalg.lstsq(X, y_80, rcond=None)
    residuals_80 = y_80 - X @ beta_80

    # Compute covariance matrix for all 4 parameters
    p = 2  # parameters per model
    mse_50 = np.sum(residuals_50**2) / (n_obs - p)
    mse_80 = np.sum(residuals_80**2) / (n_obs - p)

    # Covariance for each 2-param model
    XtX_inv = np.linalg.inv(X.T @ X)
    cov_50 = mse_50 * XtX_inv  # 2x2
    cov_80 = mse_80 * XtX_inv  # 2x2

    # Cross-covariance between h50 and h80 parameters
    # Based on correlation of residuals
    resid_corr = np.corrcoef(residuals_50, residuals_80)[0, 1]
    cross_std = np.sqrt(mse_50 * mse_80)
    cov_cross = resid_corr * cross_std * XtX_inv  # 2x2

    # Build 4x4 covariance matrix
    # Order: [intercept_50, slope_50, intercept_80, slope_80]
    cov_matrix = np.block([[cov_50, cov_cross], [cov_cross.T, cov_80]])

    # R² for each fit
    ss_res_50 = np.sum(residuals_50**2)
    ss_tot_50 = np.sum((y_50 - np.mean(y_50)) ** 2)
    r_squared_50 = 1 - ss_res_50 / ss_tot_50

    ss_res_80 = np.sum(residuals_80**2)
    ss_tot_80 = np.sum((y_80 - np.mean(y_80)) ** 2)
    r_squared_80 = 1 - ss_res_80 / ss_tot_80

    return METRFit(
        intercept_50=beta_50[0],
        slope_50=beta_50[1],
        intercept_80=beta_80[0],
        slope_80=beta_80[1],
        cov_matrix=cov_matrix,
        base_date=base_date,
        r_squared_50=r_squared_50,
        r_squared_80=r_squared_80,
        residual_std_50=np.sqrt(mse_50),
        residual_std_80=np.sqrt(mse_80),
    )


if __name__ == "__main__":
    df = load_metr_data("data/benchmark_results.yaml")
    fit = fit_metr(df)

    print(
        f"h50 fit: log(h) = {fit.intercept_50:.4f} + {fit.slope_50:.5f} * t  "
        f"(R² = {fit.r_squared_50:.3f})"
    )
    print(f"h80/h50 ratio: {fit.h80_h50_ratio:.3f}")
    print(f"Doubling time: {fit.doubling_time_months:.2f} months")
    print()

    # Compare predictions at Jan 2030
    d = date(2030, 1, 1)
    t = (d - fit.base_date).days / 30.44

    # Exponential (no ceiling)
    h50_exp, h80_exp = fit.sample_horizons(t, n=10000)

    # Logistic with fixed ceiling
    ceiling_1mo = 30 * 24 * 60  # 1 month
    h50_log, h80_log = fit.sample_horizons(t, n=10000, ceiling=ceiling_1mo)

    # Logistic with uncertain ceiling (sampled from distribution)
    import squigglepy as sq

    ceiling_dist = sq.lognorm(7 * 24 * 60, 90 * 24 * 60)  # 90% CI: 1 week to 3 months
    ceiling_samples = ceiling_dist @ 10000
    h50_unc, h80_unc = fit.sample_horizons(t, n=10000, ceiling=ceiling_samples)

    print(f"h50 at {d}:")
    print(f"  Exponential:        {np.median(h50_exp) / 60 / 24:6.1f} days")
    print(f"  Logistic (1 month): {np.median(h50_log) / 60 / 24:6.1f} days")
    print(
        f"  Logistic (uncertain): {np.median(h50_unc) / 60 / 24:6.1f} days [90% CI: "
        f"{np.percentile(h50_unc, 5) / 60 / 24:.1f}-{np.percentile(h50_unc, 95) / 60 / 24:.1f}]"
    )
