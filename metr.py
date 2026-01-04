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
    def doubling_time_months(self) -> float:
        """Doubling time for h50 (and h80, assuming same slope)."""
        return np.log(2) / self.slope_50

    def sample_params(self, n: int) -> np.ndarray:
        """Sample n sets of [intercept_50, slope_50, intercept_80, slope_80]."""
        mean = [self.intercept_50, self.slope_50, self.intercept_80, self.slope_80]
        return np.random.multivariate_normal(mean, self.cov_matrix, size=n)

    def sample_horizons(self, t_months: float, n: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """Sample h50 and h80 at time t (in months from base_date)."""
        params = self.sample_params(n)
        log_h50 = params[:, 0] + params[:, 1] * t_months
        log_h80 = params[:, 2] + params[:, 3] * t_months
        return np.exp(log_h50), np.exp(log_h80)

    def sample_k(self, t_months: float, n: int = 1000) -> np.ndarray:
        """Sample sigmoid steepness k at time t."""
        h50, h80 = self.sample_horizons(t_months, n)
        return np.log(0.25) / np.log(h80 / h50)

    def success_probability(self, task_min: float, t_months: float, n: int = 1000) -> np.ndarray:
        """
        Sample success probability for a task of given duration at time t.

        Returns array of n probability samples.
        """
        h50, h80 = self.sample_horizons(t_months, n)
        k = np.log(0.25) / np.log(h80 / h50)

        # Numerically stable: compute in log space to avoid overflow
        log_ratio = k * np.log(task_min / h50)
        # Clip to avoid overflow in exp
        log_ratio = np.clip(log_ratio, -50, 50)
        return 1 / (1 + np.exp(log_ratio))

    def success_probability_at_date(self, task_min: float, d: date, n: int = 1000) -> np.ndarray:
        """Sample success probability at a given date."""
        t_months = (d - self.base_date).days / 30.44
        return self.success_probability(task_min, t_months, n)


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
        f"h50 fit: log(h) = {fit.intercept_50:.4f} + {fit.slope_50:.5f} * t "
        "(R² = {fit.r_squared_50:.3f})"
    )
    print(
        f"h80 fit: log(h) = {fit.intercept_80:.4f} + {fit.slope_80:.5f} * t "
        "(R² = {fit.r_squared_80:.3f})"
    )
    print(f"Doubling time: {fit.doubling_time_months:.2f} months")
    print()

    # Sample at Jan 2026
    d = date(2026, 1, 4)
    h50, h80 = fit.sample_horizons((d - fit.base_date).days / 30.44, n=10000)
    k = fit.sample_k((d - fit.base_date).days / 30.44, n=10000)

    print(f"At {d}:")
    print(f"  h50: {np.median(h50):.0f} min ({np.median(h50) / 60:.1f} hr)")
    print(f"  h80: {np.median(h80):.0f} min ({np.median(h80) / 60:.1f} hr)")
    print(f"  k (steepness): {np.median(k):.2f}")
    print()

    # Success probability for different task lengths
    print("Success probability by task length:")
    for task_hr in [0.5, 1, 2, 4, 8, 16]:
        task_min = task_hr * 60
        probs = fit.success_probability_at_date(task_min, d, n=10000)
        print(
            f"  {task_hr:4.1f} hr: {np.median(probs) * 100:5.1f}% "
            f"[90% CI: {np.percentile(probs, 5) * 100:.1f}-{np.percentile(probs, 95) * 100:.1f}%]"
        )
