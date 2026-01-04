"""METR horizon model: fit and sample from AI task horizon data."""

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd
import squigglepy as sq
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
                "p50_ci_low": metrics["p50_horizon_length"]["ci_low"],
                "p50_ci_high": metrics["p50_horizon_length"]["ci_high"],
                "p80_horizon": metrics["p80_horizon_length"]["estimate"],
                "p80_ci_low": metrics["p80_horizon_length"]["ci_low"],
                "p80_ci_high": metrics["p80_horizon_length"]["ci_high"],
            }
        )

    return pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)


@dataclass
class METRFit:
    """Fitted log-linear model: log(horizon) = intercept + slope * t"""

    intercept: float
    intercept_se: float
    slope: float
    slope_se: float
    correlation: float
    base_date: date
    r_squared: float
    residual_std: float  # std dev of residuals in log space
    horizon_type: str = "p50"  # 'p50' or 'p80'

    @property
    def doubling_time_months(self) -> float:
        return np.log(2) / self.slope

    def sample_params(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample correlated (intercept, slope) pairs using squigglepy."""
        intercept_dist = sq.norm(mean=self.intercept, sd=self.intercept_se)
        slope_dist = sq.norm(mean=self.slope, sd=self.slope_se)

        corr_matrix = [[1, self.correlation], [self.correlation, 1]]
        intercept_dist, slope_dist = sq.correlate((intercept_dist, slope_dist), corr_matrix)

        return intercept_dist @ n, slope_dist @ n

    def horizon_at(self, t_months: float, n: int = 1000) -> np.ndarray:
        """Sample horizon (in minutes) at t months from base_date."""
        intercepts, slopes = self.sample_params(n)
        log_h = intercepts + slopes * t_months
        return np.exp(log_h)

    def horizon_at_date(self, d: date, n: int = 1000) -> np.ndarray:
        """Sample horizon (in minutes) at a given date."""
        t_months = (d - self.base_date).days / 30.44
        return self.horizon_at(t_months, n)

    def horizon_at_prediction(self, t_months: float, n: int = 1000) -> np.ndarray:
        """Sample horizon with prediction interval (includes residual variance)."""
        intercepts, slopes = self.sample_params(n)
        log_h = intercepts + slopes * t_months
        # Add residual noise for prediction interval
        log_h += np.random.normal(0, self.residual_std, n)
        return np.exp(log_h)


def fit_metr(df: pd.DataFrame, horizon: str = "p50") -> METRFit:
    """Fit log-linear model to METR data.

    Args:
        df: DataFrame from load_metr_data
        horizon: 'p50' or 'p80'
    """
    if horizon not in ("p50", "p80"):
        raise ValueError(f"horizon must be 'p50' or 'p80', got {horizon!r}")

    col = f"{horizon}_horizon"
    base_date = df["release_date"].min()
    t = np.array([(d - base_date).days / 30.44 for d in df["release_date"]])
    y = np.log(df[col].values)

    # OLS: y = intercept + slope * t
    X = np.column_stack([np.ones(len(t)), t])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, slope = beta

    # Residuals and covariance
    residuals = y - X @ beta
    n, p = X.shape
    mse = np.sum(residuals**2) / (n - p)
    cov = mse * np.linalg.inv(X.T @ X)

    intercept_se = np.sqrt(cov[0, 0])
    slope_se = np.sqrt(cov[1, 1])
    correlation = cov[0, 1] / (intercept_se * slope_se)

    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot

    # Residual std (for prediction intervals)
    residual_std = np.sqrt(mse)

    return METRFit(
        intercept=intercept,
        intercept_se=intercept_se,
        slope=slope,
        slope_se=slope_se,
        correlation=correlation,
        base_date=base_date,
        r_squared=r_squared,
        residual_std=residual_std,
        horizon_type=horizon,
    )


if __name__ == "__main__":
    df = load_metr_data("data/benchmark_results.yaml")

    for horizon in ["p50", "p80"]:
        fit = fit_metr(df, horizon=horizon)
        print(f"{horizon} horizon:")
        print(f"  Fit: log(h) = {fit.intercept:.4f} + {fit.slope:.5f} * t")
        print(f"  R² = {fit.r_squared:.3f}")
        print(f"  Doubling time: {fit.doubling_time_months:.2f} months")

        h = fit.horizon_at_date(date(2026, 1, 4), n=10000)
        print(
            f"  Jan 2026: {np.median(h):.0f} min ({np.median(h) / 60:.1f} hr) "
            f"[90% CI: {np.percentile(h, 5):.0f}-{np.percentile(h, 95):.0f}]"
        )
        print()
