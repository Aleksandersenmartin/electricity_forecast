"""
Evaluation framework for electricity price forecasting.

Provides metrics computation, comparison tables, and diagnostic plots
for comparing forecasting methods against naive baselines.

Usage:
    from src.models.evaluate import compute_metrics, comparison_table, plot_forecast
"""

import logging
import time
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    naive_pred: pd.Series | None = None,
) -> dict[str, float]:
    """Compute forecasting evaluation metrics.

    Args:
        y_true: Actual values (timezone-aware DatetimeIndex).
        y_pred: Predicted values (same index as y_true).
        naive_pred: Optional naive baseline predictions for skill score.

    Returns:
        Dict with keys: mae, rmse, mape, directional_accuracy,
        peak_hour_mae, skill_score (if naive_pred given).
    """
    # Align indices and drop NaNs
    aligned = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()
    if aligned.empty:
        logger.warning("No overlapping non-NaN values for metrics")
        return {}

    y_t = aligned["true"]
    y_p = aligned["pred"]

    errors = y_t - y_p

    # MAE
    mae = np.mean(np.abs(errors))

    # RMSE
    rmse = np.sqrt(np.mean(errors ** 2))

    # MAPE (skip zero-price hours to avoid division by zero)
    nonzero_mask = y_t.abs() > 0.01
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs(errors[nonzero_mask] / y_t[nonzero_mask])) * 100
    else:
        mape = np.nan

    # Directional accuracy (% of hours where predicted direction matches actual)
    if len(y_t) > 1:
        actual_direction = y_t.diff().iloc[1:]
        pred_direction = y_p.diff().iloc[1:]
        same_direction = (actual_direction * pred_direction) > 0
        directional_accuracy = same_direction.mean() * 100
    else:
        directional_accuracy = np.nan

    # Peak hour MAE (hours 8-20)
    if hasattr(y_t.index, "hour"):
        peak_mask = (y_t.index.hour >= 8) & (y_t.index.hour <= 20)
        if peak_mask.sum() > 0:
            peak_mae = np.mean(np.abs(errors[peak_mask]))
        else:
            peak_mae = np.nan
    else:
        peak_mae = np.nan

    metrics: dict[str, float] = {
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape": round(mape, 2) if not np.isnan(mape) else np.nan,
        "directional_accuracy": round(directional_accuracy, 1),
        "peak_hour_mae": round(peak_mae, 3) if not np.isnan(peak_mae) else np.nan,
    }

    # Skill score vs naive (1 - MAE_model / MAE_naive)
    if naive_pred is not None:
        naive_aligned = pd.DataFrame({"true": y_true, "naive": naive_pred}).dropna()
        if not naive_aligned.empty:
            naive_mae = np.mean(np.abs(naive_aligned["true"] - naive_aligned["naive"]))
            if naive_mae > 0:
                metrics["skill_score"] = round(1.0 - mae / naive_mae, 3)
            else:
                metrics["skill_score"] = np.nan
        else:
            metrics["skill_score"] = np.nan

    return metrics


# ---------------------------------------------------------------------------
# 2. Comparison Table
# ---------------------------------------------------------------------------

def comparison_table(
    results: list[dict[str, Any]],
) -> pd.DataFrame:
    """Build a ranked comparison table from forecasting results.

    Args:
        results: List of dicts with keys:
            - name (str): Method name
            - metrics (dict): Output of compute_metrics()
            - fit_time (float): Fit time in seconds (optional)

    Returns:
        DataFrame ranked by MAE (ascending) with one row per method.
    """
    rows = []
    for r in results:
        row = {"Method": r["name"]}
        row.update(r.get("metrics", {}))
        if "fit_time" in r:
            row["fit_time_s"] = round(r["fit_time"], 1)
        rows.append(row)

    df = pd.DataFrame(rows)

    if "mae" in df.columns:
        df = df.sort_values("mae", ascending=True).reset_index(drop=True)
        df.index = df.index + 1  # 1-based ranking
        df.index.name = "Rank"

    return df


# ---------------------------------------------------------------------------
# 3. Forecast Overlay Plot
# ---------------------------------------------------------------------------

def plot_forecast(
    y_true: pd.Series,
    forecasts_dict: dict[str, pd.Series],
    title: str = "Forecast Comparison",
    zone: str = "",
    figsize: tuple[int, int] = (16, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Overlay actual vs predicted for multiple forecasting methods.

    Args:
        y_true: Actual values.
        forecasts_dict: Dict mapping method name to predicted Series.
        title: Plot title.
        zone: Zone name (for subtitle).
        figsize: Figure size.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(y_true.index, y_true.values, color="black", linewidth=1.0,
            alpha=0.8, label="Actual", zorder=5)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown", "tab:pink", "tab:cyan"]

    for i, (name, y_pred) in enumerate(forecasts_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(y_pred.index, y_pred.values, color=color, linewidth=0.8,
                alpha=0.7, label=name)

    ax.set_xlabel("Date")
    ax.set_ylabel("EUR/MWh")
    ax.set_title(f"{title}" + (f" — {zone}" if zone else ""))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved forecast plot to %s", save_path)

    return fig


# ---------------------------------------------------------------------------
# 4. Residual Diagnostics
# ---------------------------------------------------------------------------

def plot_residuals(
    y_true: pd.Series,
    y_pred: pd.Series,
    method_name: str = "Model",
    figsize: tuple[int, int] = (16, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """3-panel residual diagnostic plot.

    Panel 1: Residual histogram with normal fit.
    Panel 2: ACF of residuals (should be white noise).
    Panel 3: Q-Q plot (should follow diagonal).

    Args:
        y_true: Actual values.
        y_pred: Predicted values.
        method_name: Name for plot title.
        figsize: Figure size.
        save_path: If given, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    residuals = (y_true - y_pred).dropna()

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(f"Residual Diagnostics — {method_name}", fontsize=13, fontweight="bold")

    # Panel 1: Histogram
    ax = axes[0]
    ax.hist(residuals, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
    # Normal distribution overlay
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=1.5, label=f"N({mu:.1f}, {sigma:.1f})")
    ax.set_xlabel("Residual (EUR/MWh)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: ACF
    ax = axes[1]
    plot_acf(residuals, ax=ax, lags=min(72, len(residuals) // 2 - 1),
             alpha=0.05, title="ACF of Residuals")

    # Panel 3: QQ plot
    ax = axes[2]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved residual plot to %s", save_path)

    return fig
