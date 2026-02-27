"""Multi-method anomaly detection for electricity market data.

Detects price spikes, forecast anomalies, and multivariate outliers
using statistical and machine learning methods.

Usage:
    from src.anomaly.detector import (
        detect_price_spikes,
        detect_forecast_anomalies,
        detect_multi_target_anomalies,
        regime_anomalies,
    )
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_price_spikes(
    prices: pd.Series,
    method: str = "rolling_zscore",
    threshold: float = 3.0,
    rolling_window: int = 720,
) -> pd.DataFrame:
    """Detect price spikes using various statistical methods.

    Args:
        prices: Price series (EUR/MWh) with DatetimeIndex.
        method: Detection method — "zscore", "iqr", "rolling_zscore".
        threshold: Detection threshold (std deviations for zscore, IQR
            multiplier for iqr).
        rolling_window: Window size for rolling methods (hours).

    Returns:
        DataFrame with columns: price, is_spike, spike_score, method.
    """
    result = pd.DataFrame({"price": prices})
    result["is_spike"] = False
    result["spike_score"] = 0.0
    result["method"] = method

    if method == "zscore":
        mean = prices.mean()
        std = prices.std()
        z_scores = (prices - mean) / std
        result["spike_score"] = z_scores.abs()
        result["is_spike"] = z_scores.abs() > threshold

    elif method == "iqr":
        q1 = prices.quantile(0.25)
        q3 = prices.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        result["spike_score"] = np.maximum(
            (prices - upper) / iqr,
            (lower - prices) / iqr,
        ).clip(lower=0)
        result["is_spike"] = (prices < lower) | (prices > upper)

    elif method == "rolling_zscore":
        rolling_mean = prices.rolling(rolling_window, min_periods=168).mean()
        rolling_std = prices.rolling(rolling_window, min_periods=168).std()
        z_scores = (prices - rolling_mean) / rolling_std
        result["spike_score"] = z_scores.abs()
        result["is_spike"] = z_scores.abs() > threshold

    else:
        raise ValueError(f"Unknown method: {method}. Use 'zscore', 'iqr', or 'rolling_zscore'")

    n_spikes = result["is_spike"].sum()
    pct = n_spikes / len(result) * 100
    logger.info(
        "detect_price_spikes(%s): %d spikes (%.2f%%) in %d hours",
        method, n_spikes, pct, len(result),
    )

    return result


def detect_forecast_anomalies(
    actuals: pd.Series,
    predictions: pd.Series,
    threshold_factor: float = 3.0,
) -> pd.DataFrame:
    """Flag hours where forecast error is abnormally large.

    Args:
        actuals: Actual values.
        predictions: Predicted values.
        threshold_factor: Flag errors exceeding threshold_factor × MAE.

    Returns:
        DataFrame with columns: actual, predicted, error, abs_error,
        is_anomaly, error_ratio.
    """
    aligned = pd.DataFrame({
        "actual": actuals,
        "predicted": predictions,
    }).dropna()

    aligned["error"] = aligned["actual"] - aligned["predicted"]
    aligned["abs_error"] = aligned["error"].abs()

    mae = aligned["abs_error"].mean()
    threshold = threshold_factor * mae

    aligned["is_anomaly"] = aligned["abs_error"] > threshold
    aligned["error_ratio"] = aligned["abs_error"] / mae

    n_anomalies = aligned["is_anomaly"].sum()
    logger.info(
        "detect_forecast_anomalies: %d anomalies (|error| > %.1f × MAE = %.1f)",
        n_anomalies, threshold_factor, threshold,
    )

    return aligned


def detect_multi_target_anomalies(
    targets_df: pd.DataFrame,
    contamination: float = 0.02,
    random_state: int = 42,
) -> pd.DataFrame:
    """Detect multivariate anomalies across multiple target variables.

    Uses Isolation Forest to find hours where the combination of target
    values is unusual (e.g., high price + low demand + high export).

    Args:
        targets_df: DataFrame with one column per target variable.
        contamination: Expected fraction of anomalies (0.01–0.10).
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with original columns plus: anomaly_score, is_anomaly.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    # Drop NaN rows
    clean = targets_df.dropna()
    if clean.empty:
        return pd.DataFrame()

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean)

    # Fit Isolation Forest
    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    labels = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)

    result = clean.copy()
    result["anomaly_score"] = -scores  # Higher = more anomalous
    result["is_anomaly"] = labels == -1

    n_anomalies = result["is_anomaly"].sum()
    logger.info(
        "detect_multi_target_anomalies: %d anomalies in %d hours (%.1f%%)",
        n_anomalies, len(result), n_anomalies / len(result) * 100,
    )

    return result


def regime_anomalies(
    prices: pd.Series,
    regime_probs: pd.DataFrame | None = None,
    k_regimes: int = 2,
    max_train_size: int = 4000,
) -> pd.DataFrame:
    """Detect anomalous regime transitions in price data.

    Fits a Markov Switching model (or uses pre-computed regime probs)
    and flags unexpected transitions between normal and spike regimes.

    Args:
        prices: Price series (EUR/MWh).
        regime_probs: Pre-computed regime probabilities (optional).
            If None, fits MarkovSwitchingForecaster.
        k_regimes: Number of regimes (default 2).
        max_train_size: Max samples for model fitting.

    Returns:
        DataFrame with columns: price, regime, regime_prob, is_transition,
        transition_surprise (entropy-based).
    """
    from src.models.forecasters import MarkovSwitchingForecaster

    result = pd.DataFrame({"price": prices})

    if regime_probs is None:
        # Fit Markov Switching model
        ms = MarkovSwitchingForecaster(
            name="MS",
            horizon=24,
            k_regimes=k_regimes,
            max_train_size=max_train_size,
        )
        ms.fit(prices)

        # Get smoothed regime probabilities
        smoothed = ms.model_.smoothed_marginal_probabilities
        if isinstance(smoothed, pd.DataFrame):
            regime_probs = smoothed
        else:
            regime_probs = pd.DataFrame(
                smoothed,
                columns=[f"regime_{i}" for i in range(k_regimes)],
            )

    # Assign most likely regime
    if len(regime_probs.columns) >= k_regimes:
        # Align index
        regime_probs_aligned = regime_probs.iloc[-len(prices):]
        regime_probs_aligned.index = prices.index[-len(regime_probs_aligned):]

        result = result.loc[regime_probs_aligned.index]
        result["regime"] = regime_probs_aligned.values.argmax(axis=1)
        result["regime_prob"] = regime_probs_aligned.max(axis=1).values

        # Detect transitions
        result["is_transition"] = result["regime"].diff().abs() > 0
        result["is_transition"] = result["is_transition"].fillna(False)

        # Transition surprise (low prob of assigned regime = surprising)
        result["transition_surprise"] = 1.0 - result["regime_prob"]

    n_transitions = result["is_transition"].sum() if "is_transition" in result.columns else 0
    logger.info("regime_anomalies: %d regime transitions detected", n_transitions)

    return result
