"""
ML training pipeline for electricity price forecasting.

Provides a unified interface for XGBoost, LightGBM, and CatBoost models
with walk-forward validation, ensemble weighting, and Yr weather forecast
integration for forward-looking predictions.

Usage:
    from src.models.train import MLPriceForecaster, prepare_ml_features

    X, y = prepare_ml_features(df)
    model = MLPriceForecaster("xgboost")
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
"""

import logging
import time
from typing import Any, Self

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Feature Preparation
# ---------------------------------------------------------------------------

# NOK columns are ~r>0.99 correlated with EUR — redundant for modeling
_NOK_PATTERN_FRAGMENTS = ["_nok_mwh", "_nok_kwh"]

# Price lag/rolling/diff columns — autoregressive, not real price drivers
_PRICE_LAG_PATTERNS = ["price_lag_", "price_rolling_", "price_diff_"]


def prepare_ml_features(
    df: pd.DataFrame,
    target_col: str = "price_eur_mwh",
    drop_nok: bool = True,
    drop_price_lags: bool = True,
    drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target for ML training.

    Separates target from features, drops NOK columns (redundant with EUR,
    r > 0.99 correlation), optionally drops price lag features (autoregressive
    shortcuts that prevent the model from learning fundamental drivers), and
    removes rows with NaN targets.

    Args:
        df: Full feature matrix from build_features.py.
        target_col: Name of the target column.
        drop_nok: If True, drop all NOK price columns (redundant).
        drop_price_lags: If True, drop all price lag/rolling/diff columns
            so the model learns from fundamental drivers (weather, supply,
            demand, commodities) instead of autoregressive shortcuts.
        drop_cols: Additional columns to drop.

    Returns:
        Tuple of (X features DataFrame, y target Series).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # Drop NOK columns (highly correlated with EUR equivalents)
    if drop_nok:
        nok_cols = [
            c for c in X.columns
            if any(frag in c for frag in _NOK_PATTERN_FRAGMENTS)
        ]
        if nok_cols:
            X = X.drop(columns=nok_cols)
            logger.info("Dropped %d NOK columns (redundant with EUR)", len(nok_cols))

    # Drop price lag/rolling/diff features (autoregressive shortcuts)
    if drop_price_lags:
        lag_cols = [
            c for c in X.columns
            if any(c.startswith(pat) for pat in _PRICE_LAG_PATTERNS)
        ]
        if lag_cols:
            X = X.drop(columns=lag_cols)
            logger.info(
                "Dropped %d price lag columns (learning fundamentals, not shortcuts): %s",
                len(lag_cols), lag_cols,
            )

    # Drop additional specified columns
    if drop_cols:
        existing = [c for c in drop_cols if c in X.columns]
        if existing:
            X = X.drop(columns=existing)

    # Drop rows where target is NaN
    valid_mask = y.notna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        logger.info("Dropped %d rows with NaN target", n_dropped)
        X = X[valid_mask]
        y = y[valid_mask]

    logger.info(
        "Prepared %d samples with %d features (target: %s)",
        len(X), X.shape[1], target_col,
    )
    return X, y


# ---------------------------------------------------------------------------
# 2. ML Price Forecaster
# ---------------------------------------------------------------------------

_DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "xgboost": {
        "n_estimators": 1000,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 1000,
        "num_leaves": 63,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": -1,
    },
    "catboost": {
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 42,
        "verbose": 0,
    },
}


class MLPriceForecaster:
    """ML-based price forecaster using gradient boosted trees.

    Wraps XGBoost, LightGBM, and CatBoost behind a consistent interface.
    Supports early stopping with a validation set to prevent overfitting.

    Args:
        model_type: One of "xgboost", "lightgbm", "catboost".
        **params: Override default hyperparameters for the chosen model.

    Example:
        >>> model = MLPriceForecaster("xgboost", max_depth=8)
        >>> model.fit(X_train, y_train, X_val, y_val)
        >>> preds = model.predict(X_test)
        >>> importance = model.feature_importance()
    """

    def __init__(self, model_type: str = "xgboost", **params: Any) -> None:
        model_type = model_type.lower()
        if model_type not in _DEFAULT_PARAMS:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Choose from: {list(_DEFAULT_PARAMS.keys())}"
            )
        self.model_type = model_type
        merged = {**_DEFAULT_PARAMS[model_type], **params}

        # CatBoost treats iterations/n_estimators as synonyms — keep only one
        if model_type == "catboost":
            if "n_estimators" in params:
                merged.pop("iterations", None)
            elif "iterations" in params:
                merged.pop("n_estimators", None)

        self.params = merged
        self.model_: Any = None
        self.fit_time_seconds: float = 0.0
        self.feature_names_: list[str] = []

    def _create_model(self) -> Any:
        """Instantiate the underlying model object."""
        if self.model_type == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(**self.params)
        elif self.model_type == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(**self.params)
        elif self.model_type == "catboost":
            from catboost import CatBoostRegressor
            return CatBoostRegressor(**self.params)
        raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        early_stopping_rounds: int = 50,
    ) -> Self:
        """Fit the model on training data with optional early stopping.

        Args:
            X_train: Training features.
            y_train: Training target.
            X_val: Validation features (for early stopping).
            y_val: Validation target (for early stopping).
            early_stopping_rounds: Stop if no improvement for this many rounds.

        Returns:
            self (for chaining).
        """
        self.feature_names_ = list(X_train.columns)
        self.model_ = self._create_model()

        # Fill remaining NaN in features (tree models handle it, but some
        # versions of LightGBM/CatBoost complain)
        X_train = X_train.ffill().bfill().fillna(0)
        if X_val is not None:
            X_val = X_val.ffill().bfill().fillna(0)

        t0 = time.time()

        if self.model_type == "xgboost":
            fit_kwargs: dict[str, Any] = {}
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["verbose"] = False
                self.model_.set_params(early_stopping_rounds=early_stopping_rounds)
            self.model_.fit(X_train, y_train, **fit_kwargs)

        elif self.model_type == "lightgbm":
            fit_kwargs = {}
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val, y_val)]
                fit_kwargs["callbacks"] = [
                    _lgbm_early_stopping(early_stopping_rounds),
                    _lgbm_log_evaluation(-1),
                ]
            self.model_.fit(X_train, y_train, **fit_kwargs)

        elif self.model_type == "catboost":
            fit_kwargs = {}
            if X_val is not None and y_val is not None:
                fit_kwargs["eval_set"] = (X_val, y_val)
                fit_kwargs["early_stopping_rounds"] = early_stopping_rounds
            self.model_.fit(X_train, y_train, **fit_kwargs)

        self.fit_time_seconds = round(time.time() - t0, 2)

        # Log best iteration if early stopping was used
        best_iter = self._get_best_iteration()
        iter_msg = f", best_iteration={best_iter}" if best_iter else ""
        logger.info(
            "%s fit: %d samples, %d features in %.1f s%s",
            self.model_type, len(X_train), X_train.shape[1],
            self.fit_time_seconds, iter_msg,
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate predictions for the given features.

        Args:
            X: Feature DataFrame (same columns as training).

        Returns:
            Series of predictions with same index as X.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_clean = X.ffill().bfill().fillna(0)
        # Ensure all columns are numeric (forecast_with_yr can produce object cols)
        for col in X_clean.columns:
            if X_clean[col].dtype == object:
                X_clean[col] = pd.to_numeric(X_clean[col], errors="coerce").fillna(0)
        preds = self.model_.predict(X_clean)
        return pd.Series(preds, index=X.index, name=f"pred_{self.model_type}")

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Get feature importance scores.

        Args:
            importance_type: Type of importance. "gain" for split gain
                (default), "weight" for split count.

        Returns:
            Series of importance scores sorted descending, indexed by
            feature name.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if self.model_type == "xgboost":
            imp = self.model_.feature_importances_
        elif self.model_type == "lightgbm":
            imp = self.model_.feature_importances_
        elif self.model_type == "catboost":
            imp = self.model_.get_feature_importance()
        else:
            imp = np.zeros(len(self.feature_names_))

        return (
            pd.Series(imp, index=self.feature_names_, name="importance")
            .sort_values(ascending=False)
        )

    def _get_best_iteration(self) -> int | None:
        """Return the best iteration from early stopping, if available."""
        if self.model_type == "xgboost":
            return getattr(self.model_, "best_iteration", None)
        elif self.model_type == "lightgbm":
            bi = getattr(self.model_, "best_iteration_", None)
            return bi if bi and bi > 0 else None
        elif self.model_type == "catboost":
            return getattr(self.model_, "best_iteration_", None)
        return None

    def __repr__(self) -> str:
        fitted = "fitted" if self.model_ is not None else "not fitted"
        return f"MLPriceForecaster(model_type='{self.model_type}', {fitted})"


# ---------------------------------------------------------------------------
# LightGBM callback helpers (avoid deprecation warnings)
# ---------------------------------------------------------------------------

def _lgbm_early_stopping(stopping_rounds: int):
    """Create LightGBM early stopping callback."""
    from lightgbm import early_stopping
    return early_stopping(stopping_rounds=stopping_rounds, verbose=False)


def _lgbm_log_evaluation(period: int):
    """Create LightGBM log evaluation callback."""
    from lightgbm import log_evaluation
    return log_evaluation(period=period)


# ---------------------------------------------------------------------------
# 3. Walk-Forward Validation
# ---------------------------------------------------------------------------

def walk_forward_validate(
    df: pd.DataFrame,
    model_type: str = "xgboost",
    n_splits: int = 6,
    val_size_hours: int = 720,
    target_col: str = "price_eur_mwh",
    min_train_hours: int = 4000,
    **model_params: Any,
) -> list[dict[str, Any]]:
    """Walk-forward expanding window validation for time series.

    Splits data into expanding training windows with fixed-size validation
    folds. Each fold trains on all data up to the fold boundary, then
    evaluates on the next ``val_size_hours`` hours.

    Args:
        df: Full feature matrix with target column.
        model_type: Model to use ("xgboost", "lightgbm", "catboost").
        n_splits: Number of validation folds.
        val_size_hours: Size of each validation fold in hours.
        target_col: Target column name.
        min_train_hours: Minimum training set size (first fold).
        **model_params: Additional model hyperparameters.

    Returns:
        List of dicts per fold with keys: fold, train_size, val_size,
        val_start, val_end, metrics, fit_time, predictions, actuals.
    """
    from src.models.evaluate import compute_metrics

    X, y = prepare_ml_features(df, target_col=target_col)

    total_hours = len(X)
    # Reserve space for n_splits validation windows from the end
    total_val_hours = n_splits * val_size_hours
    if total_hours < min_train_hours + total_val_hours:
        raise ValueError(
            f"Not enough data: {total_hours} hours, need at least "
            f"{min_train_hours + total_val_hours} for {n_splits} folds"
        )

    # First validation fold starts after the minimum training period
    first_val_start = total_hours - total_val_hours

    results = []
    for fold in range(n_splits):
        val_start_idx = first_val_start + fold * val_size_hours
        val_end_idx = val_start_idx + val_size_hours

        X_train = X.iloc[:val_start_idx]
        y_train = y.iloc[:val_start_idx]
        X_val = X.iloc[val_start_idx:val_end_idx]
        y_val = y.iloc[val_start_idx:val_end_idx]

        if len(X_val) == 0:
            break

        model = MLPriceForecaster(model_type, **model_params)
        model.fit(X_train, y_train, X_val, y_val)
        preds = model.predict(X_val)

        # Naive baseline for skill score
        naive_pred = y.shift(168).iloc[val_start_idx:val_end_idx]

        metrics = compute_metrics(y_val, preds, naive_pred=naive_pred)

        results.append({
            "fold": fold + 1,
            "train_size": len(X_train),
            "val_size": len(X_val),
            "val_start": X_val.index.min(),
            "val_end": X_val.index.max(),
            "metrics": metrics,
            "fit_time": model.fit_time_seconds,
            "predictions": preds,
            "actuals": y_val,
        })

        logger.info(
            "Fold %d/%d: train=%d, val=%d, MAE=%.2f, fit=%.1fs",
            fold + 1, n_splits, len(X_train), len(X_val),
            metrics.get("mae", float("nan")), model.fit_time_seconds,
        )

    return results


# ---------------------------------------------------------------------------
# 4. Ensemble Training
# ---------------------------------------------------------------------------

def train_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, Any]:
    """Train XGBoost + LightGBM + CatBoost and build a weighted ensemble.

    Weights are inversely proportional to each model's validation MAE
    (better models get higher weight).

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.

    Returns:
        Dict with keys: models (dict of fitted MLPriceForecaster),
        predictions (dict of per-model Series), weights (dict of floats),
        ensemble_pred (weighted average Series), metrics (per-model and
        ensemble metrics).
    """
    from src.models.evaluate import compute_metrics

    model_types = ["xgboost", "lightgbm", "catboost"]
    models: dict[str, MLPriceForecaster] = {}
    predictions: dict[str, pd.Series] = {}
    maes: dict[str, float] = {}

    for mt in model_types:
        logger.info("Training %s for ensemble...", mt)
        m = MLPriceForecaster(mt)
        m.fit(X_train, y_train, X_val, y_val)
        pred = m.predict(X_val)

        models[mt] = m
        predictions[mt] = pred
        metrics = compute_metrics(y_val, pred)
        maes[mt] = metrics.get("mae", float("inf"))
        logger.info("%s: MAE=%.3f", mt, maes[mt])

    # Inverse-MAE weights (better model = higher weight)
    inv_maes = {mt: 1.0 / mae for mt, mae in maes.items() if mae > 0}
    total_inv = sum(inv_maes.values())
    weights = {mt: w / total_inv for mt, w in inv_maes.items()}

    # Weighted ensemble prediction
    ensemble_pred = pd.Series(0.0, index=X_val.index)
    for mt, weight in weights.items():
        ensemble_pred += weight * predictions[mt]
    ensemble_pred.name = "pred_ensemble"

    # Also compute a simple average for comparison
    simple_avg = pd.Series(0.0, index=X_val.index)
    for pred in predictions.values():
        simple_avg += pred
    simple_avg /= len(predictions)
    simple_avg.name = "pred_simple_avg"

    # Naive baseline for skill scores
    naive_pred = y_train.shift(168).reindex(y_val.index)

    # Compute all metrics
    all_metrics = {}
    for mt, pred in predictions.items():
        all_metrics[mt] = compute_metrics(y_val, pred, naive_pred=naive_pred)
    all_metrics["ensemble_weighted"] = compute_metrics(
        y_val, ensemble_pred, naive_pred=naive_pred
    )
    all_metrics["ensemble_simple_avg"] = compute_metrics(
        y_val, simple_avg, naive_pred=naive_pred
    )

    logger.info(
        "Ensemble weights: %s",
        {mt: round(w, 3) for mt, w in weights.items()},
    )
    logger.info(
        "Ensemble MAE: %.3f (weighted), %.3f (simple avg)",
        all_metrics["ensemble_weighted"].get("mae", float("nan")),
        all_metrics["ensemble_simple_avg"].get("mae", float("nan")),
    )

    return {
        "models": models,
        "predictions": predictions,
        "weights": weights,
        "ensemble_pred": ensemble_pred,
        "simple_avg_pred": simple_avg,
        "metrics": all_metrics,
    }


# ---------------------------------------------------------------------------
# 5. Forward Forecasting with Yr Weather
# ---------------------------------------------------------------------------

def forecast_with_yr(
    model: MLPriceForecaster,
    last_features: pd.DataFrame,
    yr_forecast_df: pd.DataFrame,
    eur_nok: float,
    target_col: str = "price_eur_mwh",
) -> pd.DataFrame:
    """Produce forward price forecast using Yr weather forecast data.

    Builds a forward feature matrix by:
    1. Taking the last known feature values as a template
    2. Replacing weather columns with Yr forecast values
    3. Computing calendar features for the forecast horizon
    4. Running the model to get EUR/MWh predictions
    5. Converting to NOK/kWh

    Args:
        model: Fitted MLPriceForecaster.
        last_features: Last rows of historical feature matrix (at least
            168 rows for lag computation).
        yr_forecast_df: Yr forecast DataFrame with yr_* columns.
        eur_nok: EUR/NOK exchange rate for conversion.
        target_col: Target column name (excluded from features).

    Returns:
        DataFrame with columns: price_eur_mwh, price_nok_kwh,
        indexed by forecast hour (Europe/Oslo).
    """
    import holidays as holidays_lib

    if yr_forecast_df.empty:
        logger.warning("Empty Yr forecast — cannot produce forward predictions")
        return pd.DataFrame()

    forecast_index = yr_forecast_df.index
    n_steps = len(forecast_index)
    feature_cols = model.feature_names_

    # Start from last known values — forward-fill as template
    template = last_features[feature_cols].iloc[-1:].copy()
    X_future = pd.DataFrame(
        np.tile(template.values, (n_steps, 1)),
        columns=feature_cols,
        index=forecast_index,
    )

    # --- Override calendar features ---
    no_hol = holidays_lib.Norway()
    cal_map = {
        "hour_of_day": forecast_index.hour,
        "day_of_week": forecast_index.dayofweek,
        "month": forecast_index.month,
        "week_of_year": forecast_index.isocalendar().week.values,
        "is_weekend": (forecast_index.dayofweek >= 5).astype(int),
        "is_holiday": pd.Series(
            [int(d in no_hol) for d in forecast_index.date],
            index=forecast_index,
        ),
        "is_business_hour": (
            (forecast_index.hour >= 8)
            & (forecast_index.hour <= 17)
            & (forecast_index.dayofweek < 5)
        ).astype(int),
    }
    for col, values in cal_map.items():
        if col in X_future.columns:
            X_future[col] = values

    # --- Override weather features with Yr forecast ---
    yr_to_hist = {
        "yr_temperature": "temperature",
        "yr_wind_speed": "wind_speed",
        "yr_precipitation_1h": "precipitation",
        "yr_cloud_cover": "cloud_cover",
        "yr_humidity": "humidity",
        "yr_pressure": "pressure",
    }
    for yr_col, hist_col in yr_to_hist.items():
        if hist_col in X_future.columns and yr_col in yr_forecast_df.columns:
            aligned = yr_forecast_df[yr_col].reindex(forecast_index)
            X_future[hist_col] = aligned

    # --- Predict ---
    preds_eur = model.predict(X_future)

    result = pd.DataFrame({
        "price_eur_mwh": preds_eur.values,
        "price_nok_mwh": preds_eur.values * eur_nok,
        "price_nok_kwh": preds_eur.values * eur_nok / 1000,
    }, index=forecast_index)

    logger.info(
        "Forward forecast: %d hours, mean=%.1f EUR/MWh (%.3f NOK/kWh)",
        n_steps, result["price_eur_mwh"].mean(), result["price_nok_kwh"].mean(),
    )
    return result
