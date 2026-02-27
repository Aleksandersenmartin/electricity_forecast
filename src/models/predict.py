"""Inference pipeline for electricity forecasting.

Loads saved model artifacts and produces forward-looking predictions
using Yr weather forecasts. Designed for the Streamlit dashboard.

Usage:
    from src.models.predict import load_zone_model, predict_forward

    models = load_zone_model("NO_5")
    forecast = predict_forward("NO_5")
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

ZONES = ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"]

# Simple in-memory cache for predictions
_prediction_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def load_zone_model(
    zone: str,
    model_type: str = "ensemble",
) -> dict[str, Any]:
    """Load saved model artifacts for a zone.

    Args:
        zone: Norwegian zone (e.g., "NO_5").
        model_type: "ensemble" (loads all 3 + weights), or a specific
            model type ("xgboost", "lightgbm", "catboost").

    Returns:
        Dict with keys: models (dict of model objects), weights (dict),
        model_type (str).
    """
    import joblib

    if model_type == "ensemble":
        models = {}
        for mt in ["xgboost", "lightgbm", "catboost"]:
            path = ARTIFACTS_DIR / f"model_{zone}_{mt}.joblib"
            if path.exists():
                models[mt] = joblib.load(path)
                logger.info("Loaded %s model for %s", mt, zone)
            else:
                logger.warning("Model not found: %s", path)

        weights_path = ARTIFACTS_DIR / f"weights_{zone}.joblib"
        if weights_path.exists():
            weights = joblib.load(weights_path)
        else:
            # Equal weights if no saved weights
            n = len(models)
            weights = {mt: 1.0 / n for mt in models} if n > 0 else {}

        return {"models": models, "weights": weights, "model_type": "ensemble"}
    else:
        path = ARTIFACTS_DIR / f"model_{zone}_{model_type}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        model = joblib.load(path)
        return {"models": {model_type: model}, "weights": {model_type: 1.0}, "model_type": model_type}


def get_latest_features(zone: str) -> pd.DataFrame:
    """Load the latest feature matrix for a zone.

    Searches for parquet files in data/processed/ and returns the most
    recent one.

    Args:
        zone: Norwegian zone.

    Returns:
        Feature DataFrame with DatetimeIndex (Europe/Oslo).
    """
    # Find all feature files for this zone
    pattern = f"features_{zone}_*.parquet"
    files = sorted(DATA_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No feature files found for {zone} in {DATA_DIR}"
        )

    # Use the most recent file
    latest = files[-1]
    df = pd.read_parquet(latest)
    logger.info("Loaded features for %s: %s (%d rows)", zone, latest.name, len(df))

    return df


def predict_forward(
    zone: str,
    model_type: str = "ensemble",
    cache_hours: float = 1.0,
) -> pd.DataFrame:
    """Produce forward price forecast for a zone.

    End-to-end pipeline: load model, fetch Yr weather, build features,
    predict. Results are cached for cache_hours.

    Args:
        zone: Norwegian zone.
        model_type: Model to use ("ensemble" or specific type).
        cache_hours: Cache duration in hours.

    Returns:
        DataFrame with price_eur_mwh, price_nok_mwh, price_nok_kwh
        indexed by forecast hour.
    """
    from src.data.fetch_yr_forecast import fetch_yr_forecast
    from src.models.train import forecast_with_yr

    cache_key = f"{zone}_{model_type}"
    now = time.time()

    # Check cache
    if cache_key in _prediction_cache:
        cached_time, cached_df = _prediction_cache[cache_key]
        if now - cached_time < cache_hours * 3600:
            logger.info("Using cached prediction for %s (%.0f min old)",
                        zone, (now - cached_time) / 60)
            return cached_df

    # Load model
    try:
        model_info = load_zone_model(zone, model_type)
    except FileNotFoundError:
        logger.error("No saved models for %s — run notebooks first", zone)
        return pd.DataFrame()

    models = model_info["models"]
    weights = model_info["weights"]

    if not models:
        logger.error("No models loaded for %s", zone)
        return pd.DataFrame()

    # Load features for context
    try:
        features_df = get_latest_features(zone)
    except FileNotFoundError:
        logger.error("No feature data for %s", zone)
        return pd.DataFrame()

    # Get EUR/NOK rate (latest available)
    eur_nok = features_df["eur_nok"].dropna().iloc[-1] if "eur_nok" in features_df.columns else 11.5

    # Fetch Yr weather forecast
    try:
        yr_df = fetch_yr_forecast(zone, cache=True)
    except Exception as e:
        logger.error("Failed to fetch Yr forecast for %s: %s", zone, e)
        return pd.DataFrame()

    if yr_df.empty:
        logger.warning("Empty Yr forecast for %s", zone)
        return pd.DataFrame()

    # Use last 200 rows of features as context for lag computation
    last_features = features_df.iloc[-200:]

    # Predict with each model and combine
    all_preds = {}
    for mt, model in models.items():
        try:
            pred_df = forecast_with_yr(
                model, last_features, yr_df, eur_nok,
            )
            if not pred_df.empty:
                all_preds[mt] = pred_df["price_eur_mwh"]
        except Exception as e:
            logger.warning("Prediction failed for %s/%s: %s", zone, mt, e)

    if not all_preds:
        logger.error("All predictions failed for %s", zone)
        return pd.DataFrame()

    # Weighted ensemble
    if len(all_preds) > 1:
        combined = pd.DataFrame(all_preds)
        active_weights = {mt: weights.get(mt, 1.0 / len(all_preds))
                          for mt in all_preds}
        total_w = sum(active_weights.values())
        ensemble_eur = sum(
            combined[mt] * (w / total_w)
            for mt, w in active_weights.items()
        )
    else:
        ensemble_eur = next(iter(all_preds.values()))

    result = pd.DataFrame({
        "price_eur_mwh": ensemble_eur.values,
        "price_nok_mwh": ensemble_eur.values * eur_nok,
        "price_nok_kwh": ensemble_eur.values * eur_nok / 1000,
    }, index=ensemble_eur.index)

    # Cache
    _prediction_cache[cache_key] = (now, result)

    logger.info(
        "Forward forecast for %s: %d hours, mean=%.1f EUR/MWh",
        zone, len(result), result["price_eur_mwh"].mean(),
    )

    return result


def get_historical_predictions(
    zone: str,
    start_date: str,
    end_date: str,
    model_type: str = "ensemble",
) -> pd.DataFrame:
    """Run model on historical features for backtesting visualization.

    Args:
        zone: Norwegian zone.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        model_type: Model to use.

    Returns:
        DataFrame with price_eur_mwh predictions, indexed hourly.
    """
    from src.models.train import prepare_ml_features

    model_info = load_zone_model(zone, model_type)
    models = model_info["models"]
    weights = model_info["weights"]

    features_df = get_latest_features(zone)
    features_slice = features_df.loc[start_date:end_date]

    if features_slice.empty:
        return pd.DataFrame()

    X, y = prepare_ml_features(features_slice, target_col="price_eur_mwh")

    all_preds = {}
    for mt, model in models.items():
        try:
            pred = model.predict(X)
            all_preds[mt] = pred
        except Exception as e:
            logger.warning("Historical prediction failed for %s/%s: %s", zone, mt, e)

    if not all_preds:
        return pd.DataFrame()

    # Weighted ensemble
    if len(all_preds) > 1:
        combined = pd.DataFrame(all_preds)
        active_weights = {mt: weights.get(mt, 1.0 / len(all_preds))
                          for mt in all_preds}
        total_w = sum(active_weights.values())
        ensemble_pred = sum(
            combined[mt] * (w / total_w)
            for mt, w in active_weights.items()
        )
    else:
        ensemble_pred = next(iter(all_preds.values()))

    result = pd.DataFrame({
        "price_eur_mwh": ensemble_pred,
        "actual": y,
    })

    return result
