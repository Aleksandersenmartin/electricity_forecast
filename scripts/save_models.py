"""Save trained model artifacts for all zones.

Trains XGBoost/LightGBM/CatBoost per zone and saves to artifacts/.
These are needed by the Streamlit dashboard for forecasting.

Usage:
    python scripts/save_models.py          # All zones
    python scripts/save_models.py NO_5     # Single zone
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.models.train import MLPriceForecaster, prepare_ml_features, train_ensemble

ZONES = ["NO_1", "NO_2", "NO_3", "NO_4", "NO_5"]
TRAIN_END = "2024-12-31"
VAL_END = "2025-06-30"
TARGET = "price_eur_mwh"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def save_zone_models(zone: str) -> None:
    """Train and save models for a single zone."""
    path = DATA_DIR / f"features_{zone}_2022-01-01_2026-01-01.parquet"
    if not path.exists():
        print(f"  {zone}: No feature file found, skipping")
        return

    df = pd.read_parquet(path)
    df = df[df.index <= "2026-02-22"]

    df_train = df.loc[:TRAIN_END]
    df_val = df.loc[TRAIN_END:VAL_END].iloc[1:]

    X_train, y_train = prepare_ml_features(df_train, target_col=TARGET)
    X_val, y_val = prepare_ml_features(df_val, target_col=TARGET)

    print(f"  {zone}: Training ({len(X_train):,} train, {len(X_val):,} val)...")

    result = train_ensemble(X_train, y_train, X_val, y_val)

    # Save individual models
    for mt, model in result["models"].items():
        out_path = ARTIFACTS_DIR / f"model_{zone}_{mt}.joblib"
        joblib.dump(model, out_path)
        print(f"    Saved {out_path.name}")

    # Save ensemble weights
    weights_path = ARTIFACTS_DIR / f"weights_{zone}.joblib"
    joblib.dump(result["weights"], weights_path)
    print(f"    Saved {weights_path.name}")

    # Print metrics
    ens_mae = result["metrics"]["ensemble_weighted"].get("mae", "N/A")
    print(f"    Ensemble MAE: {ens_mae}")


def main():
    zones = sys.argv[1:] if len(sys.argv) > 1 else ZONES

    for z in zones:
        if z not in ZONES:
            print(f"Unknown zone: {z}. Valid: {ZONES}")
            continue

    print(f"Saving models for: {zones}")
    print(f"Artifacts dir: {ARTIFACTS_DIR}")
    print()

    for zone in zones:
        if zone not in ZONES:
            continue
        save_zone_models(zone)
        print()

    print("Done. You can now run: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
