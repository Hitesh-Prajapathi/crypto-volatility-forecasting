"""
Standalone Forecast Script
============================
Loads saved models and produces volatility forecasts
without retraining.

Usage:
    python scripts/forecast.py --model rf --horizon 30
    python scripts/forecast.py --model xgb --horizon 60
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_COLUMN
from src.features import prepare_ml_features, get_feature_columns
from src.models import load_model


def main():
    """Load saved model and generate forecasts."""
    parser = argparse.ArgumentParser(
        description="Generate volatility forecast from saved model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "xgb"],
        help="Model to use: rf (Random Forest), xgb (XGBoost)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="Number of recent forecast points to display",
    )

    args = parser.parse_args()

    # Load processed data
    data_path = os.path.join(PROCESSED_DATA_DIR, "btc_processed.csv")
    if not os.path.exists(data_path):
        print("❌ No processed data found. Run the pipeline first:")
        print("   python scripts/run_pipeline.py")
        sys.exit(1)

    df = pd.read_csv(data_path, parse_dates=["Date"])
    df_feat = prepare_ml_features(df)

    # Load model
    model_file = f"{args.model}.pkl"
    try:
        model = load_model(model_file)
    except FileNotFoundError:
        print(f"❌ Model '{model_file}' not found. Train it first.")
        sys.exit(1)

    # Generate predictions on recent data
    feature_cols = get_feature_columns(df_feat)
    X = df_feat[feature_cols].values
    predictions = model.predict(X)

    # Show last N forecasts
    n = min(args.horizon, len(predictions))
    recent = predictions[-n:]
    dates = df_feat["Date"].iloc[-n:].values

    print(f"\n🔮 {args.model.upper()} Volatility Forecast (last {n} days):")
    print("-" * 45)
    for date, pred in zip(dates, recent):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        print(f"  {date_str}  |  {pred:.6f}")

    print(f"\n  Mean: {np.mean(recent):.6f}")
    print(f"  Std:  {np.std(recent):.6f}")


if __name__ == "__main__":
    main()
