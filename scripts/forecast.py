"""
Standalone Forecast Script
============================
Loads saved models and produces volatility forecasts
without retraining.

Usage:
    python scripts/forecast.py --model rf --horizon 30
    python scripts/forecast.py --model xgb --horizon 60
    python scripts/forecast.py --model lstm --horizon 30
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_COLUMN, LSTM_SEQUENCE_LENGTH
from src.features import prepare_ml_features, get_feature_columns
from src.models import load_model
from src.lstm_model import load_lstm_model, create_sequences, predict_lstm


def main():
    """Load saved model and generate forecasts."""
    parser = argparse.ArgumentParser(
        description="Generate volatility forecast from saved model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["rf", "xgb", "lstm"],
        help="Model to use: rf (Random Forest), xgb (XGBoost), lstm (PyTorch LSTM)",
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
    feature_cols = get_feature_columns(df_feat)

    if args.model == "lstm":
        # Load and run PyTorch LSTM
        X_all = df_feat[feature_cols].values
        y_all = df_feat[TARGET_COLUMN].values.reshape(-1, 1)

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X_all)
        scaler_y.fit(y_all)

        X_seq, _ = create_sequences(X_scaled, y_all.flatten())
        try:
            model = load_lstm_model(input_size=len(feature_cols))
        except FileNotFoundError:
            print("❌ LSTM model 'lstm.pt' not found. Train it first:")
            print("   python scripts/run_pipeline.py")
            sys.exit(1)

        seq_predictions = predict_lstm(model, X_seq, scaler_y)
        pad = np.full(LSTM_SEQUENCE_LENGTH, seq_predictions[0])
        predictions = np.concatenate([pad, seq_predictions])
    else:
        # Load tree-based model (RF / XGB)
        model_file = f"{args.model}.pkl"
        try:
            model = load_model(model_file)
        except FileNotFoundError:
            print(f"❌ Model '{model_file}' not found. Train it first.")
            sys.exit(1)

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
