"""
Training Pipeline
==================
End-to-end orchestration: data download → preprocessing → feature engineering
→ train/test split → train all models → evaluate → save results.
"""

import os
import sys
import warnings
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SYMBOL, START_DATE, TARGET_COLUMN, TEST_RATIO,
    LSTM_SEQUENCE_LENGTH, MODELS_DIR,
)
from src.data_loader import download_btc_data, load_raw_data
from src.preprocessing import preprocess_pipeline
from src.features import prepare_ml_features, get_feature_columns
from src.models import (
    train_garch, garch_rolling_forecast,
    train_random_forest, train_xgboost,
)
from src.lstm_model import create_sequences, train_lstm, predict_lstm
from src.evaluate import evaluate_model, compare_models
from src.visualize import (
    plot_price_trend, plot_rolling_volatility, plot_acf_pacf,
    plot_predictions, plot_model_comparison, plot_loss_curve,
)

warnings.filterwarnings("ignore")


def run_training_pipeline(
    symbol: str = SYMBOL,
    start_date: str = START_DATE,
    skip_lstm: bool = False,
) -> Dict[str, Any]:
    """Execute the full training pipeline.

    Steps:
        1. Download / load BTC data
        2. Preprocess (log returns, rolling volatility)
        3. Engineer features
        4. Train/test split
        5. Train GARCH, Random Forest, XGBoost, LSTM
        6. Evaluate all models
        7. Generate comparison plots
        8. Save results

    Args:
        symbol: Ticker symbol to download.
        start_date: Data start date.
        skip_lstm: Skip LSTM training (faster for testing).

    Returns:
        Dictionary with keys:
            'metrics' — DataFrame of model comparison
            'predictions' — dict of model_name → prediction arrays
            'dates_test' — test period dates
            'y_test' — actual test target values
    """
    print("=" * 70)
    print("🚀 CRYPTO VOLATILITY FORECASTING PIPELINE")
    print("=" * 70)

    # ── Step 1: Data ──────────────────────────────────────────
    print("\n📦 Step 1: Data Collection")
    try:
        # Always try to fetch the latest data first
        df_raw = download_btc_data(symbol, start_date)
    except Exception as e:
        print(f"  ⚠️ Failed to download new data ({e}). Falling back to cached data.")
        df_raw = load_raw_data(symbol)

    # ── Step 2: Preprocess ────────────────────────────────────
    print("\n🔬 Step 2: Preprocessing")
    df_proc = preprocess_pipeline(df_raw)

    # ── Step 3: Feature Engineering ───────────────────────────
    print("\n🔧 Step 3: Feature Engineering")
    df_feat = prepare_ml_features(df_proc)

    # ── Step 4: Train/Test Split ──────────────────────────────
    print("\n✂️  Step 4: Train/Test Split")
    train_size = int(len(df_feat) * (1 - TEST_RATIO))
    df_train = df_feat.iloc[:train_size]
    df_test = df_feat.iloc[train_size:]

    feature_cols = get_feature_columns(df_feat)
    X_train = df_train[feature_cols].values
    X_test = df_test[feature_cols].values
    y_train = df_train[TARGET_COLUMN].values
    y_test = df_test[TARGET_COLUMN].values
    dates_test = pd.to_datetime(df_test["Date"]).values

    print(f"   Train: {len(df_train)} samples  |  Test: {len(df_test)} samples")
    print(f"   Features: {len(feature_cols)}")

    # ── Step 5: Train Models ──────────────────────────────────
    print("\n🏋️ Step 5: Training Models")
    predictions = {}
    all_metrics = []

    # --- 5a: GARCH ---
    print("\n  [1/4] GARCH(1,1)")
    try:
        returns = df_proc["log_return"].values
        garch_preds = garch_rolling_forecast(returns, train_size)
        # Align with test target
        n = min(len(garch_preds), len(y_test))
        predictions["GARCH"] = garch_preds[:n]
        all_metrics.append(evaluate_model(y_test[:n], garch_preds[:n], "GARCH"))
    except Exception as e:
        print(f"  ⚠️ GARCH failed: {e}")

    # --- 5b: Random Forest ---
    print("\n  [2/4] Random Forest")
    try:
        rf_model = train_random_forest(X_train, y_train)
        rf_preds = rf_model.predict(X_test)
        predictions["Random Forest"] = rf_preds
        all_metrics.append(evaluate_model(y_test, rf_preds, "Random Forest"))
    except Exception as e:
        print(f"  ⚠️ Random Forest failed: {e}")

    # --- 5c: XGBoost ---
    print("\n  [3/4] XGBoost")
    try:
        xgb_model = train_xgboost(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)
        predictions["XGBoost"] = xgb_preds
        all_metrics.append(evaluate_model(y_test, xgb_preds, "XGBoost"))
    except Exception as e:
        print(f"  ⚠️ XGBoost failed: {e}")

    # --- 5d: LSTM ---
    if not skip_lstm:
        print("\n  [4/4] LSTM")
        try:
            # Scale features
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            X_all = df_feat[feature_cols].values
            y_all = df_feat[TARGET_COLUMN].values.reshape(-1, 1)

            X_scaled = scaler_X.fit_transform(X_all)
            y_scaled = scaler_y.fit_transform(y_all).flatten()

            # Create sequences
            X_seq, y_seq = create_sequences(X_scaled, y_scaled)

            # Split sequences (accounting for seq_len offset)
            seq_train_size = train_size - LSTM_SEQUENCE_LENGTH
            X_train_seq = X_seq[:seq_train_size]
            y_train_seq = y_seq[:seq_train_size]
            X_test_seq = X_seq[seq_train_size:]
            y_test_seq = y_seq[seq_train_size:]

            # Further split training into train/val
            val_size = int(len(X_train_seq) * 0.15)
            X_val_seq = X_train_seq[-val_size:]
            y_val_seq = y_train_seq[-val_size:]
            X_train_seq = X_train_seq[:-val_size]
            y_train_seq = y_train_seq[:-val_size]

            # Train
            lstm_model, history = train_lstm(
                X_train_seq, y_train_seq, X_val_seq, y_val_seq
            )

            # Predict
            lstm_preds = predict_lstm(lstm_model, X_test_seq, scaler_y)
            y_test_lstm = scaler_y.inverse_transform(
                y_test_seq.reshape(-1, 1)
            ).flatten()

            predictions["LSTM"] = lstm_preds
            all_metrics.append(
                evaluate_model(y_test_lstm, lstm_preds, "LSTM")
            )

            # Plot loss curve
            plot_loss_curve(history)
        except Exception as e:
            print(f"  ⚠️ LSTM failed: {e}")
    else:
        print("\n  [4/4] LSTM — skipped")

    # ── Step 6: Evaluate & Compare ────────────────────────────
    print("\n📊 Step 6: Model Comparison")
    metrics_df = compare_models(all_metrics)

    # ── Step 7: Visualizations ────────────────────────────────
    print("\n🎨 Step 7: Generating Plots")
    plot_price_trend(df_proc)
    plot_rolling_volatility(df_proc)
    plot_acf_pacf(df_proc["log_return"])

    for name, preds in predictions.items():
        n = min(len(dates_test), len(preds))
        plot_predictions(dates_test[:n], y_test[:n], preds[:n], name)

    if len(predictions) > 1:
        plot_model_comparison(dates_test, y_test, predictions)

    # ── Save metrics & predictions ────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    metrics_df.to_csv(os.path.join(MODELS_DIR, "metrics.csv"), index=False)
    joblib.dump(predictions, os.path.join(MODELS_DIR, "predictions.joblib"))

    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)

    return {
        "metrics": metrics_df,
        "predictions": predictions,
        "dates_test": dates_test,
        "y_test": y_test,
    }


if __name__ == "__main__":
    run_training_pipeline()
