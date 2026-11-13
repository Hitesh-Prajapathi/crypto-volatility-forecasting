"""
FastAPI Backend
================
REST API for triggering model training and serving volatility predictions.

Endpoints:
    POST /train       — Run the full training pipeline
    GET  /predict     — Get volatility forecasts for a given model
    GET  /metrics     — Get model comparison results
    GET  /health      — Health check
"""

import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SYMBOL, START_DATE, MODELS_DIR
from src.train import run_training_pipeline

# ─── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title="Crypto Volatility Forecasting API",
    description="Train models and get BTC volatility predictions",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store for pipeline results
_pipeline_results = {}


def load_saved_results() -> bool:
    """Load pre-computed metrics and predictions from disk into memory."""
    global _pipeline_results
    pred_path = os.path.join(MODELS_DIR, "predictions.joblib")
    metrics_path = os.path.join(MODELS_DIR, "metrics.csv")

    loaded = False
    if os.path.exists(metrics_path) or os.path.exists(pred_path):
        if not isinstance(_pipeline_results, dict):
            _pipeline_results = {}
        if os.path.exists(metrics_path):
            try:
                _pipeline_results["metrics"] = pd.read_csv(metrics_path)
                loaded = True
            except Exception:
                pass
        if os.path.exists(pred_path):
            try:
                import joblib
                _pipeline_results["predictions"] = joblib.load(pred_path)
                loaded = True
            except Exception:
                pass
    return loaded


@app.on_event("startup")
def startup_load_models():
    """Attempt to load saved artifacts on server startup."""
    if load_saved_results():
        models = list(_pipeline_results.get("predictions", {}).keys())
        print(f"📦 Pre-loaded models and metrics into memory on startup: {models}")


# ─── Endpoints ────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """System health check."""
    return {"status": "healthy", "service": "crypto-volatility-api"}


@app.post("/train")
def train_models(
    symbol: str = Query(default=SYMBOL, description="Ticker symbol"),
    start_date: str = Query(default=START_DATE, description="Data start date"),
    skip_lstm: bool = Query(default=False, description="Skip LSTM training"),
):
    """Run the full training pipeline.

    Downloads data, trains GARCH / Random Forest / XGBoost / LSTM,
    evaluates all models, and stores results in memory.

    Returns:
        JSON with training metrics for each model.
    """
    global _pipeline_results

    try:
        results = run_training_pipeline(
            symbol=symbol,
            start_date=start_date,
            skip_lstm=skip_lstm,
        )
        _pipeline_results = results

        metrics = results["metrics"].to_dict(orient="records")
        return {
            "status": "success",
            "message": "Training complete",
            "models_trained": list(results["predictions"].keys()),
            "metrics": metrics,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/predict")
def get_predictions(
    model_type: str = Query(
        default="GARCH",
        description="Model to use: GARCH, Random Forest, XGBoost, LSTM",
    ),
    horizon: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Number of forecast points to return",
    ),
):
    """Get volatility predictions from a trained model.

    Args:
        model_type: Name of the model.
        horizon: Number of forecast data points to return.

    Returns:
        JSON with forecast values and metadata.
    """
    if not _pipeline_results or "predictions" not in _pipeline_results:
        load_saved_results()

    if not _pipeline_results or "predictions" not in _pipeline_results:
        raise HTTPException(
            status_code=400,
            detail="No trained models available. Call POST /train first.",
        )

    predictions = _pipeline_results.get("predictions", {})
    if model_type not in predictions:
        available = list(predictions.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_type}' not found. Available: {available}",
        )

    preds = predictions[model_type]
    forecast = preds[-horizon:].tolist() if len(preds) >= horizon else preds.tolist()

    return {
        "model": model_type,
        "horizon": len(forecast),
        "forecast": forecast,
    }


@app.get("/metrics")
def get_metrics():
    """Get model comparison metrics.

    Returns:
        JSON with MAE, RMSE, MAPE for all trained models.
    """
    if not _pipeline_results or "metrics" not in _pipeline_results:
        load_saved_results()

    metrics_df = _pipeline_results.get("metrics")
    if metrics_df is None:
        csv_path = os.path.join(MODELS_DIR, "metrics.csv")
        if os.path.exists(csv_path):
            metrics_df = pd.read_csv(csv_path)
        else:
            raise HTTPException(
                status_code=400,
                detail="No metrics available. Call POST /train first.",
            )

    # Convert to {model_name: {metric: value}} format
    result = {}
    for _, row in metrics_df.iterrows():
        result[row["model"]] = {
            "mae": round(row["mae"], 6),
            "rmse": round(row["rmse"], 6),
            "mape": round(row["mape"], 2),
        }

    return {"metrics": result}
