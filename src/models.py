"""
Statistical & ML Models Module
================================
Implements GARCH volatility model and tree-based ML models
(Random Forest, XGBoost) for volatility forecasting.
"""

import os
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.config import (
    GARCH_P, GARCH_Q,
    RF_N_ESTIMATORS, RF_MAX_DEPTH, RF_RANDOM_STATE,
    XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE, XGB_RANDOM_STATE,
    MODELS_DIR,
)


# ────────────────────────────────────────────────────────────
# GARCH Model
# ────────────────────────────────────────────────────────────

def train_garch(
    returns: np.ndarray,
    p: int = GARCH_P,
    q: int = GARCH_Q,
    vol: str = "Garch",
    dist: str = "normal",
) -> Any:
    """Fit a GARCH-family model to log returns.

    Args:
        returns: Array of log returns (raw scale, NOT multiplied by 100).
        p: GARCH lag order for the variance.
        q: ARCH lag order.
        vol: Volatility model type ('Garch', 'EGARCH', 'GARCH').
        dist: Error distribution ('normal', 'studentst', 'skewt').

    Returns:
        Fitted arch model result object.
    """
    # Scale returns to percentage (arch library convention)
    scaled = returns * 100

    model = arch_model(scaled, vol=vol, p=p, q=q, dist=dist, mean="Constant")
    result = model.fit(disp="off")
    print(f"✅ {vol}({p},{q}) fitted  |  AIC={result.aic:.1f}  BIC={result.bic:.1f}")
    return result


def forecast_garch(
    model_result: Any,
    horizon: int = 1,
) -> np.ndarray:
    """Generate GARCH volatility forecasts.

    Args:
        model_result: Fitted GARCH model result.
        horizon: Number of steps ahead to forecast.

    Returns:
        Array of forecasted volatility (daily std, natural scale).
    """
    forecast = model_result.forecast(horizon=horizon)
    # variance forecast → std → back from percentage scale
    variance = forecast.variance.values[-1, :]
    volatility = np.sqrt(variance) / 100.0
    return volatility


def garch_rolling_forecast(
    returns: np.ndarray,
    train_size: int,
    p: int = GARCH_P,
    q: int = GARCH_Q,
    vol: str = "Garch",
) -> np.ndarray:
    """Rolling one-step-ahead GARCH forecast over the test period.

    Re-fits the model at each step using an expanding window.

    Args:
        returns: Full array of log returns.
        train_size: Number of initial training observations.
        p: GARCH p parameter.
        q: GARCH q parameter.
        vol: Volatility specification.

    Returns:
        Array of one-step-ahead volatility forecasts for the test period.
    """
    forecasts = []
    total = len(returns)

    for t in range(train_size, total):
        train_window = returns[:t] * 100
        model = arch_model(train_window, vol=vol, p=p, q=q, mean="Constant")
        try:
            res = model.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=1)
            sigma = np.sqrt(fc.variance.values[-1, 0]) / 100.0
        except Exception:
            sigma = np.std(returns[:t])
        forecasts.append(sigma)

    print(f"📈 GARCH rolling forecast: {len(forecasts)} steps")
    return np.array(forecasts)


# ────────────────────────────────────────────────────────────
# Random Forest
# ────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save: bool = True,
) -> RandomForestRegressor:
    """Train a Random Forest regressor for volatility prediction.

    Args:
        X_train: Training feature matrix.
        y_train: Training target values.
        save: Whether to save the model to disk.

    Returns:
        Fitted RandomForestRegressor.
    """
    model = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"✅ Random Forest trained  |  {RF_N_ESTIMATORS} trees, depth={RF_MAX_DEPTH}")

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, "rf.pkl")
        joblib.dump(model, path)
        print(f"💾 Saved → {path}")

    return model


# ────────────────────────────────────────────────────────────
# XGBoost
# ────────────────────────────────────────────────────────────

def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    save: bool = True,
) -> XGBRegressor:
    """Train an XGBoost regressor for volatility prediction.

    Args:
        X_train: Training feature matrix.
        y_train: Training target values.
        save: Whether to save the model to disk.

    Returns:
        Fitted XGBRegressor.
    """
    model = XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LEARNING_RATE,
        random_state=XGB_RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    print(f"✅ XGBoost trained  |  {XGB_N_ESTIMATORS} rounds, lr={XGB_LEARNING_RATE}")

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, "xgb.pkl")
        joblib.dump(model, path)
        print(f"💾 Saved → {path}")

    return model


# ────────────────────────────────────────────────────────────
# Model Loading
# ────────────────────────────────────────────────────────────

def load_model(filename: str) -> Any:
    """Load a saved scikit-learn / XGBoost model from disk.

    Args:
        filename: Name of the model file (e.g. 'rf.pkl').

    Returns:
        Loaded model object.
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)
