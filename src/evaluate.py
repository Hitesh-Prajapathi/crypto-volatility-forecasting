"""
Evaluation Module
==================
Computes standard regression metrics (MAE, RMSE, MAPE)
and provides model comparison utilities.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAE score.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        RMSE score.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Handles near-zero true values by adding a small epsilon.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        MAPE as a percentage (e.g. 12.5 means 12.5%).
    """
    epsilon = 1e-10
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Compute all metrics for a single model.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        model_name: Name identifier for the model.

    Returns:
        Dictionary with model name and all metric values.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    # Align lengths
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]

    metrics = {
        "model": model_name,
        "mae": compute_mae(y_true, y_pred),
        "rmse": compute_rmse(y_true, y_pred),
        "mape": compute_mape(y_true, y_pred),
        "n_samples": n,
    }

    print(
        f"📊 {model_name:20s}  |  "
        f"MAE={metrics['mae']:.6f}  "
        f"RMSE={metrics['rmse']:.6f}  "
        f"MAPE={metrics['mape']:.2f}%"
    )
    return metrics


def compare_models(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a comparison DataFrame from multiple model evaluations.

    Args:
        results: List of metric dictionaries from evaluate_model().

    Returns:
        Sorted DataFrame with all models and their metrics.
    """
    df = pd.DataFrame(results)
    df = df.sort_values("rmse").reset_index(drop=True)
    print("\n🏆 Model Comparison (sorted by RMSE):")
    print(df.to_string(index=False))
    return df
