"""
Visualization Module
=====================
Publication-quality plots for BTC price data, volatility analysis,
ACF/PACF, and model prediction comparisons.
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend for macOS/server compatibility
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from src.config import FIGURES_DIR, FIGURE_DPI, FIGURE_STYLE

# Apply global style
try:
    plt.style.use(FIGURE_STYLE)
except OSError:
    plt.style.use("seaborn-v0_8")

sns.set_palette("husl")


def _save_figure(fig: plt.Figure, filename: str) -> str:
    """Save figure to the figures directory.

    Args:
        fig: Matplotlib figure object.
        filename: Output filename (e.g. 'price_trend.png').

    Returns:
        Full path to the saved file.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", facecolor="white")
    print(f"📸 Saved → {path}")
    return path


def plot_price_trend(
    df: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """Plot BTC closing price over time.

    Args:
        df: DataFrame with 'Date' and 'Close' columns.
        save: Whether to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df["Date"], df["Close"], linewidth=1.2, color="#F7931A")
    ax.set_title("Bitcoin (BTC-USD) Closing Price", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()

    if save:
        _save_figure(fig, "btc_price_trend.png")
    return fig


def plot_rolling_volatility(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    save: bool = True,
) -> plt.Figure:
    """Plot rolling realized volatility at multiple windows.

    Args:
        df: DataFrame with rv_Xd columns.
        windows: Which windows to plot. Auto-detected if None.
        save: Whether to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    if windows is None:
        windows = [int(c.split("_")[1].replace("d", ""))
                    for c in df.columns if c.startswith("rv_") and c.endswith("d")
                    and "ann" not in c]

    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for w, color in zip(sorted(windows), colors):
        col = f"rv_{w}d"
        if col in df.columns:
            ax.plot(df["Date"], df[col], label=f"{w}-day RV", linewidth=1, color=color)

    ax.set_title("Rolling Realized Volatility", fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (std of log returns)")
    ax.legend()
    fig.autofmt_xdate()

    if save:
        _save_figure(fig, "rolling_volatility.png")
    return fig


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 40,
    save: bool = True,
) -> plt.Figure:
    """Plot ACF and PACF of a time series.

    Args:
        series: Stationary time series (e.g. log returns).
        lags: Number of lags to display.
        save: Whether to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title("Autocorrelation (ACF)", fontsize=14)

    plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    axes[1].set_title("Partial Autocorrelation (PACF)", fontsize=14)

    fig.suptitle("Serial Correlation Analysis", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save:
        _save_figure(fig, "acf_pacf.png")
    return fig


def plot_predictions(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    save: bool = True,
) -> plt.Figure:
    """Plot predicted vs actual volatility.

    Args:
        dates: Date array for the x-axis.
        y_true: Actual volatility values.
        y_pred: Predicted volatility values.
        model_name: Model name for the title.
        save: Whether to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(dates, y_true, label="Actual", linewidth=1.5, color="#2ca02c")
    ax.plot(dates, y_pred, label=f"{model_name} Forecast", linewidth=1.5,
            color="#d62728", linestyle="--")
    ax.set_title(f"{model_name}: Predicted vs Actual Volatility",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.legend(fontsize=12)
    fig.autofmt_xdate()

    if save:
        safe_name = model_name.lower().replace(" ", "_").replace("/", "_")
        _save_figure(fig, f"predictions_{safe_name}.png")
    return fig


def plot_model_comparison(
    dates: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    save: bool = True,
) -> plt.Figure:
    """Overlay all model predictions on a single chart.

    Args:
        dates: Date array for x-axis.
        y_true: Actual volatility.
        predictions: Dict mapping model names to prediction arrays.
        save: Whether to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(dates, y_true, label="Actual", linewidth=2, color="black", alpha=0.7)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]
    for (name, preds), color in zip(predictions.items(), colors):
        n = min(len(dates), len(preds))
        ax.plot(dates[:n], preds[:n], label=name, linewidth=1.3, color=color)

    ax.set_title("Model Comparison: Volatility Forecasts",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    ax.legend(fontsize=11)
    fig.autofmt_xdate()

    if save:
        _save_figure(fig, "model_comparison.png")
    return fig


def plot_loss_curve(
    history: dict,
    save: bool = True,
) -> plt.Figure:
    """Plot LSTM training & validation loss curves.

    Args:
        history: Keras history.history dictionary.
        save: Whether to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history["loss"], label="Training Loss", linewidth=1.5)
    ax.plot(history["val_loss"], label="Validation Loss", linewidth=1.5)
    ax.set_title("LSTM Training Progress", fontsize=16, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE)")
    ax.legend(fontsize=12)

    if save:
        _save_figure(fig, "lstm_loss_curve.png")
    return fig
