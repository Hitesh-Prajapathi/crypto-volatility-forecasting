"""
Preprocessing Module
====================
Cleans raw price data, computes log returns, rolling realized volatility,
and variance features needed by downstream models.
"""

import os
from typing import List

import numpy as np
import pandas as pd

from src.config import (
    ROLLING_WINDOWS,
    ANNUALIZATION_FACTOR,
    PROCESSED_DATA_DIR,
)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by date and handle missing values.

    Args:
        df: Raw price DataFrame with a 'Date' column.

    Returns:
        Cleaned DataFrame sorted chronologically.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Forward-fill small gaps, then drop any remaining NaNs in price columns
    price_cols = ["Open", "High", "Low", "Close", "Volume"]
    existing = [c for c in price_cols if c in df.columns]
    df[existing] = df[existing].ffill()
    df.dropna(subset=existing, inplace=True)

    print(f"🧹 Cleaned data: {len(df)} rows")
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns from closing prices.

    log_return_t = ln(Close_t / Close_{t-1})

    Args:
        df: DataFrame with 'Close' column.

    Returns:
        DataFrame with added 'log_close' and 'log_return' columns.
    """
    df = df.copy()
    df["log_close"] = np.log(df["Close"])
    df["log_return"] = df["log_close"].diff()
    return df


def compute_rolling_volatility(
    df: pd.DataFrame,
    windows: List[int] = ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Compute rolling realized volatility at multiple windows.

    rv_Wd = rolling_std(log_return, W)
    rv_Wd_ann = rv_Wd * sqrt(252)

    Args:
        df: DataFrame with 'log_return' column.
        windows: List of rolling window sizes in days.

    Returns:
        DataFrame with rolling volatility columns added.
    """
    df = df.copy()
    for w in windows:
        df[f"rv_{w}d"] = df["log_return"].rolling(w).std()
        df[f"rv_{w}d_ann"] = df[f"rv_{w}d"] * np.sqrt(ANNUALIZATION_FACTOR)
    return df


def compute_variance_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute variance-based features useful for GARCH models.

    Args:
        df: DataFrame with 'log_return' column.

    Returns:
        DataFrame with var_1d, var_30d, and log_var_30d columns.
    """
    df = df.copy()
    df["var_1d"] = df["log_return"] ** 2
    df["var_30d"] = df["log_return"].rolling(30).var()
    df["log_var_30d"] = np.log(df["var_30d"].replace(0, np.nan)).bfill()
    return df


def preprocess_pipeline(
    df: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline.

    Steps:
        1. Clean data
        2. Compute log returns
        3. Compute rolling volatility
        4. Compute variance features
        5. Drop NaN rows
        6. Optionally save to disk

    Args:
        df: Raw price DataFrame.
        save: Whether to save processed CSV to data/processed/.

    Returns:
        Fully preprocessed DataFrame ready for feature engineering.
    """
    df = clean_data(df)
    df = compute_log_returns(df)
    df = compute_rolling_volatility(df)
    df = compute_variance_features(df)

    # Drop rows with NaN from rolling windows
    df = df.dropna().reset_index(drop=True)
    print(f"✅ Preprocessed: {len(df)} rows × {len(df.columns)} columns")

    if save:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        path = os.path.join(PROCESSED_DATA_DIR, "btc_processed.csv")
        df.to_csv(path, index=False)
        print(f"💾 Saved processed data → {path}")

    return df
