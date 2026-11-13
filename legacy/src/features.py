"""
Feature Engineering Module
===========================
Builds ML-ready features from preprocessed BTC data:
lag features, rolling statistics, calendar features,
and HAR-RV (Heterogeneous Autoregressive) features.
"""

from typing import List

import numpy as np
import pandas as pd

from src.config import LAG_PERIODS, TARGET_COLUMN


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    lags: List[int] = LAG_PERIODS,
) -> pd.DataFrame:
    """Create lagged values of the target and squared returns.

    Args:
        df: DataFrame with target column and 'log_return'.
        target_col: Column to create lags for.
        lags: List of lag periods.

    Returns:
        DataFrame with lag_<target>_<n> and lag_ret2_<n> columns.
    """
    df = df.copy()
    for lag in lags:
        df[f"lag_{target_col}_{lag}"] = df[target_col].shift(lag)
        df[f"lag_ret2_{lag}"] = (df["log_return"] ** 2).shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and standard deviation of log returns.

    Windows: 5, 10, 20 days (complementary to the 7/30 from preprocessing).

    Args:
        df: DataFrame with 'log_return' column.

    Returns:
        DataFrame with rolling mean/std features added.
    """
    df = df.copy()
    for w in [5, 10, 20]:
        df[f"roll_mean_{w}"] = df["log_return"].rolling(w).mean()
        df[f"roll_std_{w}"] = df["log_return"].rolling(w).std()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week as an integer feature (Mon=0 … Sun=6).

    Args:
        df: DataFrame with 'Date' column.

    Returns:
        DataFrame with 'day_of_week' column.
    """
    df = df.copy()
    df["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek
    return df


def prepare_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline for ML models.

    Combines lag features, rolling statistics, and calendar features,
    then drops rows with NaN introduced by the feature engineering.

    Args:
        df: Preprocessed DataFrame.

    Returns:
        Feature-rich DataFrame ready for train/test split.
    """
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = df.dropna().reset_index(drop=True)
    print(f"🔧 Feature engineering complete: {len(df)} rows × {len(df.columns)} cols")
    return df


def prepare_har_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build HAR-RV (Heterogeneous Autoregressive Realized Volatility) features.

    HAR uses three horizons:
        - Daily RV   (rv_7d, last 1 day proxy)
        - Weekly RV  (5-day rolling mean of rv_7d)
        - Monthly RV (22-day rolling mean of rv_7d)

    Args:
        df: Preprocessed DataFrame with 'rv_7d' and TARGET_COLUMN.

    Returns:
        DataFrame with HAR features: rv_daily, rv_weekly, rv_monthly.
    """
    df = df.copy()
    df["rv_daily"] = df["rv_7d"]
    df["rv_weekly"] = df["rv_7d"].rolling(5).mean()
    df["rv_monthly"] = df["rv_7d"].rolling(22).mean()
    df = df.dropna().reset_index(drop=True)
    print(f"📊 HAR features ready: {len(df)} rows")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of feature columns used by ML models.

    Excludes date, raw price, and target columns.

    Args:
        df: Feature-engineered DataFrame.

    Returns:
        List of feature column names.
    """
    exclude = [
        "Date", "Open", "High", "Low", "Close", "Volume",
        "log_close", TARGET_COLUMN,
    ]
    return [c for c in df.columns if c not in exclude]
