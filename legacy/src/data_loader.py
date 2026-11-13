"""
Data Collection Module
======================
Downloads cryptocurrency price data from Yahoo Finance (yfinance)
and saves raw data to disk.
"""

import os
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config import SYMBOL, START_DATE, RAW_DATA_DIR


def download_btc_data(
    symbol: str = SYMBOL,
    start_date: str = START_DATE,
    end_date: Optional[str] = None,
    save: bool = True,
) -> pd.DataFrame:
    """Download BTC price data from Yahoo Finance.

    Args:
        symbol: Ticker symbol (default: BTC-USD).
        start_date: Start date in YYYY-MM-DD format.
        end_date: Optional end date. Defaults to today.
        save: Whether to save raw CSV to data/raw/.

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume.
    """
    print(f"📥 Downloading {symbol} data from {start_date}...")
    df = yf.download(symbol, start=start_date, end=end_date)

    # Flatten multi-level columns (yfinance >=0.2.30 returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if col[1] == "" or col[1] == symbol else col[0]
            for col in df.columns
        ]

    df.reset_index(inplace=True)

    # Ensure standard column names
    expected = ["Date", "Open", "High", "Low", "Close", "Volume"]
    for col in expected:
        if col not in df.columns:
            # Try case-insensitive match
            for c in df.columns:
                if c.lower() == col.lower():
                    df.rename(columns={c: col}, inplace=True)
                    break

    print(f"✅ Downloaded {len(df)} rows  |  Range: {df['Date'].min()} → {df['Date'].max()}")

    if save:
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        path = os.path.join(RAW_DATA_DIR, f"{symbol.replace('-', '_')}_raw.csv")
        df.to_csv(path, index=False)
        print(f"💾 Saved raw data → {path}")

    return df


def load_raw_data(symbol: str = SYMBOL) -> pd.DataFrame:
    """Load previously saved raw CSV.

    Args:
        symbol: Ticker symbol used during download.

    Returns:
        DataFrame from the saved CSV file.

    Raises:
        FileNotFoundError: If the raw CSV doesn't exist yet.
    """
    path = os.path.join(RAW_DATA_DIR, f"{symbol.replace('-', '_')}_raw.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw data not found at {path}. Run download_btc_data() first."
        )
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"📂 Loaded {len(df)} rows from {path}")
    return df
