"""
Pipeline Runner Script
=======================
CLI entry point to execute the full training pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --skip-lstm
    python scripts/run_pipeline.py --symbol ETH-USD --start-date 2020-01-01
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import SYMBOL, START_DATE
from src.train import run_training_pipeline


def main():
    """Parse arguments and run the training pipeline."""
    parser = argparse.ArgumentParser(
        description="Crypto Volatility Forecasting — Training Pipeline"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=SYMBOL,
        help=f"Ticker symbol (default: {SYMBOL})",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=START_DATE,
        help=f"Data start date (default: {START_DATE})",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip LSTM training for faster execution",
    )

    args = parser.parse_args()

    results = run_training_pipeline(
        symbol=args.symbol,
        start_date=args.start_date,
        skip_lstm=args.skip_lstm,
    )

    print("\n📋 Final Metrics:")
    print(results["metrics"].to_string(index=False))
    print(f"\n🎯 Models trained: {list(results['predictions'].keys())}")


if __name__ == "__main__":
    main()
