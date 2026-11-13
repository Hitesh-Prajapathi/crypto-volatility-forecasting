"""
Central configuration for the Crypto Volatility Forecasting project.

All tunable parameters are defined here to keep the codebase clean
and avoid magic numbers scattered across modules.
"""

import os

# ──────────────────────────────────────────────
# macOS / Apple Silicon Anti-Deadlock Config
# Prevents libomp deadlocks when mixing XGBoost and PyTorch
# ──────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")

# ──────────────────────────────────────────────
# Data Collection
# ──────────────────────────────────────────────
SYMBOL = "BTC-USD"
START_DATE = "2018-01-01"

# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────
ROLLING_WINDOWS = [7, 30]
ANNUALIZATION_FACTOR = 252  # trading days per year

# ──────────────────────────────────────────────
# Feature Engineering
# ──────────────────────────────────────────────
LAG_PERIODS = [1, 5, 10]
TARGET_COLUMN = "rv_30d"  # 30-day realized volatility

# ──────────────────────────────────────────────
# Train / Test Split
# ──────────────────────────────────────────────
TEST_RATIO = 0.2  # 80/20 split

# ──────────────────────────────────────────────
# GARCH
# ──────────────────────────────────────────────
GARCH_P = 1
GARCH_Q = 1

# ──────────────────────────────────────────────
# Random Forest
# ──────────────────────────────────────────────
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42

# ──────────────────────────────────────────────
# XGBoost
# ──────────────────────────────────────────────
XGB_N_ESTIMATORS = 200
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05
XGB_RANDOM_STATE = 42

# ──────────────────────────────────────────────
# LSTM
# ──────────────────────────────────────────────
LSTM_SEQUENCE_LENGTH = 15
LSTM_UNITS = 32
LSTM_DROPOUT = 0.2
LSTM_EPOCHS = 20
LSTM_BATCH_SIZE = 64
LSTM_LEARNING_RATE = 0.001

# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ──────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_STYLE = "seaborn-v0_8-darkgrid"
