#!/usr/bin/env bash
# ==============================================================================
# 🚀 Crypto Volatility Forecasting — Unified Launch Script
# ==============================================================================
# Launches all components (Data Pipeline, LSTM, Models, FastAPI, and Streamlit)
#
# Usage:
#   ./launch.sh
#   ./launch.sh --retrain
#   ./launch.sh --no-browser
# ==============================================================================

set -e

# Change to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Anti-deadlock settings for macOS / Apple Silicon
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONUNBUFFERED=1

# Find appropriate Python binary with dependencies
PYTHON_CMD="python3"

if [ -f "$HOME/miniforge3/envs/ai/bin/python3" ]; then
    PYTHON_CMD="$HOME/miniforge3/envs/ai/bin/python3"
elif [ -f "$HOME/miniconda3/envs/ai/bin/python3" ]; then
    PYTHON_CMD="$HOME/miniconda3/envs/ai/bin/python3"
elif command -v conda >/dev/null 2>&1 && conda env list | grep -q "ai "; then
    exec conda run -n ai python launch.py "$@"
fi

# Run unified launcher
exec "$PYTHON_CMD" launch.py "$@"

