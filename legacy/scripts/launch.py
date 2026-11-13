#!/usr/bin/env python3
"""Convenience wrapper for root launch.py."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import launch
    launch.main()

