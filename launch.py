#!/usr/bin/env python3
"""
🚀 Crypto Volatility Forecasting — Unified Application Launcher
================================================================
Launches all components of the crypto volatility forecasting project:
  1. Data Pipeline & Model Verification (including PyTorch LSTM)
  2. FastAPI Backend Server (REST API on port 8000)
  3. Streamlit Interactive Dashboard UI (Web app on port 8501)

Usage:
    python launch.py                  # Launch all components (including LSTM)
    python launch.py --retrain        # Retrain all models (including LSTM) before launch
    python launch.py --no-browser     # Don't auto-open browser
    python launch.py --api-only       # Run only FastAPI backend
    python launch.py --dashboard-only # Run only Streamlit UI
"""

import argparse
import functools
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

# Force immediate flushing of all stdout prints
print = functools.partial(print, flush=True)

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Color formatting for terminal
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def print_banner():
    banner = f"""
{CYAN}{BOLD}===============================================================================
       ₿  CRYPTO VOLATILITY FORECASTING SYSTEM  ₿
==============================================================================={RESET}
  {BOLD}Architectural Components:{RESET}
    🧠 {GREEN}Models:{RESET}      GARCH(1,1) • Random Forest • XGBoost • {MAGENTA}{BOLD}PyTorch LSTM{RESET}
    🔌 {GREEN}Backend:{RESET}     FastAPI Microservice (Uvicorn REST API)
    📊 {GREEN}Frontend:{RESET}    Streamlit Interactive Analytics Dashboard
    📈 {GREEN}Pipeline:{RESET}    Yahoo Finance Data Ingestion → HAR-RV Features
===============================================================================
"""
    print(banner)


def check_and_switch_environment():
    """Ensure we are running with a Python interpreter that has required packages."""
    required = ["torch", "fastapi", "uvicorn", "streamlit", "sklearn", "xgboost", "arch", "yfinance"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return sys.executable

    # Check known virtual / conda environments
    home = Path.home()
    candidates = [
        home / "miniforge3" / "envs" / "ai" / "bin" / "python3",
        home / "miniforge3" / "envs" / "ai" / "bin" / "python",
        home / "miniconda3" / "envs" / "ai" / "bin" / "python3",
        home / "anaconda3" / "envs" / "ai" / "bin" / "python3",
    ]

    # Also check conda env list if conda is installed
    try:
        proc = subprocess.run(["conda", "info", "--envs"], capture_output=True, text=True, check=False)
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split()
                env_path = parts[-1]
                cand = Path(env_path) / "bin" / "python3"
                if cand not in candidates:
                    candidates.append(cand)
    except Exception:
        pass

    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            # Test if this candidate has the packages
            test_cmd = [str(cand), "-c", "import torch, fastapi, streamlit; print('OK')"]
            res = subprocess.run(test_cmd, capture_output=True, text=True)
            if res.returncode == 0 and "OK" in res.stdout:
                print(f"{YELLOW}⚡ Auto-detected compatible Python environment: {cand}{RESET}")
                os.execv(str(cand), [str(cand)] + sys.argv)

    print(f"{RED}❌ Missing dependencies: {', '.join(missing)}{RESET}")
    print(f"Please install them or activate the appropriate conda/venv environment:")
    print(f"   conda activate ai  # or: pip install -r requirements.txt")
    sys.exit(1)


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a network port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def kill_process_on_port(port: int):
    """Kill any process holding the given port (macOS/Linux)."""
    try:
        proc = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        pids = proc.stdout.strip().split()
        for pid in pids:
            if pid:
                print(f"{YELLOW}⚠️  Terminating existing process (PID {pid}) on port {port}...{RESET}")
                os.kill(int(pid), signal.SIGTERM)
                time.sleep(0.5)
    except Exception:
        pass


def verify_or_train_models(force_retrain: bool = False, skip_lstm: bool = False):
    """Verify that models and predictions exist, training them (including LSTM) if needed."""
    models_dir = PROJECT_ROOT / "models"
    lstm_path = models_dir / "lstm.pt"
    rf_path = models_dir / "rf.pkl"
    xgb_path = models_dir / "xgb.pkl"
    metrics_path = models_dir / "metrics.csv"
    data_path = PROJECT_ROOT / "data" / "processed" / "btc_processed.csv"

    needs_training = force_retrain or not (
        lstm_path.exists() and rf_path.exists() and xgb_path.exists() and metrics_path.exists() and data_path.exists()
    )

    if needs_training:
        print(f"\n{YELLOW}{BOLD}⚙️  Models not fully trained or retrain requested.{RESET}")
        print(f"{CYAN}🏋️ Running end-to-end training pipeline (INCLUDING PYTORCH LSTM)...{RESET}\n")

        from src.train import run_training_pipeline
        results = run_training_pipeline(skip_lstm=skip_lstm)

        print(f"\n{GREEN}{BOLD}✅ Training complete! Models ready:{RESET}")
        print(results["metrics"].to_string(index=False))
        return results
    else:
        print(f"{GREEN}✅ Verified existing trained models:{RESET}")
        print(f"   • {BOLD}LSTM (PyTorch):{RESET}      {lstm_path} ({lstm_path.stat().st_size:,} bytes)")
        print(f"   • {BOLD}Random Forest:{RESET}       {rf_path} ({rf_path.stat().st_size:,} bytes)")
        print(f"   • {BOLD}XGBoost:{RESET}             {xgb_path} ({xgb_path.stat().st_size:,} bytes)")
        print(f"   • {BOLD}Metrics & Data:{RESET}      {metrics_path}")
        return None


def wait_for_url(url: str, timeout: float = 25.0, name: str = "Service") -> bool:
    """Wait until an HTTP endpoint returns a successful status."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LauncherHealthCheck"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if 200 <= resp.status < 400:
                    return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


def stream_process_output(pipe, prefix: str, color_code: str):
    """Stream process logs to terminal with colored prefix."""
    try:
        for line in iter(pipe.readline, ""):
            if not line:
                break
            stripped = line.rstrip()
            if stripped:
                print(f"{color_code}[{prefix}]{RESET} {stripped}")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Unified Launcher for Crypto Volatility Forecasting System"
    )
    parser.add_argument(
        "--retrain",
        "--train",
        dest="retrain",
        action="store_true",
        help="Force retrain all models (including PyTorch LSTM) before launch",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        default=False,
        help="Skip LSTM during training (default: False, LSTM is always included)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser",
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Launch only the FastAPI backend server",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Launch only the Streamlit dashboard UI",
    )
    parser.add_argument(
        "--port-api",
        type=int,
        default=8000,
        help="Port for FastAPI backend (default: 8000)",
    )
    parser.add_argument(
        "--port-dashboard",
        type=int,
        default=8501,
        help="Port for Streamlit dashboard (default: 8501)",
    )
    parser.add_argument(
        "--kill-existing",
        action="store_true",
        default=True,
        help="Terminate any existing processes on target ports",
    )

    args = parser.parse_args()

    # Step 1: Environment verification
    py_exec = check_and_switch_environment()

    # Step 2: Banner
    print_banner()

    # Step 3: Verify or train models (including LSTM)
    if not args.dashboard_only:
        verify_or_train_models(force_retrain=args.retrain, skip_lstm=args.skip_lstm)

    # Step 4: Check ports & cleanup if requested
    if args.kill_existing:
        if not args.dashboard_only and is_port_in_use(args.port_api):
            kill_process_on_port(args.port_api)
        if not args.api_only and is_port_in_use(args.port_dashboard):
            kill_process_on_port(args.port_dashboard)

    processes = []

    def shutdown_all(*_):
        print(f"\n{YELLOW}🛑 Shutting down all services gracefully...{RESET}")
        for p, name in processes:
            if p.poll() is None:
                print(f"   Stopping {name} (PID {p.pid})...")
                p.terminate()
        for p, _ in processes:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
        print(f"{GREEN}👋 All components stopped. Goodbye!{RESET}\n")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_all)
    signal.signal(signal.SIGTERM, shutdown_all)

    # Step 5: Start FastAPI Backend
    api_proc = None
    if not args.dashboard_only:
        print(f"\n{CYAN}🚀 Launching FastAPI Backend on port {args.port_api}...{RESET}")
        api_env = os.environ.copy()
        api_env["PYTHONUNBUFFERED"] = "1"
        api_cmd = [
            py_exec,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(args.port_api),
            "--log-level",
            "info",
        ]
        api_proc = subprocess.Popen(
            api_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=api_env,
        )
        processes.append((api_proc, "FastAPI Backend"))

        t_api = threading.Thread(
            target=stream_process_output,
            args=(api_proc.stdout, "API", CYAN),
            daemon=True,
        )
        t_api.start()

        # Wait for API to respond
        health_url = f"http://127.0.0.1:{args.port_api}/health"
        print(f"   Waiting for FastAPI health check ({health_url})...")
        if wait_for_url(health_url, timeout=20.0, name="FastAPI"):
            print(f"{GREEN}✅ FastAPI backend is healthy and responding!{RESET}")
        else:
            print(f"{RED}⚠️  FastAPI did not respond in time. Checking process status...{RESET}")
            if api_proc.poll() is not None:
                print(f"{RED}FastAPI exited prematurely with code {api_proc.returncode}{RESET}")
                shutdown_all()

    # Step 6: Start Streamlit Dashboard UI
    dash_proc = None
    if not args.api_only:
        print(f"\n{MAGENTA}🚀 Launching Streamlit Dashboard on port {args.port_dashboard}...{RESET}")
        dash_env = os.environ.copy()
        dash_env["PYTHONUNBUFFERED"] = "1"
        dash_cmd = [
            py_exec,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "dashboard" / "streamlit_app.py"),
            "--server.port",
            str(args.port_dashboard),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]
        dash_proc = subprocess.Popen(
            dash_cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=dash_env,
        )
        processes.append((dash_proc, "Streamlit Dashboard"))

        t_dash = threading.Thread(
            target=stream_process_output,
            args=(dash_proc.stdout, "UI", MAGENTA),
            daemon=True,
        )
        t_dash.start()

        dash_url = f"http://localhost:{args.port_dashboard}"
        print(f"   Waiting for Streamlit dashboard ({dash_url})...")
        if wait_for_url(dash_url, timeout=25.0, name="Streamlit"):
            print(f"{GREEN}✅ Streamlit dashboard is online!{RESET}")
        else:
            print(f"{YELLOW}Streamlit starting up...{RESET}")

    # Step 7: Print status dashboard & open browser
    dashboard_url = f"http://localhost:{args.port_dashboard}"
    api_docs_url = f"http://localhost:{args.port_api}/docs"
    api_metrics_url = f"http://localhost:{args.port_api}/metrics"

    print("\n" + "=" * 79)
    print(f"{GREEN}{BOLD}✨ ALL COMPONENTS ACTIVE & RUNNING! ✨{RESET}")
    print("=" * 79)
    if not args.api_only:
        print(f"  📊 {BOLD}Streamlit UI Dashboard:{RESET}  {CYAN}{BOLD}{dashboard_url}{RESET}")
    if not args.dashboard_only:
        print(f"  🔌 {BOLD}FastAPI Backend Server:{RESET}   {CYAN}http://localhost:{args.port_api}{RESET}")
        print(f"  📖 {BOLD}Interactive Swagger Docs:{RESET} {CYAN}{api_docs_url}{RESET}")
        print(f"  🧠 {BOLD}Model Volatility API:{RESET}     {CYAN}{api_metrics_url}{RESET}")
        print(f"  🔮 {BOLD}Active Deep Learning:{RESET}     {MAGENTA}PyTorch LSTM (Sequence=15, Units=32){RESET}")
    print("=" * 79)
    print(f"{DIM}Press Ctrl+C at any time to shut down all components cleanly.{RESET}\n")

    if not args.api_only and not args.no_browser:
        try:
            webbrowser.open(dashboard_url)
        except Exception:
            pass

    # Step 8: Keep running and monitor child processes
    try:
        while True:
            for p, name in processes:
                ret = p.poll()
                if ret is not None:
                    print(f"{RED}⚠️  Component '{name}' exited with status code {ret}.{RESET}")
                    shutdown_all()
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_all()


if __name__ == "__main__":
    main()
