"""
Streamlit Dashboard
====================
Interactive UI for cryptocurrency volatility forecasting.

Features:
    - BTC price & volatility charts
    - Model selection (GARCH / Random Forest / XGBoost / LSTM)
    - Forecast horizon slider
    - Model performance comparison
    - Connects to FastAPI backend (with local fallback)

Run:
    streamlit run dashboard/streamlit_app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SYMBOL, START_DATE, API_HOST, API_PORT,
    PROCESSED_DATA_DIR, MODELS_DIR,
)

# ─── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Crypto Volatility Forecasting",
    page_icon="₿",
    layout="wide",
)

API_URL = f"http://localhost:{API_PORT}"


# ─── Helper Functions ─────────────────────────────────────────

def check_api_health() -> bool:
    """Check if the FastAPI backend is running."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def load_processed_data() -> pd.DataFrame:
    """Load processed data from disk."""
    path = os.path.join(PROCESSED_DATA_DIR, "btc_processed.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])
    return pd.DataFrame()


def load_metrics() -> dict:
    """Load metrics from API or disk."""
    # Try API first
    try:
        r = requests.get(f"{API_URL}/metrics", timeout=5)
        if r.status_code == 200:
            return r.json().get("metrics", {})
    except Exception:
        pass

    # Fallback: load from CSV
    csv_path = os.path.join(MODELS_DIR, "metrics.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        result = {}
        for _, row in df.iterrows():
            result[row["model"]] = {
                "mae": row["mae"],
                "rmse": row["rmse"],
                "mape": row["mape"],
            }
        return result
    return {}


def get_forecast(model_type: str, horizon: int) -> dict:
    """Get forecast from API (with local fallback)."""
    try:
        r = requests.get(
            f"{API_URL}/predict",
            params={"model_type": model_type, "horizon": horizon},
            timeout=5,
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    # Local fallback from saved predictions
    joblib_path = os.path.join(MODELS_DIR, "predictions.joblib")
    if os.path.exists(joblib_path):
        try:
            import joblib
            preds_dict = joblib.load(joblib_path)
            if model_type in preds_dict:
                preds = preds_dict[model_type]
                forecast = preds[-horizon:].tolist() if len(preds) >= horizon else preds.tolist()
                return {"model": model_type, "horizon": len(forecast), "forecast": forecast, "source": "local"}
        except Exception:
            pass

    return {}


# ─── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/4/46/Bitcoin.svg", width=60)
    st.title("⚙️ Controls")

    api_status = check_api_health()
    if api_status:
        st.success("🟢 API Connected")
    else:
        st.warning("🟡 API Offline — Using local data")

    st.divider()

    model_choice = st.selectbox(
        "📈 Select Model",
        ["GARCH", "Random Forest", "XGBoost", "LSTM"],
        index=0,
    )

    forecast_horizon = st.slider(
        "🔮 Forecast Horizon (days)",
        min_value=7,
        max_value=180,
        value=30,
        step=7,
    )

    st.divider()

    if api_status:
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    r = requests.post(
                        f"{API_URL}/train",
                        params={"skip_lstm": False},
                        timeout=600,
                    )
                    if r.status_code == 200:
                        st.success("✅ Training complete!")
                        st.rerun()
                    else:
                        st.error(f"Training failed: {r.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.caption("Built with Streamlit + FastAPI")
    st.caption(f"Data: {SYMBOL}")


# ─── Main Content ─────────────────────────────────────────────

st.title("₿ Crypto Volatility Forecasting")
st.markdown(
    "Predict Bitcoin price volatility using **GARCH**, "
    "**Random Forest**, **XGBoost**, and **LSTM** models."
)

# ── Data Visualization ────────────────────────────────────────
df = load_processed_data()

if not df.empty:
    tab1, tab2, tab3 = st.tabs(["📈 Price & Volatility", "📊 Model Comparison", "🔮 Forecast"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("BTC-USD Closing Price")
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df["Date"], df["Close"], color="#F7931A", linewidth=0.8)
            ax1.set_ylabel("Price (USD)")
            ax1.set_title("Bitcoin Price History")
            fig1.autofmt_xdate()
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            st.subheader("Rolling Volatility")
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            for col_name in df.columns:
                if col_name.startswith("rv_") and col_name.endswith("d") and "ann" not in col_name:
                    window = col_name.replace("rv_", "").replace("d", "")
                    ax2.plot(df["Date"], df[col_name], label=f"{window}-day RV", linewidth=0.8)
            ax2.set_ylabel("Volatility")
            ax2.set_title("Realized Volatility")
            ax2.legend()
            fig2.autofmt_xdate()
            st.pyplot(fig2)
            plt.close(fig2)

        # Key statistics
        st.subheader("📋 Data Summary")
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Days", f"{len(df):,}")
        col_b.metric("Latest Close", f"${df['Close'].iloc[-1]:,.0f}")
        col_c.metric("Avg Daily Vol", f"{df['rv_7d'].mean():.4f}")
        col_d.metric("Max Daily Vol", f"{df['rv_7d'].max():.4f}")

    with tab2:
        st.subheader("🏆 Model Performance Comparison")
        metrics = load_metrics()

        if metrics:
            metrics_df = pd.DataFrame(metrics).T
            metrics_df.index.name = "Model"
            metrics_df = metrics_df.reset_index()
            metrics_df.columns = ["Model", "MAE", "RMSE", "MAPE (%)"]

            # Display table
            st.dataframe(
                metrics_df.style.highlight_min(
                    subset=["MAE", "RMSE", "MAPE (%)"],
                    color="#2ecc71",
                ),
                use_container_width=True,
            )

            # Bar chart comparison
            fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))
            for idx, metric in enumerate(["MAE", "RMSE", "MAPE (%)"]):
                colors = ["#3498db" if v != metrics_df[metric].min() else "#2ecc71"
                          for v in metrics_df[metric]]
                axes3[idx].bar(metrics_df["Model"], metrics_df[metric], color=colors)
                axes3[idx].set_title(metric, fontsize=14, fontweight="bold")
                axes3[idx].tick_params(axis="x", rotation=45)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.info("No metrics available. Train models first using the sidebar button or run the API.")

    with tab3:
        st.subheader(f"🔮 {model_choice} Forecast")

        forecast_data = get_forecast(model_choice, forecast_horizon)
        if forecast_data and "forecast" in forecast_data:
            forecast_vals = forecast_data["forecast"]

            fig4, ax4 = plt.subplots(figsize=(12, 5))
            ax4.plot(range(len(forecast_vals)), forecast_vals,
                     color="#e74c3c", linewidth=1.5, marker="o", markersize=3)
            ax4.fill_between(range(len(forecast_vals)), forecast_vals,
                             alpha=0.2, color="#e74c3c")
            ax4.set_title(f"{model_choice} Volatility Forecast ({forecast_horizon} days)",
                          fontsize=14, fontweight="bold")
            ax4.set_xlabel("Days Ahead")
            ax4.set_ylabel("Predicted Volatility")
            st.pyplot(fig4)
            plt.close(fig4)

            # Show stats
            col_x, col_y, col_z = st.columns(3)
            col_x.metric("Mean Forecast", f"{np.mean(forecast_vals):.6f}")
            col_y.metric("Max Forecast", f"{np.max(forecast_vals):.6f}")
            col_z.metric("Min Forecast", f"{np.min(forecast_vals):.6f}")

            if forecast_data.get("source") == "local":
                st.caption("ℹ️ Showing cached local forecasts (API offline).")
        else:
            if api_status:
                st.warning(
                    f"No predictions available for {model_choice}. "
                    "Train models first."
                )
            else:
                st.info(
                    "Start the API to enable live forecasting or run the training pipeline:\n\n"
                    "```bash\npython scripts/run_pipeline.py\n```"
                )

else:
    st.warning(
        "📂 No processed data found. Run the pipeline first:\n\n"
        "```bash\npython scripts/run_pipeline.py\n```"
    )
