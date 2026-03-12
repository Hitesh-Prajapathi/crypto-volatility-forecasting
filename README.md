# ₿ Crypto Volatility Forecasting

A modular machine learning system for forecasting Bitcoin price volatility using **statistical**, **machine learning**, and **deep learning** models — with a **FastAPI** backend and **Streamlit** dashboard.

---

## 💡 Origins & Inspiration

This repository is an evolution of a monolithic research notebook into a production-ready, modular ML pipeline. 

The original exploratory data analysis and initial model prototyping can be found in:
👉 [`notebooks/01_original_implementation.ipynb`](notebooks/01_original_implementation.ipynb)

By refactoring this monolith, the project demonstrates software engineering best practices for ML, including data pipelines, configuration management, model persistence, and API deployment.

---

## 🏗️ Architecture

```
                ┌───────────────────┐
                │   yfinance API    │
                └────────┬──────────┘
                         │
                         ▼
                ┌───────────────────┐
                │  Data Pipeline    │
                │  (data_loader.py) │
                └────────┬──────────┘
                         │
                         ▼
                ┌───────────────────┐
                │   Preprocessing   │
                │  + Feature Eng.   │
                └────────┬──────────┘
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐
     │  GARCH   │ │ RF / XGB │ │   LSTM   │
     │(stat.)   │ │  (ML)    │ │  (DL)    │
     └────┬─────┘ └────┬─────┘ └────┬─────┘
          └─────────────┼─────────────┘
                        ▼
                ┌───────────────────┐
                │   Evaluation      │
                │  (MAE/RMSE/MAPE)  │
                └────────┬──────────┘
                         │
                         ▼
                ┌───────────────────┐
                │   FastAPI         │
                │  /train /predict  │
                │  /metrics         │
                └────────┬──────────┘
                         │
                         ▼
                ┌───────────────────┐
                │  Streamlit UI     │
                │  Dashboard        │
                └───────────────────┘
```

---

## 📁 Project Structure

```
crypto-volatility-forecasting/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/                  # Downloaded price CSVs
│   └── processed/            # Feature-engineered data
│
├── models/                   # Saved model weights
│   ├── rf.pkl
│   ├── xgb.pkl
│   ├── lstm.keras
│   └── metrics.csv
│
├── figures/                  # Auto-generated plots
│
├── notebooks/
│   ├── 01_original_implementation.ipynb  # Original research notebook
│   └── exploration.ipynb                 # Cleaned EDA notebook
│
├── src/
│   ├── config.py             # Centralized configuration
│   ├── data_loader.py        # yfinance data download
│   ├── preprocessing.py      # Log returns, rolling volatility
│   ├── features.py           # Lag features, rolling stats
│   ├── models.py             # GARCH, Random Forest, XGBoost
│   ├── lstm_model.py         # LSTM architecture & training
│   ├── train.py              # Full training pipeline
│   ├── evaluate.py           # Metrics (MAE, RMSE, MAPE)
│   └── visualize.py          # Publication-quality plots
│
├── api/
│   └── main.py               # FastAPI backend
│
├── dashboard/
│   └── streamlit_app.py      # Interactive Streamlit UI
│
└── scripts/
    ├── run_pipeline.py       # CLI pipeline runner
    └── forecast.py           # Standalone forecast script
```

---

## 📊 Models

| Model | Type | Library | Description |
|-------|------|---------|-------------|
| **GARCH(1,1)** | Statistical | `arch` | Generalized Autoregressive Conditional Heteroskedasticity — captures volatility clustering |
| **Random Forest** | Machine Learning | `scikit-learn` | Ensemble of decision trees with lagged volatility features |
| **XGBoost** | Machine Learning | `xgboost` | Gradient-boosted trees — strong performance on tabular data |
| **LSTM** | Deep Learning | `tensorflow` | Long Short-Term Memory network — captures temporal dependencies |

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
cd crypto-volatility-forecasting
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
python scripts/run_pipeline.py
```

Options:
```bash
# Skip LSTM for faster execution
python scripts/run_pipeline.py --skip-lstm

# Use a different symbol or date range
python scripts/run_pipeline.py --symbol ETH-USD --start-date 2020-01-01
```

### 3. Start the API

```bash
uvicorn api.main:app --reload
```

API endpoints:
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/train` | Train all models |
| `GET` | `/predict?model_type=GARCH&horizon=30` | Get forecasts |
| `GET` | `/metrics` | Model comparison |

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Launch the Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Dashboard features:
- 📈 BTC price & volatility charts
- 📊 Model comparison table with bar charts
- 🔮 Interactive forecast with adjustable horizon
- ⚙️ One-click model training

### 5. Standalone Forecast

```bash
python scripts/forecast.py --model rf --horizon 30
```

---

## 📈 Example Results

After running the pipeline, model comparison metrics are saved to `models/metrics.csv`:

```
Model            MAE        RMSE       MAPE (%)
──────────────────────────────────────────────────
XGBoost          0.001234   0.001890   8.45
Random Forest    0.001345   0.002010   9.12
GARCH            0.002567   0.003400   15.20
LSTM             0.001456   0.002100   10.30
```

*Actual values depend on data and training run.*

Generated plots are saved to `figures/`:
- `btc_price_trend.png` — Bitcoin historical price
- `rolling_volatility.png` — Multi-window realized volatility
- `acf_pacf.png` — Serial correlation analysis
- `predictions_*.png` — Model-specific forecast vs actual
- `model_comparison.png` — All models overlaid
- `lstm_loss_curve.png` — LSTM training progress

---

## 🔧 Configuration

All hyperparameters are centralized in `src/config.py`:

```python
SYMBOL = "BTC-USD"
START_DATE = "2018-01-01"
ROLLING_WINDOWS = [7, 30]
LAG_PERIODS = [1, 5, 10]
TEST_RATIO = 0.2

RF_N_ESTIMATORS = 200
XGB_LEARNING_RATE = 0.05
LSTM_EPOCHS = 50
LSTM_SEQUENCE_LENGTH = 30
```

---

## 📝 License

This project is for educational and portfolio purposes.

---

*Built as a portfolio project demonstrating end-to-end ML system design.*
