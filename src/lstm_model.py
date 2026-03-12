"""
LSTM Volatility Predictor
==========================
Deep learning model for time-series volatility forecasting
using PyTorch LSTM architecture.

Uses PyTorch instead of TensorFlow for reliable Apple Silicon support.
"""

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    LSTM_SEQUENCE_LENGTH,
    LSTM_UNITS,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_LEARNING_RATE,
    MODELS_DIR,
)

# Use MPS (Apple Silicon GPU) if available, else CPU
DEVICE = torch.device("cpu")  # CPU is fast enough for this small model


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = LSTM_SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for LSTM input.

    Args:
        X: Feature matrix of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        seq_len: Number of time steps per sequence.

    Returns:
        Tuple of (X_seq, y_seq) where:
            X_seq has shape (n_sequences, seq_len, n_features)
            y_seq has shape (n_sequences,)
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_len):
        X_seq.append(X[i : i + seq_len])
        y_seq.append(y[i + seq_len])
    return np.array(X_seq), np.array(y_seq)


class LSTMModel(nn.Module):
    """Simple LSTM for volatility prediction.

    Architecture: Input → LSTM(32) → Dropout(0.2) → Dense(8) → Dense(1)
    """

    def __init__(self, input_size: int, hidden_size: int = LSTM_UNITS,
                 dropout: float = LSTM_DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        last = lstm_out[:, -1, :]
        out = self.dropout(last)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out.squeeze(-1)


def build_lstm_model(input_size: int) -> LSTMModel:
    """Build the LSTM architecture.

    Args:
        input_size: Number of input features.

    Returns:
        LSTMModel instance.
    """
    model = LSTMModel(input_size=input_size).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 LSTM model built  |  {total_params:,} parameters  |  device={DEVICE}")
    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = LSTM_EPOCHS,
    batch_size: int = LSTM_BATCH_SIZE,
    save: bool = True,
) -> Tuple[LSTMModel, dict]:
    """Train the LSTM model with early stopping.

    Args:
        X_train: Training sequences (n, seq_len, features).
        y_train: Training targets.
        X_val: Validation sequences.
        y_val: Validation targets.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        save: Whether to save the trained model.

    Returns:
        Tuple of (trained model, training history dict).
    """
    # Convert to tensors
    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    y_tr = torch.FloatTensor(y_train).to(DEVICE)
    X_vl = torch.FloatTensor(X_val).to(DEVICE)
    y_vl = torch.FloatTensor(y_val).to(DEVICE)

    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = build_lstm_model(input_size=X_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LEARNING_RATE)
    criterion = nn.MSELoss()

    history = {"loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 5
    best_state = None

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_train_loss = epoch_loss / n_batches

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_preds = model(X_vl)
            val_loss = criterion(val_preds, y_vl).item()

        history["loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)

        print(
            f"  Epoch {epoch + 1:2d}/{epochs}  |  "
            f"loss={avg_train_loss:.6f}  val_loss={val_loss:.6f}",
            flush=True,
        )

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  ⏹️  Early stopping at epoch {epoch + 1}")
                break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    print(f"✅ LSTM trained  |  Best val_loss: {best_val_loss:.6f}")

    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, "lstm.pt")
        torch.save(model.state_dict(), path)
        print(f"💾 Saved → {path}")

    return model, history


def predict_lstm(
    model: LSTMModel,
    X_test: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Generate predictions and inverse-transform to original scale.

    Args:
        model: Trained LSTM model.
        X_test: Test sequences (n, seq_len, features).
        scaler: Fitted MinMaxScaler used on the target during training.

    Returns:
        Array of predictions in original scale.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X_test).to(DEVICE)
        y_pred_scaled = model(X_t).cpu().numpy().reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    return y_pred


def load_lstm_model(input_size: int) -> LSTMModel:
    """Load saved LSTM model from disk.

    Args:
        input_size: Number of input features (must match saved model).

    Returns:
        Loaded LSTMModel.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    path = os.path.join(MODELS_DIR, "lstm.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"LSTM model not found: {path}")
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model
