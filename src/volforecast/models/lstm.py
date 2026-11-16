"""LSTM challenger for h-day realized variance (requires the optional ``dl`` extra: torch).

The model reads the last ``lookback`` days of a per-day realized-measure vector and regresses the
log of the h-day cumulative RV target, exactly the target of the GBMs and log-HAR:

    log( sum_{k=1..h} RV_{t+k} ) = f( x_{t-L+1}, ..., x_t ) + e,   forecast: exp( f + sigma^2 / 2 ),

where ``sigma^2`` is the mean squared log residual on the inner validation block (the last
``val_frac`` of the usable training rows, chronologically). The weights never see that block
except through early stopping, so the log-normal bias correction is (nearly) out of sample.

Per-day features (all causal, one row per panel row; no rolling windows, so a forecast at origin
``t`` depends only on the last ``lookback`` rows): ``log rv``, ``log bpv``, ``log(jump + eps)``,
``log rs_pos``, ``log rs_neg``, ``sqrt(rq)``, ``ret``, ``|ret|`` and ``sin/cos`` of the weekday of
the day itself (the next day's weekday is a deterministic function of it).

Leakage control (the legacy A5 bug was a scaler fitted on the full sample):

* feature standardisation (mean/std) and target standardisation are estimated ONCE per ``fit`` on
  the panel handed to ``fit`` (the training window) and stored on the instance;
* ``predict_variance`` only applies the stored statistics, it never re-estimates them;
* early stopping uses the chronologically last ``val_frac`` of the usable rows, with an h-day
  embargo between the train and validation blocks (as in `gbm.py`); only minibatches inside the
  training block are shuffled.

Reproducibility: weights, minibatch order and dropout are driven by ``seed`` (torch, numpy and
python RNG states are saved, seeded and restored around each call, so the model does not disturb
global RNG state); deterministic algorithms are requested and torch runs on CPU with
``num_threads`` threads (default 1; multithreaded reductions may differ in the last bits).
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from torch import nn

from volforecast.contracts import (
    COL_BPV,
    COL_JUMP,
    COL_RET,
    COL_RQ,
    COL_RS_NEG,
    COL_RS_POS,
    COL_RV,
)
from volforecast.logging import get_logger
from volforecast.models.base import BaseVolModel, BoolArray, FloatArray, ModelInputError

log = get_logger(__name__)

F32Array = NDArray[np.float32]
JUMP_EPS = 1e-8
MIN_TRAIN_ROWS = 120
MIN_LOOKBACK, MAX_LOOKBACK = 22, 66
MAX_STD_LOG_PRED = 10.0  # |standardised log forecast| clip, guards exp() overflow
FEATURE_NAMES: tuple[str, ...] = (
    "log_rv",
    "log_bpv",
    "log_jump",
    "log_rs_pos",
    "log_rs_neg",
    "sqrt_rq",
    "ret",
    "abs_ret",
    "dow_sin",
    "dow_cos",
)


def _safe_log(x: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(x)
    out[~np.isfinite(out)] = np.nan
    return np.asarray(out, dtype=float)


def lstm_features(panel: pd.DataFrame) -> FloatArray:
    """Per-day feature matrix ``(len(panel), len(FEATURE_NAMES))``; NaN where undefined. Causal."""

    def col(name: str) -> FloatArray:
        return panel[name].to_numpy(dtype=float)

    ret = col(COL_RET)
    dow = np.asarray(pd.DatetimeIndex(panel.index).day_of_week, dtype=float)
    with np.errstate(invalid="ignore"):
        sqrt_rq = np.sqrt(col(COL_RQ))
    cols = [
        _safe_log(col(COL_RV)),
        _safe_log(col(COL_BPV)),
        _safe_log(col(COL_JUMP) + JUMP_EPS),
        _safe_log(col(COL_RS_POS)),
        _safe_log(col(COL_RS_NEG)),
        sqrt_rq,
        ret,
        np.abs(ret),
        np.sin(2.0 * np.pi * dow / 7.0),
        np.cos(2.0 * np.pi * dow / 7.0),
    ]
    out: FloatArray = np.column_stack(cols).astype(float)
    return out


@contextmanager
def _runtime(seed: int, num_threads: int) -> Iterator[None]:
    """Seed torch/numpy/python, force determinism and thread count; restore everything after."""
    py_state, np_state = random.getstate(), np.random.get_state()  # noqa: NPY002
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    random.seed(seed)
    np.random.seed(seed % (2**32))  # noqa: NPY002
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
    finally:
        torch.use_deterministic_algorithms(prev_det, warn_only=prev_warn)
        torch.set_num_threads(prev_threads)
        random.setstate(py_state)
        np.random.set_state(np_state)  # noqa: NPY002


class _Net(nn.Module):
    def __init__(self, n_features: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        y: torch.Tensor = self.head(self.drop(out[:, -1, :])).squeeze(-1)
        return y


class LSTMVol(BaseVolModel):
    """Small LSTM on the last ``lookback`` days of realized measures; log-target, bias-corrected."""

    name = "lstm"
    required_columns = (COL_RV, COL_BPV, COL_JUMP, COL_RS_POS, COL_RS_NEG, COL_RQ, COL_RET)

    def __init__(
        self,
        lookback: int = 22,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 50,
        patience: int = 5,
        val_frac: float = 0.15,
        seed: int = 0,
        num_threads: int = 1,
        drop_incomplete_targets: bool = True,
    ) -> None:
        super().__init__(drop_incomplete_targets)
        if not MIN_LOOKBACK <= lookback <= MAX_LOOKBACK:
            raise ValueError(f"lookback must lie in [{MIN_LOOKBACK}, {MAX_LOOKBACK}]")
        if num_layers not in (1, 2):
            raise ValueError("num_layers must be 1 or 2")
        if hidden_size < 1 or batch_size < 1 or max_epochs < 1 or patience < 1 or num_threads < 1:
            raise ValueError(
                "hidden_size, batch_size, max_epochs, patience, num_threads must be >= 1"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if not 0.0 < val_frac < 0.5:
            raise ValueError("val_frac must lie in (0, 0.5)")
        self.lookback = lookback
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.seed = seed
        self.num_threads = num_threads
        self._net: _Net | None = None
        self._mu: FloatArray | None = None  # feature mean, fitted on the training window only
        self._sd: FloatArray | None = None
        self._z_mu = float("nan")  # log-target mean/std on the training rows
        self._z_sd = float("nan")
        self._sigma2 = float("nan")
        self._n_train = 0
        self._n_val = 0
        self._epochs_run = 0
        self._best_epoch = 0
        self._best_val_loss = float("nan")

    # ------------------------------------------------------------------ helpers
    def _scale(self, feats: FloatArray) -> FloatArray:
        assert self._mu is not None
        assert self._sd is not None
        x = (feats - self._mu) / self._sd
        x[~np.isfinite(x)] = 0.0  # undefined entries become the training mean
        out: FloatArray = x.astype(np.float32)  # type: ignore[assignment]
        return out

    def _windows(self, scaled: FloatArray) -> FloatArray:
        """Sequences ``(n - L + 1, L, F)``; window k ends at panel row ``k + L - 1``."""
        w = sliding_window_view(scaled, self.lookback, axis=0)  # (n-L+1, F, L)
        out: FloatArray = np.ascontiguousarray(w.transpose(0, 2, 1))
        return out

    def _forward(self, net: _Net, x: FloatArray) -> FloatArray:
        net.eval()
        with torch.no_grad():
            out = net(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)))
        return np.asarray(out.numpy(), dtype=float)

    # --------------------------------------------------------------------- fitting
    def _fit(self, panel: pd.DataFrame, h: int, y: FloatArray, ok: BoolArray) -> None:
        self._net = None
        raw = lstm_features(panel)
        L = self.lookback
        # scaler: training-window statistics only, stored for predict
        with np.errstate(invalid="ignore"), _quiet_nan_warnings():
            mu = np.nanmean(raw, axis=0)
            sd = np.nanstd(raw, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        self._mu, self._sd = mu, sd

        use = ok & (y > 0.0) & np.isfinite(raw[:, 0])
        use[: L - 1] = False
        pos = np.flatnonzero(use)
        n = pos.size
        if n < MIN_TRAIN_ROWS:
            raise ModelInputError(
                f"{self.name}: only {n} usable training rows (need >= {MIN_TRAIN_ROWS})"
            )
        windows = self._windows(self._scale(raw))
        z = np.log(y[pos])
        cut = round(n * (1.0 - self.val_frac))
        boundary = pos[cut]
        tr = pos[:cut] <= boundary - h  # h-day embargo: train targets are known by `boundary`
        tr_pos, va_pos = pos[:cut][tr], pos[cut:]
        z_tr, z_va = z[:cut][tr], z[cut:]
        self._z_mu = float(z_tr.mean())
        self._z_sd = float(z_tr.std()) if z_tr.std() > 1e-12 else 1.0
        Xtr, Xva = windows[tr_pos - (L - 1)], windows[va_pos - (L - 1)]
        ytr = ((z_tr - self._z_mu) / self._z_sd).astype(np.float32)
        yva = ((z_va - self._z_mu) / self._z_sd).astype(np.float32)

        with _runtime(self.seed, self.num_threads):
            net, best_state = self._train(Xtr, ytr, Xva, yva)
        net.load_state_dict(best_state)
        resid = (yva - self._forward(net, Xva)) * self._z_sd
        self._sigma2 = float(np.mean(resid**2))
        self._net = net
        self._n_train, self._n_val = int(tr_pos.size), int(va_pos.size)
        self._fallback = float(np.exp(z.mean() + 0.5 * self._sigma2))

    def _train(
        self, Xtr: FloatArray, ytr: F32Array, Xva: FloatArray, yva: F32Array
    ) -> tuple[_Net, dict[str, torch.Tensor]]:
        net = _Net(Xtr.shape[2], self.hidden_size, self.num_layers, self.dropout)
        opt = torch.optim.Adam(
            net.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()
        gen = torch.Generator().manual_seed(self.seed)
        xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
        xv, yv = torch.from_numpy(Xva), torch.from_numpy(yva)
        best = math.inf
        best_state = copy.deepcopy(net.state_dict())
        self._best_epoch, bad = 0, 0
        for epoch in range(1, self.max_epochs + 1):
            net.train()
            perm = torch.randperm(xt.shape[0], generator=gen)  # shuffle inside train only
            for i in range(0, xt.shape[0], self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(xt[idx]), yt[idx])
                loss.backward()  # type: ignore[no-untyped-call]
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                val = float(loss_fn(net(xv), yv))
            self._epochs_run = epoch
            if val < best - 1e-6:
                best, bad, self._best_epoch = val, 0, epoch
                best_state = copy.deepcopy(net.state_dict())
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self._best_val_loss = best
        return net, best_state

    # ------------------------------------------------------------------ prediction
    def _predict(self, panel: pd.DataFrame, h: int) -> float:
        assert self._net is not None
        if len(panel) < self.lookback:
            raise ModelInputError(
                f"{self.name}: need >= {self.lookback} rows of history, got {len(panel)}"
            )
        feats = lstm_features(panel.iloc[-self.lookback :])
        x = self._scale(feats)[None, :, :]  # stored training-window scaler, never refitted
        with _runtime(self.seed, self.num_threads):
            zs = float(self._forward(self._net, x)[0])
        zs = float(np.clip(zs, -MAX_STD_LOG_PRED, MAX_STD_LOG_PRED))
        return math.exp(zs * self._z_sd + self._z_mu + 0.5 * self._sigma2)

    def get_params(self) -> dict[str, Any]:
        def lst(a: FloatArray | None) -> list[float] | None:
            return None if a is None else [float(v) for v in a]

        def fin(v: float) -> float | None:
            return v if math.isfinite(v) else None

        return {
            **self._base_params(),
            "lookback": self.lookback,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "val_frac": self.val_frac,
            "seed": self.seed,
            "num_threads": self.num_threads,
            "features": list(FEATURE_NAMES),
            "scaler_mean": lst(self._mu),
            "scaler_std": lst(self._sd),
            "target_mean": fin(self._z_mu),
            "target_std": fin(self._z_sd),
            "sigma2": fin(self._sigma2),
            "n_train": self._n_train,
            "n_val": self._n_val,
            "epochs_run": self._epochs_run,
            "best_epoch": self._best_epoch,
            "best_val_loss": fin(self._best_val_loss),
        }


@contextmanager
def _quiet_nan_warnings() -> Iterator[None]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN column in nanmean/nanstd
        yield
