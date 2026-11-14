"""Gradient-boosted-tree challengers (LightGBM, XGBoost) for h-day realized variance.

Both models share `_GBMVol`: a log-target direct h-step regression on HAR-style features,

    log( sum_{k=1..h} RV_{t+k} ) = f( x_t ) + e,        prediction: exp( f(x_t) + sigma^2 / 2 ),

where ``sigma^2`` is the mean squared **out-of-fold** log residual (a model fitted on the first
80% of the usable training rows, scored on the last 20%, with an ``h``-day embargo between the
two blocks), which makes the log-normal bias correction honest instead of in-sample.

Features at origin ``t`` use only panel rows dated <= t (rolling windows are backward-looking and
the day-of-week feature is the weekday of ``t + 1``, a pure calendar fact):

* HAR aggregates (1/5/22-day means) of ``log rv``, ``log bpv``, ``log rs_pos``, ``log rs_neg``
  and ``sqrt(rq)``;
* ``log(jump + eps)``, ``|ret|``, ``ret`` and ``ret * 1[ret < 0]`` (leverage);
* weekday of ``t + 1`` (0 = Monday);
* 1-, 5- and 22-day changes of ``log rv``.

Training rows follow the `BaseVolModel` convention (targets that are unobserved or contain an
incomplete day are dropped, which gives the ``h-1`` embargo). Hyperparameters are fixed and
single-threaded, so fits are deterministic.

Extension point for exogenous features (ablations: DVOL, funding, open interest, ...): register a
function with `register_exog_features` and pass ``exog="<name>"``. The function maps the panel
(rows <= t only) to a frame indexed like the panel; its columns are appended to the feature
matrix with an ``exog_`` prefix. The function is responsible for lagging so that the value on
row ``t`` was known at the end of day ``t``. No external data is fetched by this module.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from volforecast.contracts import (
    COL_BPV,
    COL_JUMP,
    COL_RET,
    COL_RQ,
    COL_RS_NEG,
    COL_RS_POS,
    COL_RV,
)
from volforecast.features.har import har_aggregates
from volforecast.logging import get_logger
from volforecast.models.base import BaseVolModel, BoolArray, FloatArray, ModelInputError

log = get_logger(__name__)

#: rows of history needed to evaluate the features at the origin (22-day window + 22-day change).
PREDICT_LOOKBACK = 60
JUMP_EPS = 1e-8
MIN_TRAIN_ROWS = 120
MIN_OOF_ROWS = 30
_WINDOWS = {"d": 1, "w": 5, "m": 22}

ExogFn = Callable[[pd.DataFrame], pd.DataFrame]
EXOG_FEATURES: dict[str, ExogFn] = {}


def register_exog_features(name: str, fn: ExogFn) -> None:
    """Register an exogenous-feature builder for the ablation hook (see module docstring)."""
    EXOG_FEATURES[name] = fn


def _safe_log(x: pd.Series) -> pd.Series:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = pd.Series(np.log(x.to_numpy(dtype=float)), index=x.index)
    return out.where(np.isfinite(out))


def _har_block(x: pd.Series, name: str) -> dict[str, pd.Series]:
    return {
        f"{name}_{k}": x.rolling(w, min_periods=w).mean() if w > 1 else x
        for k, w in _WINDOWS.items()
    }


def gbm_features(panel: pd.DataFrame, exog: str | None = None) -> pd.DataFrame:
    """Feature matrix, one row per panel row (NaN where a window is incomplete). Causal."""
    cols: dict[str, pd.Series] = {}
    lrv = _safe_log(panel[COL_RV])
    cols.update(_har_block(lrv, "log_rv"))
    for col in (COL_BPV, COL_RS_POS, COL_RS_NEG):
        cols.update(_har_block(_safe_log(panel[col]), f"log_{col}"))
    with np.errstate(invalid="ignore"):
        sqrt_rq = pd.Series(np.sqrt(panel[COL_RQ].to_numpy(dtype=float)), index=panel.index)
    cols.update(_har_block(sqrt_rq, "sqrt_rq"))
    cols["log_jump"] = _safe_log(panel[COL_JUMP].astype(float) + JUMP_EPS)
    ret = panel[COL_RET].astype(float)
    cols["abs_ret"] = ret.abs()
    cols["ret"] = ret
    cols["neg_ret"] = ret * (ret < 0).astype(float)
    cols["dow_next"] = pd.Series(
        ((panel.index.day_of_week + 1) % 7).astype(float), index=panel.index
    )
    for k in (1, 5, 22):
        cols[f"d_log_rv_{k}"] = lrv - lrv.shift(k)
    out = pd.DataFrame(cols, index=panel.index)
    if exog is not None:
        try:
            fn = EXOG_FEATURES[exog]
        except KeyError:
            raise ValueError(
                f"unknown exogenous feature set {exog!r}; registered: {sorted(EXOG_FEATURES)}"
            ) from None
        extra = fn(panel).reindex(panel.index).astype(float)
        extra.columns = [f"exog_{c}" for c in extra.columns]
        out = pd.concat([out, extra], axis=1)
    return out


class _GBMVol(BaseVolModel):
    """Shared log-target GBM machinery; subclasses supply the booster."""

    name = "gbm"
    required_columns = (COL_RV, COL_BPV, COL_JUMP, COL_RS_POS, COL_RS_NEG, COL_RQ, COL_RET)

    def __init__(
        self,
        n_estimators: int = 400,
        learning_rate: float = 0.03,
        min_data_in_leaf: int = 50,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        seed: int = 0,
        early_stopping_rounds: int | None = None,
        oof_frac: float = 0.2,
        val_frac: float = 0.15,
        exog: str | None = None,
        drop_incomplete_targets: bool = True,
    ) -> None:
        super().__init__(drop_incomplete_targets)
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if not 0.0 < oof_frac < 0.5 or not 0.0 < val_frac < 0.5:
            raise ValueError("oof_frac and val_frac must lie in (0, 0.5)")
        if exog is not None and exog not in EXOG_FEATURES:
            raise ValueError(f"unknown exogenous feature set {exog!r}")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_data_in_leaf = min_data_in_leaf
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.oof_frac = oof_frac
        self.val_frac = val_frac
        self.exog = exog
        self._booster: Any = None
        self._n_rounds = n_estimators
        self._sigma2 = float("nan")
        self._n_train = 0
        self._oof_n = 0
        self._feature_names: list[str] = []
        self._importance: dict[str, float] = {}

    # ------------------------------------------------------------ subclass interface
    @abstractmethod
    def _train(
        self,
        X: FloatArray,
        y: FloatArray,
        names: list[str],
        n_rounds: int,
        valid: tuple[FloatArray, FloatArray] | None = None,
    ) -> tuple[Any, int]:
        """Fit a booster on log targets ``y``; returns (booster, rounds actually kept)."""

    @abstractmethod
    def _predict_log(self, booster: Any, X: FloatArray, n_rounds: int) -> FloatArray:
        """Booster prediction on the log scale."""

    @abstractmethod
    def _gain_importance(self, booster: Any, names: list[str]) -> dict[str, float]:
        """Total-gain importance per feature name."""

    @abstractmethod
    def _model_params(self) -> dict[str, Any]:
        """Model-specific hyperparameters for `get_params`."""

    # --------------------------------------------------------------------- fitting
    def _fit(self, panel: pd.DataFrame, h: int, y: FloatArray, ok: BoolArray) -> None:
        self._booster = None
        feats = gbm_features(panel, self.exog)
        names = list(feats.columns)
        Xall = feats.to_numpy(dtype=float)
        core = np.isfinite(feats[["log_rv_d", "log_rv_m"]].to_numpy(dtype=float)).all(axis=1)
        use = ok & (y > 0.0) & core
        pos = np.flatnonzero(use)
        n = pos.size
        if n < MIN_TRAIN_ROWS:
            raise ModelInputError(f"{self.name}: only {n} usable training rows (need >= 120)")
        X, z = Xall[pos], np.log(y[pos])

        def split(frac: float) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
            """Time-ordered (train, holdout) split with an ``h``-day embargo between blocks."""
            cut = int(round(n * (1.0 - frac)))
            boundary = pos[cut]
            tr = pos[:cut] <= boundary - h  # targets of train rows are known by `boundary`
            return X[:cut][tr], z[:cut][tr], X[cut:], z[cut:]

        n_rounds = self.n_estimators
        if self.early_stopping_rounds is not None:
            Xa, za, Xv, zv = split(self.val_frac)
            _, best = self._train(Xa, za, names, n_rounds, valid=(Xv, zv))
            n_rounds = max(best, 20)
        # out-of-fold residuals on the last `oof_frac` of the training window
        Xa, za, Xo, zo = split(self.oof_frac)
        if zo.size >= MIN_OOF_ROWS and za.size >= MIN_TRAIN_ROWS // 2:
            booster_a, kept = self._train(Xa, za, names, n_rounds)
            resid = zo - self._predict_log(booster_a, Xo, kept)
            self._oof_n = int(zo.size)
        else:  # tiny sample: fall back to in-sample residuals
            booster_a, kept = self._train(X, z, names, n_rounds)
            resid = z - self._predict_log(booster_a, X, kept)
            self._oof_n = 0
        self._sigma2 = float(np.mean(resid**2))
        booster, kept = self._train(X, z, names, n_rounds)
        self._booster, self._n_rounds = booster, kept
        self._feature_names = names
        self._n_train = int(n)
        imp = self._gain_importance(booster, names)
        total = sum(imp.values())
        self._importance = {k: (v / total if total > 0 else 0.0) for k, v in imp.items()}
        self._fallback = float(np.exp(z.mean() + 0.5 * self._sigma2))

    # ------------------------------------------------------------------ prediction
    def _predict(self, panel: pd.DataFrame, h: int) -> float:
        feats = gbm_features(panel.iloc[-PREDICT_LOOKBACK:], self.exog)
        x = feats.to_numpy(dtype=float)[-1:]
        lp = float(self._predict_log(self._booster, x, self._n_rounds)[0])
        return math.exp(lp + 0.5 * self._sigma2)

    def get_params(self) -> dict[str, Any]:
        return {
            **self._base_params(),
            "n_estimators": self.n_estimators,
            "n_rounds_used": self._n_rounds,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "seed": self.seed,
            "early_stopping_rounds": self.early_stopping_rounds,
            "oof_frac": self.oof_frac,
            "exog": self.exog,
            **self._model_params(),
            "sigma2": self._sigma2 if math.isfinite(self._sigma2) else None,
            "sigma2_oof_rows": self._oof_n,
            "n_train": self._n_train,
            "features": list(self._feature_names),
            "feature_importance_gain": dict(self._importance),
        }


class LightGBMVol(_GBMVol):
    """LightGBM regression on log h-day RV (leaf-wise trees, 15 leaves by default)."""

    name = "lgbm"

    def __init__(self, num_leaves: int = 15, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_leaves = num_leaves

    def _params(self) -> dict[str, Any]:
        return {
            "objective": "regression",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": 1,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            "verbose": -1,
        }

    def _train(
        self,
        X: FloatArray,
        y: FloatArray,
        names: list[str],
        n_rounds: int,
        valid: tuple[FloatArray, FloatArray] | None = None,
    ) -> tuple[Any, int]:
        import lightgbm as lgb

        train = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=False)
        callbacks = []
        sets = []
        if valid is not None and self.early_stopping_rounds is not None:
            sets = [lgb.Dataset(valid[0], label=valid[1], reference=train)]
            callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
        booster = lgb.train(
            self._params(), train, num_boost_round=n_rounds, valid_sets=sets, callbacks=callbacks
        )
        best = int(booster.best_iteration) if valid is not None and booster.best_iteration else 0
        return booster, best if best > 0 else n_rounds

    def _predict_log(self, booster: Any, X: FloatArray, n_rounds: int) -> FloatArray:
        out: FloatArray = np.asarray(booster.predict(X, num_iteration=n_rounds), dtype=float)
        return out

    def _gain_importance(self, booster: Any, names: list[str]) -> dict[str, float]:
        gains = booster.feature_importance(importance_type="gain")
        return {n: float(g) for n, g in zip(names, gains, strict=True)}

    def _model_params(self) -> dict[str, Any]:
        return {"num_leaves": self.num_leaves}


class XGBoostVol(_GBMVol):
    """XGBoost regression on log h-day RV (depth-4 histogram trees by default)."""

    name = "xgb"

    def __init__(self, max_depth: int = 4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_depth = max_depth

    def _params(self) -> dict[str, Any]:
        return {
            "objective": "reg:squarederror",
            "eta": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": float(self.min_data_in_leaf),  # hessian = 1 per row
            "subsample": self.bagging_fraction,
            "colsample_bytree": self.feature_fraction,
            "tree_method": "hist",
            "seed": self.seed,
            "nthread": 1,
            "verbosity": 0,
        }

    def _train(
        self,
        X: FloatArray,
        y: FloatArray,
        names: list[str],
        n_rounds: int,
        valid: tuple[FloatArray, FloatArray] | None = None,
    ) -> tuple[Any, int]:
        import xgboost as xgb

        dtrain = xgb.DMatrix(X, label=y, feature_names=names)
        evals = []
        es = None
        if valid is not None and self.early_stopping_rounds is not None:
            evals = [(xgb.DMatrix(valid[0], label=valid[1], feature_names=names), "valid")]
            es = self.early_stopping_rounds
        booster = xgb.train(
            self._params(),
            dtrain,
            num_boost_round=n_rounds,
            evals=evals,
            early_stopping_rounds=es,
            verbose_eval=False,
        )
        kept = int(booster.best_iteration) + 1 if es is not None else n_rounds
        return booster, kept

    def _predict_log(self, booster: Any, X: FloatArray, n_rounds: int) -> FloatArray:
        import xgboost as xgb

        d = xgb.DMatrix(X, feature_names=list(booster.feature_names))
        out: FloatArray = np.asarray(booster.predict(d, iteration_range=(0, n_rounds)), dtype=float)
        return out

    def _gain_importance(self, booster: Any, names: list[str]) -> dict[str, float]:
        score = booster.get_score(importance_type="total_gain")
        return {n: float(score.get(n, 0.0)) for n in names}

    def _model_params(self) -> dict[str, Any]:
        return {"max_depth": self.max_depth}
