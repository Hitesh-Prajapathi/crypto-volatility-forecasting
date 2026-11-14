"""MLflow pyfunc wrapper around a fitted ``VolModel``.

The model is cloudpickled into an artifact next to the pyfunc (``model.pkl``), together with the
walk-forward out-of-sample forecasts it was selected on (``oos.parquet``, used by the forecast
flow as the standardized-residual history for filtered-historical-simulation VaR/ES).

``predict`` takes the *daily panel* as a DataFrame with a ``date`` column (an MLflow signature
cannot carry a DataFrame index) plus the contract panel columns, and returns a one-row frame
with the h-day variance forecast for the last row.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import cloudpickle
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.pyfunc.model import PythonModel
from mlflow.types import ColSpec, Schema

from volforecast.contracts import DATE_INDEX, PANEL_COLUMNS, VolModel
from volforecast.registry.client import registered_model_name

#: input signature columns (the panel index becomes the ``date`` column)
INPUT_SCHEMA = Schema(
    [
        ColSpec("datetime", DATE_INDEX),
        ColSpec("double", "rv"),
        ColSpec("double", "bpv"),
        ColSpec("double", "jump"),
        ColSpec("double", "rs_pos"),
        ColSpec("double", "rs_neg"),
        ColSpec("double", "rq"),
        ColSpec("double", "ret"),
        ColSpec("long", "n_bars"),
        ColSpec("boolean", "complete"),
    ]
)
OUTPUT_SCHEMA = Schema([ColSpec("double", "forecast_var")])
SIGNATURE = ModelSignature(inputs=INPUT_SCHEMA, outputs=OUTPUT_SCHEMA)
INPUT_EXAMPLE_ROWS = 120
OOS_COLUMNS = ("origin_date", "forecast_var")


def panel_to_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily panel (DatetimeIndex) -> model-input frame with an explicit ``date`` column."""
    frame = panel[list(PANEL_COLUMNS)].copy()
    for c in frame.columns:
        if c == "complete":
            frame[c] = frame[c].astype(bool)
        elif c == "n_bars":
            frame[c] = frame[c].astype("int64")
        else:
            frame[c] = frame[c].astype("float64")
    frame.index = pd.DatetimeIndex(frame.index).as_unit("ns")
    frame.index.name = DATE_INDEX
    return frame.reset_index()


def frame_to_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Inverse of :func:`panel_to_frame` (also accepts a frame that already has the index)."""
    panel = frame.set_index(DATE_INDEX) if DATE_INDEX in frame.columns else frame.copy()
    panel.index = pd.DatetimeIndex(pd.to_datetime(panel.index)).normalize()
    panel.index.name = DATE_INDEX
    return panel.sort_index()


class VolForecastPyfunc(PythonModel):
    """Serve a fitted VolModel; ``metadata`` carries asset / h / model / train end / data hash."""

    def __init__(self, metadata: dict[str, Any] | None = None, model: VolModel | None = None):
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._model: VolModel | None = model
        self._oos: pd.DataFrame | None = None

    # -- MLflow hooks -----------------------------------------------------------------
    def load_context(self, context: Any) -> None:
        arts = context.artifacts
        with Path(arts["model"]).open("rb") as fh:
            self._model = cloudpickle.load(fh)
        if "oos" in arts:
            self._oos = pd.read_parquet(arts["oos"])

    def predict(self, context: Any, model_input: pd.DataFrame, params: Any = None) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("model not loaded; call load_context / pass model=")
        panel = frame_to_panel(model_input)
        h = int(self.metadata["horizon"])
        value = float(self._model.predict_variance(panel, h))
        return pd.DataFrame({"forecast_var": np.array([value], dtype="float64")})

    # -- convenience ------------------------------------------------------------------
    @property
    def model(self) -> VolModel:
        if self._model is None:
            raise RuntimeError("model not loaded")
        return self._model

    @property
    def oos_forecasts(self) -> pd.DataFrame:
        """Walk-forward OOS forecasts (``origin_date``, ``forecast_var``); may be empty."""
        if self._oos is None:
            return pd.DataFrame({c: [] for c in OOS_COLUMNS})
        return self._oos


def log_volforecast_model(
    model: VolModel,
    *,
    asset: str,
    h: int,
    model_name: str,
    panel: pd.DataFrame,
    data_hash: str,
    oos: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
    register: bool = True,
) -> Any:
    """Log ``model`` as a pyfunc in the *active* MLflow run and register it.

    Returns the MLflow ``ModelInfo``; the registered version is ``info.registered_model_version``.
    The input example is the tail of the real panel (models need a warm-up history).
    """
    metadata: dict[str, Any] = {
        "asset": asset,
        "horizon": int(h),
        "model_name": model_name,
        "train_end_date": str(panel.index[-1].date()),
        "train_start_date": str(panel.index[0].date()),
        "n_train_days": len(panel),
        "data_hash": data_hash,
        "params": params or {},
        **(extra_metadata or {}),
    }
    example = panel_to_frame(panel.tail(INPUT_EXAMPLE_ROWS))
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / "model.pkl").write_bytes(cloudpickle.dumps(model))
        (tmp_dir / "metadata.json").write_text(json.dumps(metadata, default=str, indent=2))
        artifacts = {
            "model": str(tmp_dir / "model.pkl"),
            "metadata": str(tmp_dir / "metadata.json"),
        }
        if oos is not None and not oos.empty:
            (oos[list(OOS_COLUMNS)]).to_parquet(tmp_dir / "oos.parquet", index=False)
            artifacts["oos"] = str(tmp_dir / "oos.parquet")
        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=VolForecastPyfunc(metadata=metadata),
            artifacts=artifacts,
            signature=SIGNATURE,
            input_example=example,
            registered_model_name=registered_model_name(asset, h) if register else None,
            pip_requirements=[
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
                f"cloudpickle=={cloudpickle.__version__}",
                "scipy",
                "statsmodels",
                "arch",
            ],
        )
    return info
