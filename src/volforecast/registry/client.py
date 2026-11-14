"""MLflow connection setup, registered-model naming and alias helpers (ADR-0006).

Registered models are named ``volforecast-{asset}-h{h}`` (asset lower-cased), and consumers load
``models:/<name>@champion``. Aliases, not stages.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from volforecast.logging import get_logger

log = get_logger(__name__)

CHAMPION = "champion"
CHALLENGER = "challenger"

# Our pyfunc declares explicit pip requirements; skip mlflow's uv.lock copy / auto-detection.
os.environ.setdefault("MLFLOW_LOG_UV_FILES", "false")
os.environ.setdefault("MLFLOW_UV_AUTO_DETECT", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def registered_model_name(asset: str, h: int) -> str:
    """``volforecast-btcusdt-h1`` for ("BTCUSDT", 1)."""
    return f"volforecast-{asset.lower()}-h{int(h)}"


def model_uri(asset: str, h: int, alias: str) -> str:
    return f"models:/{registered_model_name(asset, h)}@{alias}"


def _is_remote(uri: str) -> bool:
    return uri.startswith(("http://", "https://", "databricks"))


def configure_mlflow(
    tracking_uri: str,
    experiment: str = "volforecast",
    artifact_root: str | Path | None = None,
) -> MlflowClient:
    """Point MLflow (tracking + registry) at ``tracking_uri`` and select the experiment.

    For local (sqlite/file) backends the experiment is created with an explicit artifact
    location (``artifact_root`` or ``<db dir>/mlartifacts``) so nothing lands in ``./mlruns``.
    Remote servers decide their own artifact root (e.g. MinIO via ``--default-artifact-root``).
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    existing = client.get_experiment_by_name(experiment)
    if existing is None:
        location: str | None = None
        if not _is_remote(tracking_uri):
            if artifact_root is None:
                if tracking_uri.startswith("sqlite:///"):
                    artifact_root = Path(tracking_uri.removeprefix("sqlite:///")).parent / (
                        "mlartifacts"
                    )
                else:
                    artifact_root = Path("artifacts") / "mlartifacts"
            root = Path(artifact_root).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            location = root.as_uri()
        try:
            client.create_experiment(experiment, artifact_location=location)
        except MlflowException:  # lost a creation race with another process
            log.info("experiment_exists", name=experiment)
    mlflow.set_experiment(experiment)
    return client


def alias_version(client: MlflowClient, name: str, alias: str) -> ModelVersion | None:
    """The model version the alias points to, or None if the model/alias does not exist."""
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        if getattr(exc, "error_code", "") in ("RESOURCE_DOES_NOT_EXIST", "NOT_FOUND"):
            return None
        if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
            return None
        raise


def set_alias(client: MlflowClient, name: str, alias: str, version: str | int) -> None:
    """Point ``alias`` at ``version``: a single registry call, so the move is atomic."""
    client.set_registered_model_alias(name, alias, str(version))


def drop_alias(client: MlflowClient, name: str, alias: str) -> None:
    try:
        client.delete_registered_model_alias(name, alias)
    except MlflowException as exc:
        log.warning("alias_delete_failed", name=name, alias=alias, err=str(exc))


def tag_version(client: MlflowClient, name: str, version: str | int, tags: dict[str, Any]) -> None:
    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, str(value))
