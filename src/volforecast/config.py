"""Runtime settings (paths, assets, horizons). Override with VOLF_* environment variables.

The platform contract also fixes a few un-prefixed variables that this module honors via
``AliasChoices``: ``MLFLOW_TRACKING_URI`` (tracking/registry server). ``VOLF_DATABASE_URL`` is the
forecast-store URL. Both default to sqlite files under ``artifacts_dir`` for local runs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VOLF_", env_file=".env", extra="ignore", populate_by_name=True
    )

    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    reports_dir: Path = PROJECT_ROOT / "reports"
    assets: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    horizons: tuple[int, ...] = (1, 5)
    interval: str = "5m"
    log_level: str = "INFO"
    log_json: bool = False

    # --- MLOps layer -----------------------------------------------------------------
    #: SQLAlchemy URL of the forecast store; empty -> sqlite:///<artifacts_dir>/volforecast.db
    database_url: str = ""
    #: MLflow tracking + registry URI; empty -> sqlite:///<artifacts_dir>/mlflow.db
    mlflow_tracking_uri: str = Field(
        default="",
        validation_alias=AliasChoices("MLFLOW_TRACKING_URI", "VOLF_MLFLOW_TRACKING_URI"),
    )
    mlflow_experiment: str = "volforecast"
    #: Pushgateway for batch-job metrics (VOLF_PUSHGATEWAY_URL); empty -> push is a no-op.
    pushgateway_url: str = ""
    #: Registry aliases (ADR-0006).
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"
    #: Pipeline/model settings file.
    production_config: Path = PROJECT_ROOT / "configs" / "production.yaml"

    @model_validator(mode="after")
    def _fill_local_defaults(self) -> Settings:
        if not self.database_url:
            self.database_url = f"sqlite:///{self.artifacts_dir / 'volforecast.db'}"
        if not self.mlflow_tracking_uri:
            self.mlflow_tracking_uri = f"sqlite:///{self.artifacts_dir / 'mlflow.db'}"
        return self

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def daily_dir(self) -> Path:
        return self.data_dir / "daily"

    @property
    def mlflow_artifact_root(self) -> Path:
        """Default artifact root for experiments created on a local (non-HTTP) tracking URI."""
        return self.artifacts_dir / "mlartifacts"

    @property
    def drift_reports_dir(self) -> Path:
        return self.artifacts_dir / "reports" / "drift"


def get_settings() -> Settings:
    return Settings()
