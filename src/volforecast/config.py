"""Runtime settings (paths, assets, horizons). Override with VOLF_* environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VOLF_", env_file=".env", extra="ignore")

    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    reports_dir: Path = PROJECT_ROOT / "reports"
    assets: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    horizons: tuple[int, ...] = (1, 5)
    interval: str = "5m"
    log_level: str = "INFO"
    log_json: bool = False

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def daily_dir(self) -> Path:
        return self.data_dir / "daily"


def get_settings() -> Settings:
    return Settings()
