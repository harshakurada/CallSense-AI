"""Centralized configuration.

Two layers, deliberately kept separate:
  - Settings (this class): environment-specific / secret values, from .env.
  - get_config(): non-secret ML/audio parameters, from configs/config.yaml.

Nothing in src/, api/, or app/ should hard-code a path, model name, threshold,
or connection string — it should come from one of these two.
"""
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    app_env: str = "development"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    db_host: str = "localhost"
    db_port: int = 5432
    db_name: str = "callsense"
    db_user: str = "callsense"
    db_password: str = ""

    mlflow_tracking_uri: str = "./mlruns"
    model_cache_dir: str = str(BASE_DIR / "models")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_config() -> dict:
    config_path = BASE_DIR / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache
def get_audio_config() -> dict:
    config_path = BASE_DIR / "configs" / "audio.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
