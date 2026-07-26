"""Configuration loading for models and their isolated backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .errors import InputValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "models.yaml"


class ModelConfig(BaseModel):
    """One configured backend. Extra backend-specific fields are preserved."""

    model_config = ConfigDict(extra="allow", frozen=True)

    backend: str
    mode: Literal["local", "api"]
    source_path: str | None = None
    checkpoint: str | None = None
    endpoint: str | None = None
    base_url: str | None = None
    model_name: str | None = None
    api_key_env: str | None = None
    device: str = "auto"
    device_map: str | None = "auto"
    max_new_tokens: int = Field(default=512, gt=0)
    default_temperature: float | None = Field(default=0.0, ge=0.0)
    timeout_seconds: float = Field(default=120.0, gt=0.0)

    def option(self, name: str, default: Any = None) -> Any:
        return self.model_extra.get(name, default) if self.model_extra else default


@dataclass(frozen=True)
class AppConfig:
    default_model: str
    models: dict[str, ModelConfig]
    path: Path

    @property
    def project_root(self) -> Path:
        return self.path.parent.parent

    def resolve_path(self, value: str | None) -> Path | None:
        if value is None:
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.project_root / candidate


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise InputValidationError(f"Model configuration file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        models_raw = raw.get("models", {})
        if not isinstance(models_raw, dict) or not models_raw:
            raise ValueError("models must be a non-empty mapping")
        models = {model_id: ModelConfig.model_validate(value) for model_id, value in models_raw.items()}
        default_model = str(raw["default_model"])
        if default_model not in models:
            raise ValueError(f"default_model '{default_model}' is not present in models")
        return AppConfig(default_model=default_model, models=models, path=config_path.resolve())
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise InputValidationError(f"Invalid model configuration {config_path}: {exc}") from exc
