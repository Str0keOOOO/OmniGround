"""Configuration-driven backend registry and instance cache."""

from __future__ import annotations

from threading import RLock

from .base import BaseBackend
from .local.molmo2 import Molmo2Backend
from .local.qwen35 import Qwen35Backend
from .local.robobrain import RoboBrainBackend
from .local.rynnbrain import RynnBrainBackend
from .remote.openai import OpenAIBackend
from ..core.config import AppConfig, ModelConfig
from ..core.errors import UnknownModelError


class ModelRegistry:
    """Creates a backend only for the requested model and caches it thereafter."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._backends: dict[str, BaseBackend] = {}
        self._lock = RLock()

    def get_config(self, model_id: str) -> ModelConfig:
        try:
            return self.config.models[model_id]
        except KeyError as exc:
            raise UnknownModelError(f"Unknown model_id '{model_id}'") from exc

    def _create_backend(self, model_id: str, model: ModelConfig) -> BaseBackend:
        if model.backend == "molmo2":
            return Molmo2Backend(model, self.config)
        if model.backend == "qwen35":
            return Qwen35Backend(model, self.config)
        if model.backend == "rynnbrain":
            return RynnBrainBackend(model, self.config)
        if model.backend == "robobrain":
            return RoboBrainBackend(model, self.config)
        if model.backend == "openai":
            return OpenAIBackend(model)
        raise UnknownModelError(f"Model '{model_id}' uses unsupported backend '{model.backend}'")

    def get_backend(self, model_id: str) -> tuple[BaseBackend, bool]:
        """Return a loaded backend and whether this call performed the first load."""

        model = self.get_config(model_id)
        with self._lock:
            backend = self._backends.get(model_id)
            if backend is None:
                backend = self._create_backend(model_id, model)
                self._backends[model_id] = backend
        first_load = backend.ensure_loaded()
        return backend, first_load

    def probe(self, model_id: str) -> tuple[bool, str]:
        model = self.get_config(model_id)
        with self._lock:
            backend = self._backends.get(model_id)
            if backend is None:
                backend = self._create_backend(model_id, model)
                self._backends[model_id] = backend
        return backend.check_ready()

    def describe_models(self) -> list[dict[str, object]]:
        descriptions: list[dict[str, object]] = []
        for model_id, model in self.config.models.items():
            backend = self._backends.get(model_id)
            descriptions.append(
                {
                    "id": model_id,
                    "backend": model.backend,
                    "mode": model.mode,
                    "loaded": backend.is_loaded if backend else False,
                }
            )
        return descriptions

    def unload_all(self) -> None:
        with self._lock:
            for backend in self._backends.values():
                backend.unload()
