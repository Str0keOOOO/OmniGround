"""Backend contract implemented by local and API adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from threading import RLock

from PIL import Image

from ..schemas import GroundingResult


@dataclass(frozen=True)
class GenerationRequest:
    image: Image.Image
    prompt: str
    model_id: str
    temperature: float | None = None


class BaseBackend(ABC):
    """A lazily loaded VLM adapter with no HTTP-server coupling."""

    def __init__(self) -> None:
        self._loaded = False
        self._load_lock = RLock()
        self.last_raw_text: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def ensure_loaded(self) -> bool:
        """Load exactly once per backend instance and report whether it was first load."""

        with self._load_lock:
            if self._loaded:
                return False
            self.load()
            self._loaded = True
            return True

    @abstractmethod
    def load(self) -> None:
        """Load model resources lazily; do not call this at server import time."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GroundingResult:
        """Return a validated GroundingResult."""

    def check_ready(self) -> tuple[bool, str]:
        """Perform a cheap non-loading readiness check."""

        return True, "loaded" if self.is_loaded else "configured; model will load on first request"

    def unload(self) -> None:
        """Release model/GPU resources when the adapter supports it."""

        self._loaded = False
