"""Backend adapters. Optional model libraries are imported only by load()."""

from .base import BaseBackend, GenerationRequest

__all__ = ["BaseBackend", "GenerationRequest"]
