"""OmniGround: a unified grounding contract for multiple VLM backends."""

from .core.contracts import BoundingBox, GroundingResult, Predicate

__all__ = ["BoundingBox", "GroundingResult", "Predicate"]
__version__ = "0.1.0"
