"""Validation helpers used at backend boundaries."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from .errors import ModelOutputValidationError
from .schemas import GroundingResult


def validate_grounding_result(value: GroundingResult | dict[str, Any]) -> GroundingResult:
    """Validate a backend result and expose a useful, non-silent error."""

    try:
        return value if isinstance(value, GroundingResult) else GroundingResult.model_validate(value)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in issue['loc'])}: {issue['msg']}" for issue in exc.errors()
        )
        raise ModelOutputValidationError(f"Model output violates the OmniGround contract: {details}") from exc
