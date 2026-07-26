"""The exact public response schema shared by every backend."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator


class BoundingBox(BaseModel):
    """A normalized image box in TiPToP's required order."""

    model_config = ConfigDict(extra="forbid")

    box_2d: tuple[StrictInt, StrictInt, StrictInt, StrictInt]
    label: str

    @field_validator("box_2d")
    @classmethod
    def validate_box(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        ymin, xmin, ymax, xmax = value
        if not all(0 <= coordinate <= 1000 for coordinate in value):
            raise ValueError("box_2d coordinates must be integers in the inclusive range 0..1000")
        if ymin >= ymax:
            raise ValueError("box_2d must satisfy ymin < ymax")
        if xmin >= xmax:
            raise ValueError("box_2d must satisfy xmin < xmax")
        return value

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("label must be a non-empty string")
        return cleaned


class Predicate(BaseModel):
    """A relation over labels emitted in the same result."""

    model_config = ConfigDict(extra="forbid")

    name: str
    args: Annotated[list[str], Field(min_length=1)]

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("predicate name must be a non-empty string")
        return cleaned

    @field_validator("args")
    @classmethod
    def validate_args(cls, value: list[str]) -> list[str]:
        cleaned = [argument.strip() if isinstance(argument, str) else argument for argument in value]
        if any(not isinstance(argument, str) or not argument for argument in cleaned):
            raise ValueError("predicate args must contain only non-empty strings")
        return cleaned


class GroundingResult(BaseModel):
    """The response body. It deliberately has no enclosing result/text field."""

    model_config = ConfigDict(extra="forbid")

    bboxes: list[BoundingBox]
    predicates: list[Predicate]

    @model_validator(mode="after")
    def validate_cross_references(self) -> "GroundingResult":
        labels = [box.label for box in self.bboxes]
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        if duplicates:
            raise ValueError(f"bbox labels must be unique; duplicates: {', '.join(duplicates)}")

        available = set(labels)
        missing = sorted(
            {argument for predicate in self.predicates for argument in predicate.args if argument not in available}
        )
        if missing:
            raise ValueError(f"predicate args must reference bbox labels; unknown: {', '.join(missing)}")
        return self
