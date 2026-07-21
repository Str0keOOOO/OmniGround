"""Conservative parsing of VLM text into the public JSON contract."""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import ModelOutputParseError, PointOnlyOutputError
from .schemas import GroundingResult
from .validation import validate_grounding_result

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA = re.compile(r",(?=\s*[}\]])")


def _remove_safe_trailing_commas(candidate: str) -> str:
    """Remove only commas directly before a closing JSON list/object delimiter."""

    return _TRAILING_COMMA.sub("", candidate)


def _extract_single_json_object(raw_text: str) -> dict[str, Any]:
    if not raw_text or not raw_text.strip():
        raise ModelOutputParseError("Selected model returned an empty response")

    fences = _FENCED_JSON.findall(raw_text)
    if len(fences) > 1:
        raise ModelOutputParseError("Model output contains multiple Markdown JSON blocks")
    source = fences[0] if fences else raw_text
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    cursor = 0
    while True:
        start = source.find("{", cursor)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(_remove_safe_trailing_commas(source[start:]))
        except json.JSONDecodeError:
            cursor = start + 1
            continue
        if not isinstance(value, dict):
            cursor = start + end
            continue
        objects.append(value)
        cursor = start + end

    if not objects:
        raise ModelOutputParseError("Model output does not contain a valid JSON object")
    if len(objects) != 1:
        raise ModelOutputParseError("Model output contains multiple JSON objects; return exactly one")
    return objects[0]


def parse_and_validate(raw_text: str) -> GroundingResult:
    """Parse one model response without guessing box order or coordinate scales."""

    payload = _extract_single_json_object(raw_text)
    if "bboxes" not in payload and any("point" in str(key).lower() for key in payload):
        raise PointOnlyOutputError("Model returned point coordinates, but /generate requires bounding boxes")
    if "result" in payload or "text" in payload:
        raise ModelOutputParseError("Model output must directly contain bboxes and predicates, without an envelope")
    return validate_grounding_result(payload)
