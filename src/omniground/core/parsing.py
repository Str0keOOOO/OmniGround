"""Conservative parsing of VLM text into the public JSON contract."""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import ModelOutputParseError, PointOnlyOutputError
from .contracts import GroundingResult
from .validation import validate_grounding_result

_FENCED_JSON = re.compile(r"```(?:json)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA = re.compile(r",(?=\s*[}\]])")


def _remove_safe_trailing_commas(candidate: str) -> str:
    """Remove only commas directly before a closing JSON list/object delimiter."""

    return _TRAILING_COMMA.sub("", candidate)


def _extract_json_objects(raw_text: str, *, allow_multiple_fences: bool = False) -> list[dict[str, Any]]:
    if not raw_text or not raw_text.strip():
        raise ModelOutputParseError("Selected model returned an empty response")

    fences = _FENCED_JSON.findall(raw_text)
    if len(fences) > 1 and not allow_multiple_fences:
        raise ModelOutputParseError("Model output contains multiple Markdown JSON blocks")
    source = fences[0] if len(fences) == 1 else raw_text
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
    return objects


def _extract_single_json_object(raw_text: str) -> dict[str, Any]:
    objects = _extract_json_objects(raw_text)
    if len(objects) != 1:
        raise ModelOutputParseError("Model output contains multiple JSON objects; return exactly one")
    return objects[0]


def _validate_payload(payload: dict[str, Any]) -> GroundingResult:
    if "bboxes" not in payload and any("point" in str(key).lower() for key in payload):
        raise PointOnlyOutputError("Model returned point coordinates, but /generate requires bounding boxes")
    if "result" in payload or "text" in payload:
        raise ModelOutputParseError("Model output must directly contain bboxes and predicates, without an envelope")
    return validate_grounding_result(payload)


def parse_and_validate(raw_text: str) -> GroundingResult:
    """Parse one model response without guessing box order or coordinate scales."""

    return _validate_payload(_extract_single_json_object(raw_text))


def parse_and_validate_last_valid_json(raw_text: str) -> GroundingResult:
    """Return the final valid contract object from a verbose model response.

    Some chat models occasionally emit an incomplete JSON attempt before their
    final answer. This opt-in parser remains strict about the public schema,
    but lets that final complete object be used instead of rejecting the whole
    response merely because an earlier JSON fragment was present.
    """

    valid_results: list[GroundingResult] = []
    for payload in _extract_json_objects(raw_text, allow_multiple_fences=True):
        try:
            valid_results.append(_validate_payload(payload))
        except ModelOutputParseError:
            continue
    if valid_results:
        return valid_results[-1]
    return parse_and_validate(raw_text)
