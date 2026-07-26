"""RoboBrain 2.5 adapter for its native visual-grounding output."""

from __future__ import annotations

import re

from ...core.config import AppConfig, ModelConfig
from ...core.errors import ModelOutputParseError
from ...core.contracts import GroundingResult
from ...core.validation import validate_grounding_result
from ..base import GenerationRequest
from .transformers import TransformersGroundingBackend


_TASK_INSTRUCTION = re.compile(
    r'Perform two tasks on this image based on the task instruction:\s*"(?P<task>.*?)"',
    flags=re.IGNORECASE | re.DOTALL,
)
_BOX = re.compile(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]")
_HOLDING_TASK = re.compile(r"\b(?:pick\s+up|grab|hold|take)\b", flags=re.IGNORECASE)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an)\s+", flags=re.IGNORECASE)
_TARGET_AFTER_ACTION = re.compile(
    r"\b(?:pick\s+up|grab|hold|take)\s+(?P<target>.+?)(?:[.!?]|$)",
    flags=re.IGNORECASE,
)


def _task_instruction(prompt: str) -> str:
    """Extract the user task from TiPToP's long prompt when it is present."""
    match = _TASK_INSTRUCTION.search(prompt)
    return match.group("task").strip() if match else prompt.strip()


def _target_label(task_instruction: str) -> str:
    target_match = _TARGET_AFTER_ACTION.search(task_instruction)
    target = target_match.group("target").strip() if target_match else "object"
    target = _LEADING_ARTICLE.sub("", target)
    return re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_") or "object"


def parse_robobrain_grounding(raw_text: str, task_instruction: str) -> GroundingResult:
    """Convert RoboBrain's native ``[x1, y1, x2, y2]`` boxes to the API schema."""
    boxes: list[tuple[int, int, int, int]] = []
    for match in _BOX.finditer(raw_text):
        xmin, ymin, xmax, ymax = (int(value) for value in match.groups())
        box = (xmin, ymin, xmax, ymax)
        if box not in boxes:
            boxes.append(box)
    if not boxes:
        raise ModelOutputParseError(
            "RoboBrain must return at least one native bounding box in the format "
            "[x1, y1, x2, y2]"
        )

    target = _target_label(task_instruction)
    payload_boxes = [
        {
            "box_2d": [ymin, xmin, ymax, xmax],
            "label": target if index == 0 else f"{target}_{index + 1}",
        }
        for index, (xmin, ymin, xmax, ymax) in enumerate(boxes)
    ]
    predicates = [{"name": "holding", "args": [target]}] if _HOLDING_TASK.search(task_instruction) else []
    return validate_grounding_result({"bboxes": payload_boxes, "predicates": predicates})


class RoboBrainBackend(TransformersGroundingBackend):
    """Use RoboBrain's official grounding prompt and parse its box list."""

    def __init__(self, config: ModelConfig, app_config: AppConfig) -> None:
        super().__init__(config, app_config, "robobrain")

    @staticmethod
    def _render_prompt(prompt: str) -> str:
        task_instruction = _task_instruction(prompt)
        return (
            "Please provide the bounding box coordinate of the region this sentence describes: "
            f"{task_instruction}. Return only [x1, y1, x2, y2] with coordinates in 0..1000."
        )

    def _parse_output(self, raw_text: str, request: GenerationRequest) -> GroundingResult:
        return parse_robobrain_grounding(raw_text, _task_instruction(request.prompt))
