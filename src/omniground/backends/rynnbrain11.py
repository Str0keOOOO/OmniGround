"""RynnBrain 1.1 adapter for its native single-object grounding format."""

from __future__ import annotations

import re

from ..config import AppConfig, ModelConfig
from ..errors import ModelOutputParseError
from ..schemas import GroundingResult
from ..validation import validate_grounding_result
from .base import GenerationRequest
from .transformers_grounding import TransformersGroundingBackend


_TASK_INSTRUCTION = re.compile(
    r'Perform two tasks on this image based on the task instruction:\s*"(?P<task>.*?)"',
    flags=re.IGNORECASE | re.DOTALL,
)
_BOX = re.compile(
    r"<(?P<tag>[^<>]+)>\s*"
    r"\(\s*(?P<xmin>-?\d+)\s*,\s*(?P<ymin>-?\d+)\s*\)\s*,\s*"
    r"\(\s*(?P<xmax>-?\d+)\s*,\s*(?P<ymax>-?\d+)\s*\)\s*"
    r"</(?P=tag)>",
    flags=re.IGNORECASE,
)
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


def _label_for_task(task_instruction: str, model_tag: str) -> str:
    """Use the task target when RynnBrain returns its generic ``object`` tag."""
    label_source = model_tag.strip()
    if label_source.lower() in {"object", "target", "item"}:
        target_match = _TARGET_AFTER_ACTION.search(task_instruction)
        if target_match:
            label_source = _LEADING_ARTICLE.sub("", target_match.group("target").strip())
    label = re.sub(r"[^a-z0-9]+", "_", label_source.lower()).strip("_")
    return label or "object"


def parse_rynnbrain_grounding(raw_text: str, task_instruction: str) -> GroundingResult:
    """Convert RynnBrain's ``<object> (x1,y1), (x2,y2) </object>`` response."""
    matches = list(_BOX.finditer(raw_text))
    if len(matches) != 1:
        raise ModelOutputParseError(
            "RynnBrain must return exactly one native bounding box in the format "
            "<object> (x1, y1), (x2, y2) </object>"
        )

    match = matches[0]
    xmin = int(match.group("xmin"))
    ymin = int(match.group("ymin"))
    xmax = int(match.group("xmax"))
    ymax = int(match.group("ymax"))
    label = _label_for_task(task_instruction, match.group("tag"))
    predicates = [{"name": "holding", "args": [label]}] if _HOLDING_TASK.search(task_instruction) else []
    return validate_grounding_result(
        {
            "bboxes": [{"box_2d": [ymin, xmin, ymax, xmax], "label": label}],
            "predicates": predicates,
        }
    )


class RynnBrainBackend(TransformersGroundingBackend):
    """Use RynnBrain's documented grounding prompt and convert its output."""

    def __init__(self, config: ModelConfig, app_config: AppConfig) -> None:
        super().__init__(config, app_config, "rynnbrain11")

    @staticmethod
    def _render_prompt(prompt: str) -> str:
        task_instruction = _task_instruction(prompt)
        return (
            f"Locate the single object needed to complete this task: {task_instruction}\n"
            "Generate coordinates for one object bounding box. "
            "Constraints: x1,y1,x2,y2 in [0,1000]. "
            "Response must be exactly in this format: "
            "<object> (x1, y1), (x2, y2) </object>"
        )

    def _parse_output(self, raw_text: str, request: GenerationRequest) -> GroundingResult:
        return parse_rynnbrain_grounding(raw_text, _task_instruction(request.prompt))
