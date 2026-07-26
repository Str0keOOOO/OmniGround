"""Local Transformers adapter for the Qwen/Qwen3.5-9B multimodal checkpoint."""

from __future__ import annotations

import logging
import re
import time
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Any

from ..base import BaseBackend, GenerationRequest
from ...core.config import AppConfig, ModelConfig
from ...core.errors import BackendInferenceError, BackendUnavailableError, ModelOutputParseError
from ...core.parsing import _extract_json_objects, parse_and_validate_last_valid_json
from ...core.contracts import GroundingResult
from ...core.validation import validate_grounding_result


_THINKING_BLOCK = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)
_LOG = logging.getLogger(__name__)

_QWEN35_OUTPUT_CONTRACT = """
Return exactly one JSON object and no Markdown or explanatory text. The object
must have this schema:
{"bboxes":[{"box_2d":[xmin,ymin,xmax,ymax],"label":"short_unique_label"}],"predicates":[]}
Use zero or more bboxes. Coordinates are integers normalized to 0..1000 and
must satisfy xmin < xmax and ymin < ymax. Every predicate argument must equal
one of the bbox labels. Use `[xmin, ymin, xmax, ymax]` specifically; the
server converts this Qwen-native order to OmniGround's public coordinate order.
""".strip()

_TASK_INSTRUCTION = re.compile(
    r'Perform two tasks on this image based on the task instruction:\s*"(?P<task>.*?)"',
    flags=re.IGNORECASE | re.DOTALL,
)
_HOLDING_TASK = re.compile(r"\b(?:pick\s+up|grab|hold|take)\b", flags=re.IGNORECASE)
_TARGET_AFTER_ACTION = re.compile(
    r"\b(?:pick\s+up|grab|hold|take)\s+(?P<target>.+?)(?:[.!?]|$)",
    flags=re.IGNORECASE,
)


def _task_targets(raw_text: str) -> list[str]:
    match = _TASK_INSTRUCTION.search(raw_text)
    task = match.group("task") if match else raw_text
    target_match = _TARGET_AFTER_ACTION.search(task)
    if not target_match:
        return []
    targets = []
    for part in re.split(r"\s*(?:,|and|&)\s*", target_match.group("target")):
        normalized = re.sub(r"^(?:the|a|an)\s+", "", part.strip().lower())
        targets.append(normalized.replace(" ", "_"))
    return targets


def parse_qwen35_grounding(raw_text: str, task_prompt: str = "") -> GroundingResult:
    """Convert Qwen3.5's native ``[xmin, ymin, xmax, ymax]`` boxes to OmniGround."""
    # Extract the outer payload before schema validation so we can repair the
    # common small-model error of repeating a label for multiple instances.
    try:
        payloads = _extract_json_objects(raw_text, allow_multiple_fences=True)
        payload = next(
            payload for payload in reversed(payloads)
            if "bboxes" in payload and "predicates" in payload
        )
    except (ModelOutputParseError, StopIteration):
        payload = parse_and_validate_last_valid_json(raw_text).model_dump(mode="json")

    labels_seen: dict[str, int] = {}
    original_to_unique: dict[str, list[str]] = {}
    for bbox in payload["bboxes"]:
        original_label = str(bbox["label"]).strip()
        occurrence = labels_seen.get(original_label, 0) + 1
        labels_seen[original_label] = occurrence
        unique_label = original_label if occurrence == 1 else f"{original_label}_{occurrence}"
        bbox["label"] = unique_label
        original_to_unique.setdefault(original_label, []).append(unique_label)
        xmin, ymin, xmax, ymax = bbox["box_2d"]
        bbox["box_2d"] = [ymin, xmin, ymax, xmax]

    instruction_source = task_prompt or raw_text
    if _HOLDING_TASK.search(instruction_source):
        targets = _task_targets(instruction_source)
        target_labels = [
            label
            for original, unique_labels in original_to_unique.items()
            if any(target in original.lower().replace(" ", "_") for target in targets)
            for label in unique_labels[:1]
        ]
        payload["predicates"] = [{"name": "holding", "args": [label]} for label in target_labels]
    else:
        for predicate in payload["predicates"]:
            predicate["args"] = [
                original_to_unique.get(argument, [argument])[0] for argument in predicate["args"]
            ]
    return validate_grounding_result(payload)


class Qwen35Backend(BaseBackend):
    """Run a configured Qwen3.5 checkpoint with its native chat template."""

    def __init__(self, config: ModelConfig, app_config: AppConfig) -> None:
        super().__init__()
        self._config = config
        self._app_config = app_config
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._input_device: Any | None = None
        self._generation_lock = Lock()

    @property
    def _checkpoint_path(self) -> Path | None:
        return self._app_config.resolve_path(self._config.checkpoint)

    def _preflight(self) -> None:
        checkpoint = self._checkpoint_path
        if checkpoint is None or not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
            raise BackendUnavailableError(
                f"Checkpoint is missing at {checkpoint}. "
                "Run `pixi run download-checkpoints -- "
                f"{self._config.option('model_id_hint', 'MODEL_ID')}`."
            )

    def check_ready(self) -> tuple[bool, str]:
        try:
            self._preflight()
        except BackendUnavailableError as exc:
            return False, exc.message
        return True, "Qwen3.5 checkpoint files are present; model loads during server startup"

    def load(self) -> None:
        self._preflight()
        checkpoint = self._checkpoint_path
        assert checkpoint is not None
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise BackendUnavailableError(
                "Qwen3.5 requires Transformers multimodal support. Run `pixi install` and retry."
            ) from exc

        device = self._config.device
        device_map = self._config.device_map
        if device == "cpu":
            device_map = None
        elif device not in {"auto", "cuda"}:
            device_map = device

        kwargs: dict[str, Any] = {"dtype": "auto"}
        if device_map:
            kwargs["device_map"] = device_map
        try:
            self._model = AutoModelForMultimodalLM.from_pretrained(str(checkpoint), **kwargs)
        except TypeError:  # compatibility with older Transformers releases
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            self._model = AutoModelForMultimodalLM.from_pretrained(str(checkpoint), **kwargs)
        self._processor = AutoProcessor.from_pretrained(str(checkpoint))
        if device == "cpu":
            self._model.to("cpu")
        self._model.eval()
        self._torch = torch
        self._input_device = next(self._model.parameters()).device

    @staticmethod
    def _render_prompt(prompt: str) -> str:
        """Keep TiPToP's task, while requiring OmniGround's public JSON schema."""
        return f"{prompt}\n\n{_QWEN35_OUTPUT_CONTRACT}"

    def _chat_inputs(self, request: GenerationRequest) -> Any:
        assert self._processor is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": request.image},
                    {"type": "text", "text": self._render_prompt(request.prompt)},
                ],
            }
        ]
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": False,
        }
        try:
            return self._processor.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            return self._processor.apply_chat_template(messages, **kwargs)

    def generate(self, request: GenerationRequest) -> GroundingResult:
        self.ensure_loaded()
        assert self._model is not None and self._processor is not None and self._torch is not None
        assert self._input_device is not None
        timing: dict[str, float] = {}
        backend_started_at = time.perf_counter()
        with self._generation_lock:
            try:
                phase_started_at = time.perf_counter()
                inputs = self._chat_inputs(request).to(self._input_device)
                timing["prepare_inputs"] = time.perf_counter() - phase_started_at
                temperature = request.temperature
                if temperature is None:
                    temperature = self._config.default_temperature
                generate_kwargs: dict[str, Any] = {
                    "max_new_tokens": self._config.max_new_tokens,
                    "do_sample": bool(temperature and temperature > 0),
                }
                if temperature is not None and temperature > 0:
                    generate_kwargs["temperature"] = temperature
                autocast = (
                    self._torch.autocast("cuda", dtype=self._torch.bfloat16)
                    if getattr(self._input_device, "type", None) == "cuda"
                    else nullcontext()
                )
                phase_started_at = time.perf_counter()
                with self._torch.inference_mode(), autocast:
                    output_ids = self._model.generate(**inputs, **generate_kwargs)
                if getattr(self._input_device, "type", None) == "cuda":
                    self._torch.cuda.synchronize(self._input_device)
                timing["model_generate"] = time.perf_counter() - phase_started_at
                phase_started_at = time.perf_counter()
                generated_ids = output_ids[:, inputs["input_ids"].size(1) :]
                raw_text = self._processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                timing["decode_output"] = time.perf_counter() - phase_started_at
            except Exception as exc:
                raise BackendInferenceError(f"Qwen3.5 local inference failed: {exc.__class__.__name__}") from exc

        self.last_raw_text = raw_text
        cleaned_text = _THINKING_BLOCK.sub("", raw_text).strip()
        try:
            phase_started_at = time.perf_counter()
            result = parse_qwen35_grounding(cleaned_text, request.prompt)
            timing["parse_and_validate"] = time.perf_counter() - phase_started_at
            timing["backend_total"] = time.perf_counter() - backend_started_at
            self.last_timing = timing
            return result
        except ModelOutputParseError:
            _LOG.warning(
                "Qwen3.5 raw model output (parse failed):\n---BEGIN RAW OUTPUT---\n%s\n---END RAW OUTPUT---",
                cleaned_text,
            )
            raise

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
        self._input_device = None
        super().unload()
