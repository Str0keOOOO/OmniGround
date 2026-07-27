"""Local Transformers adapter for the Qwen/Qwen3.5-9B multimodal checkpoint."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Any

from ..base import BaseBackend, GenerationRequest
from ...core.config import AppConfig, ModelConfig
from ...core.errors import BackendInferenceError, BackendUnavailableError, ModelOutputParseError
from ...core.parsing import _extract_json_objects, parse_and_validate
from ...core.contracts import GroundingResult
from ...core.validation import validate_grounding_result


def parse_qwen35_grounding(raw_text: str, task_prompt: str = "") -> GroundingResult:
    """Parse a Qwen3.5 response using OmniGround's public contract.

    Qwen3.5 receives the same TiPToP prompt, but its native decoder emits box
    coordinates in ``[xmin, ymin, xmax, ymax]`` order. Convert those boxes to
    OmniGround's public ``[ymin, xmin, ymax, xmax]`` order. ``task_prompt`` is
    retained for backwards-compatible calls.
    """
    del task_prompt
    try:
        payloads = _extract_json_objects(raw_text, allow_multiple_fences=True)
        payload = next(
            payload for payload in reversed(payloads)
            if "bboxes" in payload and "predicates" in payload
        )
    except (ModelOutputParseError, StopIteration):
        return parse_and_validate(raw_text)

    labels_seen: dict[str, int] = {}
    original_to_unique: dict[str, list[str]] = {}
    for bbox in payload["bboxes"]:
        xmin, ymin, xmax, ymax = bbox["box_2d"]
        bbox["box_2d"] = [ymin, xmin, ymax, xmax]
        original_label = str(bbox["label"]).strip()
        occurrence = labels_seen.get(original_label, 0) + 1
        labels_seen[original_label] = occurrence
        unique_label = original_label if occurrence == 1 else f"{original_label}_{occurrence}"
        bbox["label"] = unique_label
        original_to_unique.setdefault(original_label, []).append(unique_label)

    # Keep predicate references valid after duplicate labels are disambiguated.
    for predicate in payload["predicates"]:
        predicate["args"] = [
            original_to_unique.get(argument, [argument])[0]
            for argument in predicate["args"]
        ]
    return validate_grounding_result(payload)


class Qwen35Backend(BaseBackend):
    """Run a configured Qwen3.5 checkpoint with its native chat template."""

    _GROUNDING_CONSTRAINTS = """

IMPORTANT OUTPUT CONSTRAINTS:
- The table surface is implicit and must never appear in bboxes or predicates.
- Never use "table_surface" as a predicate argument.
- Only generate predicates explicitly required by the task instruction; do not
  add on(object, table_surface) relations merely because objects rest on the table.
- Every predicate argument must exactly match, character-for-character, a label
  present in the bboxes array.
- Never invent aliases, synonyms, or labels that are absent from bboxes.
""".strip()

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
        """Use the same TiPToP prompt sent to API-backed models."""
        return f"{prompt.rstrip()}\n\n{Qwen35Backend._GROUNDING_CONSTRAINTS}"

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
        phase_started_at = time.perf_counter()
        result = parse_qwen35_grounding(raw_text, request.prompt)
        timing["parse_and_validate"] = time.perf_counter() - phase_started_at
        timing["backend_total"] = time.perf_counter() - backend_started_at
        self.last_timing = timing
        return result

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
        self._input_device = None
        super().unload()
