"""Shared Transformers adapter for local embodied VLM checkpoints.

RynnBrain 1.1 and RoboBrain 2.5 both publish Hugging Face checkpoints.  They
need different model-side code, but their inference path is the same: a PIL
image and prompt are rendered with the checkpoint's chat template, generated
with Transformers, then validated against OmniGround's public schema.
"""

from __future__ import annotations

import logging
import importlib.util
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from .base import BaseBackend, GenerationRequest
from ..config import AppConfig, ModelConfig
from ..errors import BackendInferenceError, BackendUnavailableError
from ..parser import parse_and_validate
from ..schemas import GroundingResult

_LOG = logging.getLogger(__name__)
ModelFamily = Literal["rynnbrain11", "robobrain25"]

_OUTPUT_CONTRACT = """

Return exactly one JSON object and no Markdown or explanatory text. The object
must have this schema:
{"bboxes":[{"box_2d":[ymin,xmin,ymax,xmax],"label":"short_unique_label"}],"predicates":[]}
Use zero or more bboxes. Coordinates are integers normalized to 0..1000 and
must satisfy ymin < ymax and xmin < xmax. Every predicate argument must equal
one of the bbox labels. Preserve the requested task while obeying this output
contract.
""".strip()


class TransformersGroundingBackend(BaseBackend):
    """Run a configured RynnBrain 1.1 or RoboBrain 2.5 checkpoint locally."""

    def __init__(self, config: ModelConfig, app_config: AppConfig, family: ModelFamily) -> None:
        super().__init__()
        self._config = config
        self._app_config = app_config
        self._family = family
        self._model: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None
        self._input_device: Any | None = None
        self._robobrain_module: Any | None = None
        self._generation_lock = Lock()

    @property
    def _checkpoint_path(self) -> Path | None:
        return self._app_config.resolve_path(self._config.checkpoint)

    @property
    def _source_path(self) -> Path | None:
        return self._app_config.resolve_path(self._config.source_path)

    def _preflight(self) -> None:
        checkpoint = self._checkpoint_path
        if checkpoint is None or not checkpoint.is_dir() or not (checkpoint / "config.json").is_file():
            raise BackendUnavailableError(
                f"Checkpoint is missing at {checkpoint}. "
                f"Run `pixi run download-checkpoints -- {self._config.option('model_id_hint', 'MODEL_ID')}`."
            )
        if self._family == "robobrain25":
            source = self._source_path
            if source is None or not (source / "inference.py").is_file():
                raise BackendUnavailableError(
                    f"RoboBrain 2.5 source is missing at {source}. Run `pixi run setup`."
                )

    def check_ready(self) -> tuple[bool, str]:
        try:
            self._preflight()
        except BackendUnavailableError as exc:
            return False, exc.message
        return True, f"{self._family} files are present; model remains unloaded until first request"

    def load(self) -> None:
        self._preflight()
        checkpoint = self._checkpoint_path
        assert checkpoint is not None
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise BackendUnavailableError(
                "This local model requires optional dependencies. Install with "
                "`pip install -e .[embodied]` and retry."
            ) from exc

        device = self._config.device
        device_map = self._config.device_map
        if device == "cpu":
            device_map = None
        elif device not in {"auto", "cuda"}:
            device_map = device

        kwargs: dict[str, Any] = {}
        if device_map:
            kwargs["device_map"] = device_map
        if self._family == "rynnbrain11":
            # RynnBrain 1.1 publishes custom model code alongside each checkpoint.
            kwargs["trust_remote_code"] = True
        if self._family == "robobrain25":
            source = self._source_path
            assert source is not None
            module_path = source / "inference.py"
            spec = importlib.util.spec_from_file_location("omniground_robobrain25", module_path)
            if spec is None or spec.loader is None:  # pragma: no cover - guarded by _preflight
                raise BackendUnavailableError(f"Cannot load RoboBrain 2.5 source from {module_path}")
            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
                # This is the official loading implementation pinned in .gitmodules.
                official = module.UnifiedInference(str(checkpoint), device_map=device_map)
            except Exception as exc:
                raise BackendUnavailableError(
                    f"Unable to initialize official RoboBrain 2.5 inference code: {exc.__class__.__name__}"
                ) from exc
            self._robobrain_module = module
            self._model = official.model
            self._processor = official.processor
        else:
            try:
                self._model = AutoModelForImageTextToText.from_pretrained(str(checkpoint), dtype="auto", **kwargs)
            except TypeError:  # compatibility with pre-v5 Transformers
                self._model = AutoModelForImageTextToText.from_pretrained(
                    str(checkpoint), torch_dtype="auto", **kwargs
                )
            self._processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
        if device == "cpu":
            self._model.to("cpu")
        self._model.eval()
        self._torch = torch
        self._input_device = next(self._model.parameters()).device
        _LOG.info("Loaded %s checkpoint from %s on %s", self._family, checkpoint, self._input_device)

    @staticmethod
    def _render_prompt(prompt: str) -> str:
        return f"{prompt}\n\n{_OUTPUT_CONTRACT}"

    def _parse_output(self, raw_text: str, request: GenerationRequest) -> GroundingResult:
        """Parse the shared JSON response contract used by non-RynnBrain models."""
        return parse_and_validate(raw_text)

    def _chat_inputs(self, request: GenerationRequest) -> Any:
        assert self._processor is not None
        if self._family == "robobrain25":
            assert self._robobrain_module is not None
            temporary_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temporary_path = Path(temporary_image.name)
            temporary_image.close()
            try:
                request.image.save(temporary_path, format="PNG")
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": temporary_path.resolve().as_uri()},
                            {"type": "text", "text": self._render_prompt(request.prompt)},
                        ],
                    }
                ]
                text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = self._robobrain_module.process_vision_info(messages)
                return self._processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            finally:
                # The processor has materialized image tensors before this point.
                if temporary_path.exists():
                    os.unlink(temporary_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": request.image},
                    {"type": "text", "text": self._render_prompt(request.prompt)},
                ],
            }
        ]
        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        if self._family == "rynnbrain11":
            template_kwargs["enable_thinking"] = False
        try:
            return self._processor.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            # Older processor versions do not accept RynnBrain's thinking flag.
            template_kwargs.pop("enable_thinking", None)
            return self._processor.apply_chat_template(messages, **template_kwargs)

    def generate(self, request: GenerationRequest) -> GroundingResult:
        self.ensure_loaded()
        assert self._model is not None and self._processor is not None and self._torch is not None
        assert self._input_device is not None
        with self._generation_lock:
            try:
                inputs = self._chat_inputs(request).to(self._input_device)
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
                with self._torch.inference_mode(), autocast:
                    output_ids = self._model.generate(**inputs, **generate_kwargs)
                generated_ids = output_ids[:, inputs["input_ids"].size(1) :]
                raw_text = self._processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            except BackendInferenceError:
                raise
            except Exception as exc:
                raise BackendInferenceError(
                    f"{self._family} local inference failed: {exc.__class__.__name__}"
                ) from exc
        self.last_raw_text = raw_text
        _LOG.debug("%s returned %s characters", self._family, len(raw_text))
        return self._parse_output(raw_text, request)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
        self._input_device = None
        self._robobrain_module = None
        super().unload()
