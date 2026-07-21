"""Lazy local Molmo2 adapter.

This adapter intentionally imports Torch and Transformers only when the
configured local model is first selected. Deployments with incompatible CUDA
or Transformers stacks should use the configured OpenAI-compatible API backend instead.
"""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Any

from .base import BaseBackend, GenerationRequest
from ..config import AppConfig, ModelConfig
from ..errors import BackendInferenceError, BackendUnavailableError
from ..parser import parse_and_validate
from ..schemas import GroundingResult

_LOG = logging.getLogger(__name__)


class Molmo2Backend(BaseBackend):
    """Run a local Hugging Face-compatible Molmo2 checkpoint, one request at a time."""

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
    def _source_path(self) -> Path | None:
        return self._app_config.resolve_path(self._config.source_path)

    @property
    def _checkpoint_path(self) -> Path | None:
        return self._app_config.resolve_path(self._config.checkpoint)

    def _preflight(self) -> None:
        source = self._source_path
        checkpoint = self._checkpoint_path
        if source is None or not source.is_dir():
            raise BackendUnavailableError(
                f"Molmo2 source is missing at {source}. Run git submodule update --init --recursive."
            )
        if checkpoint is None or not checkpoint.is_dir():
            raise BackendUnavailableError(
                f"Molmo2 checkpoint is missing at {checkpoint}. Run pixi run download-checkpoints -- molmo2-er."
            )

    def check_ready(self) -> tuple[bool, str]:
        try:
            self._preflight()
        except BackendUnavailableError as exc:
            return False, exc.message
        return True, "Molmo2 files are present; model remains unloaded until first request"

    def load(self) -> None:
        self._preflight()
        source = self._source_path
        checkpoint = self._checkpoint_path
        assert source is not None and checkpoint is not None
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise BackendUnavailableError(
                "Local Molmo2 requires optional dependencies. Install with `pip install -e .[molmo2]` "
                "in a compatible CUDA environment, or use openai-compatible."
            ) from exc

        device = self._config.device
        device_map = self._config.device_map
        if device == "cpu":
            device_map = None
        elif device not in {"auto", "cuda"}:
            device_map = device
        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device_map:
            kwargs["device_map"] = device_map
        try:
            self._model = AutoModelForImageTextToText.from_pretrained(str(checkpoint), dtype="auto", **kwargs)
        except TypeError:  # compatibility with older Transformers versions
            self._model = AutoModelForImageTextToText.from_pretrained(str(checkpoint), torch_dtype="auto", **kwargs)
        if device == "cpu":
            self._model.to("cpu")
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True, padding_side="left")
        self._torch = torch
        self._input_device = next(self._model.parameters()).device
        _LOG.info("Loaded Molmo2 checkpoint from %s on %s", checkpoint, self._input_device)

    def generate(self, request: GenerationRequest) -> GroundingResult:
        self.ensure_loaded()
        assert self._model is not None and self._processor is not None and self._torch is not None
        with self._generation_lock:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {"type": "image", "image": request.image},
                    ],
                }
            ]
            try:
                inputs = self._processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,
                )
                inputs = {
                    name: value.to(self._input_device) if hasattr(value, "to") else value for name, value in inputs.items()
                }
                temperature = request.temperature
                if temperature is None:
                    temperature = self._config.default_temperature
                generate_kwargs: dict[str, Any] = {"max_new_tokens": self._config.max_new_tokens, "do_sample": False}
                if temperature is not None and temperature > 0:
                    generate_kwargs.update({"do_sample": True, "temperature": temperature})
                autocast = (
                    self._torch.autocast("cuda", dtype=self._torch.bfloat16)
                    if getattr(self._input_device, "type", None) == "cuda"
                    else nullcontext()
                )
                with self._torch.inference_mode(), autocast:
                    output_ids = self._model.generate(**inputs, **generate_kwargs)
                generated_ids = output_ids[:, inputs["input_ids"].size(1) :]
                raw_text = self._processor.post_process_image_text_to_text(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            except Exception as exc:
                raise BackendInferenceError(f"Local Molmo2 inference failed: {exc.__class__.__name__}") from exc
        self.last_raw_text = raw_text
        _LOG.debug("Molmo2 returned %s characters", len(raw_text))
        return parse_and_validate(raw_text)

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._torch = None
        self._input_device = None
        super().unload()
