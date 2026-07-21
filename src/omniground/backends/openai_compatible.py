"""Adapter for OpenAI-compatible vision chat-completions services."""

from __future__ import annotations

import base64
import io
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from .base import BaseBackend, GenerationRequest
from ..config import ModelConfig
from ..errors import BackendInferenceError, BackendUnavailableError
from ..parser import parse_and_validate
from ..schemas import GroundingResult


class OpenAICompatibleBackend(BaseBackend):
    def __init__(self, config: ModelConfig, client: Any | None = None) -> None:
        super().__init__()
        self._config = config
        self._api_key: str | None = None
        self._client = client

    def load(self) -> None:
        parsed = urlparse(self._config.base_url or "")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BackendUnavailableError("openai_compatible backend requires a valid base_url")
        key_env = self._config.api_key_env or "OPENAI_API_KEY"
        self._api_key = os.environ.get(key_env)
        if not self._api_key:
            raise BackendUnavailableError(f"openai_compatible backend requires environment variable {key_env}")
        if not self._config.model_name:
            raise BackendUnavailableError("openai_compatible backend requires model_name")
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - declared core dependency
                raise BackendUnavailableError(
                    "openai_compatible backend requires the 'openai' package; run `pixi install` and retry"
                ) from exc
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._config.base_url,
                timeout=self._config.timeout_seconds,
            )

    def check_ready(self) -> tuple[bool, str]:
        try:
            self.load()
        except BackendUnavailableError as exc:
            return False, exc.message
        return True, "API backend credentials and endpoint are configured"

    @staticmethod
    def _data_url(image: Image.Image) -> str:
        output = io.BytesIO()
        image.save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "") for item in content
            )
        return "" if content is None else str(content)

    def generate(self, request: GenerationRequest) -> GroundingResult:
        self.ensure_loaded()
        assert self._client is not None

        temperature = request.temperature
        if temperature is None:
            temperature = self._config.default_temperature
        extra_body = self._config.option("extra_body", {})
        if not isinstance(extra_body, Mapping):
            raise BackendUnavailableError("openai_compatible extra_body must be a YAML mapping")
        request_options: dict[str, Any] = {
            "model": self._config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": request.prompt},
                        {"type": "image_url", "image_url": {"url": self._data_url(request.image)}},
                    ],
                }
            ],
            # OmniGround's HTTP contract is one complete JSON object, never a stream.
            "stream": False,
            "extra_body": dict(extra_body),
        }
        if temperature is not None:
            request_options["temperature"] = temperature
        try:
            completion = self._client.chat.completions.create(**request_options)
            choices = getattr(completion, "choices", [])
            raw_text = self._content_to_text(choices[0].message.content) if choices else ""
        except Exception as exc:
            raise BackendInferenceError(f"OpenAI-compatible backend failed: {exc.__class__.__name__}") from exc
        if not raw_text:
            raise BackendInferenceError("OpenAI-compatible backend returned no message content")
        self.last_raw_text = raw_text
        return parse_and_validate(raw_text)
