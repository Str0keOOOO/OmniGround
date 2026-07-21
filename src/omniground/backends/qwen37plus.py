"""Adapter for Qwen3.7-Plus via the OpenAI chat-completions API."""

from __future__ import annotations

from typing import Any

from .openai_backend import OpenAIBackend
from ..config import ModelConfig


class Qwen37PlusBackend(OpenAIBackend):
    """Qwen3.7-Plus backend reusing the OpenAI chat-completions API.

    This backend inherits the full OpenAI chat-completions protocol
    protocol from :class:`OpenAIBackend`, including image data URL
    encoding, ``extra_body`` passthrough (used for ``enable_thinking``),
    and environment-variable-based API key resolution.
    """

    def __init__(self, config: ModelConfig, client: Any | None = None) -> None:
        super().__init__(config, client)

    def check_ready(self) -> tuple[bool, str]:
        try:
            self.load()
        except Exception as exc:
            return False, str(exc)
        return True, "Qwen3.7-Plus API credentials and endpoint are configured"
