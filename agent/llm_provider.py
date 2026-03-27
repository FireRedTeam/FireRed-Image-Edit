"""LLM provider abstraction for text generation.

Supports multiple backends for the text-only recaption step:

- **gemini** (default): Uses ``google-generativeai`` SDK.
- **openai_compatible**: Any OpenAI-compatible API (MiniMax, OpenAI, etc.).

The ROI detection step (multimodal) always uses Gemini directly and is
not affected by this provider selection.

Provider selection is controlled by the ``RECAPTION_PROVIDER`` environment
variable (default ``"gemini"``).
"""

from __future__ import annotations

import os
import time
import traceback
import warnings
from abc import ABC, abstractmethod
from typing import Any


# ───────────────── abstract base ──────────────────


class LLMProvider(ABC):
    """Abstract interface for text generation."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> str:
        """Generate a text completion.

        Parameters
        ----------
        system_prompt:
            System-level instruction.
        user_prompt:
            User-level prompt / question.
        temperature:
            Sampling temperature.
        max_tokens:
            Maximum number of output tokens.

        Returns
        -------
        The generated text (stripped).
        """


# ───────────────── Gemini provider ──────────────────


class GeminiProvider(LLMProvider):
    """Text generation via Google Gemini (``google-generativeai`` SDK)."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        from agent.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

        self._api_key = api_key or GEMINI_API_KEY
        self._model_name = model_name or GEMINI_MODEL_NAME

    def _init_model(self, temperature: float, max_tokens: int) -> Any:
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "google-generativeai is required for the Gemini provider. "
                "Install it with:  pip install google-generativeai"
            ) from exc
        genai.configure(api_key=self._api_key)
        return genai.GenerativeModel(
            model_name=self._model_name,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "top_p": 0.95,
            },
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> str:
        model = self._init_model(temperature, max_tokens)
        response = model.generate_content(
            [{"role": "user", "parts": [system_prompt + "\n\n" + user_prompt]}]
        )
        return response.text.strip() if response.text else ""


# ───────────────── OpenAI-compatible provider ──────────────────


class OpenAICompatibleProvider(LLMProvider):
    """Text generation via any OpenAI-compatible API.

    Works with MiniMax, OpenAI, DeepSeek, and other providers that expose
    a ``/v1/chat/completions`` endpoint.

    Configuration is read from environment variables:

    - ``OPENAI_COMPATIBLE_API_KEY`` or ``MINIMAX_API_KEY``: API key.
    - ``OPENAI_COMPATIBLE_BASE_URL``: Base URL (default: MiniMax
      ``https://api.minimax.io/v1``).
    - ``OPENAI_COMPATIBLE_MODEL``: Model name (default: ``MiniMax-M2.7``).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get("OPENAI_COMPATIBLE_API_KEY")
            or os.environ.get("MINIMAX_API_KEY", "")
        )
        self._base_url = (
            base_url
            or os.environ.get(
                "OPENAI_COMPATIBLE_BASE_URL", "https://api.minimax.io/v1"
            )
        )
        self._model = (
            model
            or os.environ.get("OPENAI_COMPATIBLE_MODEL", "MiniMax-M2.7")
        )

    def _get_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for the OpenAI-compatible "
                "provider. Install it with:  pip install openai"
            ) from exc
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> str:
        client = self._get_client()
        # Clamp temperature to (0, 1] for providers like MiniMax
        temperature = max(0.01, min(temperature, 1.0))
        response = client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content or ""
        # Strip thinking tags that some models (e.g. MiniMax M2.5+) may emit
        import re

        text = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL
        ).strip()
        return text


# ───────────────── factory ──────────────────

_PROVIDERS: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "minimax": OpenAICompatibleProvider,
}


def get_recaption_provider() -> LLMProvider:
    """Return the configured recaption LLM provider.

    The provider is selected via the ``RECAPTION_PROVIDER`` environment
    variable.  Accepted values: ``"gemini"`` (default),
    ``"openai_compatible"``, ``"minimax"``.

    When ``"minimax"`` is selected, the OpenAI-compatible provider is used
    with MiniMax defaults (``https://api.minimax.io/v1``, model
    ``MiniMax-M2.7``).
    """
    provider_name = os.environ.get("RECAPTION_PROVIDER", "gemini").lower()
    provider_cls = _PROVIDERS.get(provider_name)
    if provider_cls is None:
        raise ValueError(
            f"Unknown RECAPTION_PROVIDER: {provider_name!r}. "
            f"Supported providers: {sorted(_PROVIDERS.keys())}"
        )
    return provider_cls()
