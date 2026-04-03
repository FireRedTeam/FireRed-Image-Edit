"""Unit tests for agent.llm_provider and agent.recaption with provider support."""

from __future__ import annotations

import os
import re
import types
from unittest import mock

import pytest

# ---------------------------------------------------------------------------
# Import helpers – the agent package may not be on sys.path by default
# ---------------------------------------------------------------------------
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agent.llm_provider import (
    GeminiProvider,
    LLMProvider,
    OpenAICompatibleProvider,
    get_recaption_provider,
)
from agent.recaption import (
    _replace_image_refs,
    build_reference_map,
    recaption,
)


# ===================================================================
# Tests for build_reference_map
# ===================================================================


class TestBuildReferenceMap:
    def test_single_group(self):
        assert build_reference_map([[0, 1, 2]]) == {1: 1, 2: 1, 3: 1}

    def test_two_groups(self):
        assert build_reference_map([[0, 1], [2, 3]]) == {
            1: 1, 2: 1, 3: 2, 4: 2,
        }

    def test_three_groups(self):
        result = build_reference_map([[0], [1, 2], [3, 4, 5]])
        assert result == {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3}

    def test_empty(self):
        assert build_reference_map([]) == {}


# ===================================================================
# Tests for _replace_image_refs
# ===================================================================


class TestReplaceImageRefs:
    def test_chinese_refs(self):
        ref_map = {1: 1, 2: 1, 3: 2}
        text = "将图2的猫放到图3的背景上"
        result = _replace_image_refs(text, ref_map)
        assert result == "将图1的猫放到图2的背景上"

    def test_english_refs(self):
        ref_map = {1: 1, 2: 1, 3: 2}
        text = "Place image 2 onto image 3"
        result = _replace_image_refs(text, ref_map)
        assert result == "Place image 1 onto image 2"

    def test_img_abbreviation(self):
        ref_map = {1: 1, 2: 2}
        text = "Merge img1 with IMG 2"
        result = _replace_image_refs(text, ref_map)
        assert result == "Merge img1 with IMG 2"

    def test_ordinal_chinese(self):
        ref_map = {1: 1, 2: 1, 3: 2}
        text = "将第3张图的背景换成蓝天"
        result = _replace_image_refs(text, ref_map)
        assert result == "将第2张图的背景换成蓝天"

    def test_no_mapping(self):
        ref_map = {1: 1}
        text = "图5的风格"
        result = _replace_image_refs(text, ref_map)
        # image 5 not in map → stays as-is
        assert result == "图5的风格"

    def test_empty_text(self):
        assert _replace_image_refs("", {1: 1}) == ""


# ===================================================================
# Tests for LLMProvider abstraction
# ===================================================================


class TestLLMProviderABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            LLMProvider()

    def test_subclass_must_implement_generate(self):
        class BadProvider(LLMProvider):
            pass

        with pytest.raises(TypeError):
            BadProvider()


# ===================================================================
# Tests for GeminiProvider
# ===================================================================


class TestGeminiProvider:
    def test_init_uses_config_defaults(self):
        with mock.patch("agent.config.GEMINI_API_KEY", "test-key"):
            provider = GeminiProvider()
            assert provider._api_key == "test-key"

    def test_init_with_explicit_params(self):
        provider = GeminiProvider(api_key="my-key", model_name="gemini-pro")
        assert provider._api_key == "my-key"
        assert provider._model_name == "gemini-pro"

    def test_generate_calls_gemini(self):
        provider = GeminiProvider(api_key="k", model_name="m")

        mock_response = mock.MagicMock()
        mock_response.text = "  rewritten instruction  "

        mock_model = mock.MagicMock()
        mock_model.generate_content.return_value = mock_response

        mock_genai = mock.MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model

        with mock.patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            result = provider.generate("sys", "user")
            assert result == "rewritten instruction"
            mock_model.generate_content.assert_called_once()


# ===================================================================
# Tests for OpenAICompatibleProvider
# ===================================================================


class TestOpenAICompatibleProvider:
    def test_defaults_to_minimax(self):
        with mock.patch.dict(os.environ, {"MINIMAX_API_KEY": "mm-key"}, clear=False):
            os.environ.pop("OPENAI_COMPATIBLE_API_KEY", None)
            os.environ.pop("OPENAI_COMPATIBLE_BASE_URL", None)
            os.environ.pop("OPENAI_COMPATIBLE_MODEL", None)
            provider = OpenAICompatibleProvider()
            assert provider._api_key == "mm-key"
            assert "minimax" in provider._base_url.lower()
            assert provider._model == "MiniMax-M2.7"

    def test_explicit_params(self):
        provider = OpenAICompatibleProvider(
            api_key="k", base_url="http://localhost", model="test-model"
        )
        assert provider._api_key == "k"
        assert provider._base_url == "http://localhost"
        assert provider._model == "test-model"

    def test_env_override(self):
        env = {
            "OPENAI_COMPATIBLE_API_KEY": "oc-key",
            "OPENAI_COMPATIBLE_BASE_URL": "https://custom.api/v1",
            "OPENAI_COMPATIBLE_MODEL": "custom-model",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            provider = OpenAICompatibleProvider()
            assert provider._api_key == "oc-key"
            assert provider._base_url == "https://custom.api/v1"
            assert provider._model == "custom-model"

    def test_temperature_clamping(self):
        provider = OpenAICompatibleProvider(api_key="k", base_url="http://x", model="m")

        mock_choice = mock.MagicMock()
        mock_choice.message.content = "output"

        mock_response = mock.MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        mock_openai = mock.MagicMock()
        mock_openai.return_value = mock_client

        with mock.patch("agent.llm_provider.OpenAICompatibleProvider._get_client", return_value=mock_client):
            # Temperature 0.0 should be clamped to 0.01
            provider.generate("sys", "user", temperature=0.0)
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == pytest.approx(0.01)

            # Temperature 1.5 should be clamped to 1.0
            provider.generate("sys", "user", temperature=1.5)
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == pytest.approx(1.0)

    def test_strips_think_tags(self):
        provider = OpenAICompatibleProvider(api_key="k", base_url="http://x", model="m")

        mock_choice = mock.MagicMock()
        mock_choice.message.content = "<think>reasoning here</think>actual output"

        mock_response = mock.MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with mock.patch("agent.llm_provider.OpenAICompatibleProvider._get_client", return_value=mock_client):
            result = provider.generate("sys", "user")
            assert result == "actual output"
            assert "<think>" not in result


# ===================================================================
# Tests for get_recaption_provider factory
# ===================================================================


class TestGetRecaptionProvider:
    def test_default_is_gemini(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RECAPTION_PROVIDER", None)
            provider = get_recaption_provider()
            assert isinstance(provider, GeminiProvider)

    def test_gemini_explicit(self):
        with mock.patch.dict(os.environ, {"RECAPTION_PROVIDER": "gemini"}):
            provider = get_recaption_provider()
            assert isinstance(provider, GeminiProvider)

    def test_openai_compatible(self):
        with mock.patch.dict(os.environ, {"RECAPTION_PROVIDER": "openai_compatible"}):
            provider = get_recaption_provider()
            assert isinstance(provider, OpenAICompatibleProvider)

    def test_minimax(self):
        with mock.patch.dict(os.environ, {"RECAPTION_PROVIDER": "minimax"}):
            provider = get_recaption_provider()
            assert isinstance(provider, OpenAICompatibleProvider)

    def test_case_insensitive(self):
        with mock.patch.dict(os.environ, {"RECAPTION_PROVIDER": "MiniMax"}):
            provider = get_recaption_provider()
            assert isinstance(provider, OpenAICompatibleProvider)

    def test_unknown_provider_raises(self):
        with mock.patch.dict(os.environ, {"RECAPTION_PROVIDER": "unknown"}):
            with pytest.raises(ValueError, match="Unknown RECAPTION_PROVIDER"):
                get_recaption_provider()


# ===================================================================
# Tests for recaption() with provider integration
# ===================================================================


class TestRecaptionWithProvider:
    def test_recaption_uses_provider(self):
        """recaption() should use the configured provider for generation."""
        mock_provider = mock.MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = "Rewritten instruction"

        with mock.patch(
            "agent.llm_provider.get_recaption_provider", return_value=mock_provider
        ):
            result = recaption("Edit image 1", [[0], [1]])
            assert result == "Rewritten instruction"
            mock_provider.generate.assert_called_once()

    def test_recaption_fallback_on_provider_error(self):
        """When provider init fails, recaption should fall back to regex-only."""
        with mock.patch(
            "agent.llm_provider.get_recaption_provider",
            side_effect=ImportError("no provider"),
        ):
            result = recaption("Edit 图2", [[0, 1]])
            # 图2 → 图1 (regex pass), no LLM expansion
            assert "图1" in result

    def test_recaption_fallback_on_generate_error(self):
        """When generate() fails all retries, fall back to regex-fixed text."""
        mock_provider = mock.MagicMock(spec=LLMProvider)
        mock_provider.generate.side_effect = RuntimeError("API down")

        with mock.patch(
            "agent.llm_provider.get_recaption_provider", return_value=mock_provider
        ), mock.patch("agent.recaption._RETRY_BACKOFF", 0):
            result = recaption("Edit 图3", [[0, 1, 2]])
            # Should fall back after 3 retries
            assert "图1" in result
            assert mock_provider.generate.call_count == 3

    def test_recaption_empty_response_uses_fallback(self):
        """When provider returns empty string, use the regex-fixed instruction."""
        mock_provider = mock.MagicMock(spec=LLMProvider)
        mock_provider.generate.return_value = ""

        with mock.patch(
            "agent.llm_provider.get_recaption_provider", return_value=mock_provider
        ):
            result = recaption("Edit 图2", [[0, 1]])
            assert "图1" in result


# ===================================================================
# Tests for OpenAICompatibleProvider message format
# ===================================================================


class TestOpenAICompatibleMessageFormat:
    def test_system_and_user_messages(self):
        provider = OpenAICompatibleProvider(api_key="k", base_url="http://x", model="m")

        mock_choice = mock.MagicMock()
        mock_choice.message.content = "output"
        mock_response = mock.MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = mock.MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with mock.patch("agent.llm_provider.OpenAICompatibleProvider._get_client", return_value=mock_client):
            provider.generate("system prompt", "user prompt")
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            messages = call_kwargs["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "system prompt"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "user prompt"


# ===================================================================
# Tests for config module
# ===================================================================


class TestConfig:
    def test_recaption_provider_default(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RECAPTION_PROVIDER", None)
            # Re-import to get fresh value
            import importlib
            import agent.config
            importlib.reload(agent.config)
            assert agent.config.RECAPTION_PROVIDER == "gemini"

    def test_recaption_provider_env(self):
        with mock.patch.dict(os.environ, {"RECAPTION_PROVIDER": "minimax"}):
            import importlib
            import agent.config
            importlib.reload(agent.config)
            assert agent.config.RECAPTION_PROVIDER == "minimax"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
