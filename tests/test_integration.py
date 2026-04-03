"""Integration tests for LLM provider with real API calls.

These tests require valid API keys and network access. They are skipped
by default unless the appropriate environment variables are set:

- ``MINIMAX_API_KEY``: Required for MiniMax integration tests.
- ``GEMINI_API_KEY``: Required for Gemini integration tests.
- ``RUN_INTEGRATION_TESTS=1``: Must be set to enable integration tests.

Run with::

    RUN_INTEGRATION_TESTS=1 MINIMAX_API_KEY=... pytest tests/test_integration.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_SKIP_INTEGRATION = os.environ.get("RUN_INTEGRATION_TESTS") != "1"


@pytest.mark.skipif(_SKIP_INTEGRATION, reason="RUN_INTEGRATION_TESTS not set")
class TestMiniMaxIntegration:
    """Integration tests using real MiniMax API."""

    @pytest.fixture(autouse=True)
    def _require_key(self):
        if not os.environ.get("MINIMAX_API_KEY"):
            pytest.skip("MINIMAX_API_KEY not set")

    def test_minimax_provider_generate(self):
        from agent.llm_provider import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url="https://api.minimax.io/v1",
            model="MiniMax-M2.7",
        )
        result = provider.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Reply with exactly: HELLO",
            temperature=0.01,
            max_tokens=64,
        )
        assert len(result) > 0
        assert "HELLO" in result.upper()

    def test_minimax_recaption_end_to_end(self):
        from agent.recaption import recaption

        with (
            os.environ.__class__.__mro__[0].__init__
            and False
            or True
        ) and \
        pytest.MonkeyPatch().context() as mp:
            mp.setenv("RECAPTION_PROVIDER", "minimax")
            mp.setenv("MINIMAX_API_KEY", os.environ["MINIMAX_API_KEY"])
            result = recaption(
                "将图2的猫放到图1的背景上",
                [[0], [1]],
                target_length=128,
            )
            # Should return a non-empty rewritten instruction
            assert len(result) > 10
            # Original intent should be preserved
            assert any(c in result for c in ["猫", "cat", "背景", "background"])

    def test_minimax_temperature_boundaries(self):
        from agent.llm_provider import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url="https://api.minimax.io/v1",
            model="MiniMax-M2.7",
        )
        # Should not raise with edge temperature values
        result = provider.generate(
            system_prompt="Reply briefly.",
            user_prompt="Say OK",
            temperature=0.01,
            max_tokens=256,
        )
        assert len(result) > 0


@pytest.mark.skipif(_SKIP_INTEGRATION, reason="RUN_INTEGRATION_TESTS not set")
class TestGeminiIntegration:
    """Integration tests using real Gemini API."""

    @pytest.fixture(autouse=True)
    def _require_key(self):
        if not os.environ.get("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

    def test_gemini_provider_generate(self):
        from agent.llm_provider import GeminiProvider

        provider = GeminiProvider(
            api_key=os.environ["GEMINI_API_KEY"],
            model_name="gemini-2.5-flash",
        )
        result = provider.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Reply with exactly: HELLO",
            temperature=0.1,
            max_tokens=64,
        )
        assert len(result) > 0
        assert "HELLO" in result.upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
