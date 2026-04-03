"""Unit tests for Qwen3LLM."""

import unittest
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletionMessage

import llm_utils
from llm_utils import Qwen3LLM
from llm_utils.lm.llm_qwen3 import (
    DEFAULT_CONTENT_MAX_TOKENS,
    DEFAULT_THINKING_MAX_TOKENS,
)


class TestQwen3LLM(unittest.TestCase):
    """Verify prefix continuation flow for Qwen3LLM."""

    def test_qwen3_llm_class_name(self):
        self.assertEqual(Qwen3LLM.__name__, "Qwen3LLM")

    def test_top_level_exports_only_qwen3_llm(self):
        self.assertTrue(hasattr(llm_utils, "Qwen3LLM"))
        self.assertFalse(hasattr(llm_utils, "AsyncLM_Qwen3"))
        self.assertFalse(hasattr(llm_utils, "AsyncLM_DeepSeekR1"))

    def test_legacy_mixins_removed_from_lm_package(self):
        import llm_utils.lm as lm

        for name in (
            "TemperatureRangeMixin",
            "TwoStepPydanticMixin",
            "ModelUtilsMixin",
            "TokenizationMixin",
        ):
            self.assertFalse(hasattr(lm, name), name)

    @staticmethod
    def _make_mock_client():
        mock_client = MagicMock()
        mock_model = MagicMock(id="test-model")
        mock_client.models.list.return_value = MagicMock(data=[mock_model])
        return mock_client

    @staticmethod
    def _make_text_completion(content="hello", finish_reason="stop"):
        mock_choice = MagicMock()
        mock_choice.message.content = content
        mock_choice.finish_reason = finish_reason
        return MagicMock(choices=[mock_choice])

    @patch("llm_utils.lm.llm.get_base_client")
    def test_generate_with_prefix_returns_reasoning_and_content(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            side_effect=[
                ("reasoning step</think> final answer", "stop"),
            ],
        ):
            result = llm.generate_with_prefix(
                "prompt",
                assistant_prompt_prefix="<think>\nseed",
                thinking_max_tokens=32,
                content_max_tokens=64,
            )

        self.assertIsInstance(result, ChatCompletionMessage)
        self.assertEqual(result.role, "assistant")
        self.assertEqual(result.content, "final answer")
        self.assertEqual(result.reasoning_content, "seedreasoning step")

    @patch("llm_utils.lm.llm.get_base_client")
    def test_generate_with_prefix_uses_default_think_prefix(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            side_effect=[
                ("reasoning step</think> final answer", "stop"),
            ],
        ) as mock_generate_with_prefix_step:
            result = llm.generate_with_prefix(
                "prompt",
                thinking_max_tokens=32,
                content_max_tokens=64,
            )

        self.assertEqual(result.content, "final answer")
        self.assertEqual(
            mock_generate_with_prefix_step.call_args.args[1],
            "<|im_start|>assistant\n<think>\n",
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_reasoning_returns_prefix_state(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            return_value=("reasoning step</think> final answer", "stop"),
        ) as mock_generate_with_prefix_step:
            state = llm.complete_reasoning(
                "prompt",
                assistant_prompt_prefix="<think>\nseed",
            )

        self.assertEqual(state.reasoning, "seedreasoning step")
        self.assertEqual(state.content, "final answer")
        self.assertTrue(state.think_done)
        self.assertEqual(state.stop_reason, "stop")
        self.assertEqual(
            mock_generate_with_prefix_step.call_args.kwargs["max_tokens"],
            DEFAULT_THINKING_MAX_TOKENS,
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_content_uses_reasoning_state(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            side_effect=[
                ("still thinking", "length"),
                ("final answer", "stop"),
            ],
        ) as mock_generate_with_prefix_step:
            reasoning_state = llm.complete_reasoning(
                "prompt",
                assistant_prompt_prefix="<think>\nseed",
                thinking_max_tokens=32,
            )
            result = llm.complete_content("prompt", reasoning_state)

        self.assertEqual(reasoning_state.reasoning, "seedstill thinking")
        self.assertEqual(reasoning_state.content, "")
        self.assertTrue(reasoning_state.think_done)
        self.assertEqual(result.content, "final answer")
        self.assertEqual(result.reasoning_content, "seedstill thinking")
        self.assertEqual(
            mock_generate_with_prefix_step.call_args_list[1].kwargs["max_tokens"],
            DEFAULT_CONTENT_MAX_TOKENS,
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_generate_with_prefix_uses_default_token_budgets(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            side_effect=[
                ("reasoning step</think>", "stop"),
                ("final answer", "stop"),
            ],
        ) as mock_generate_with_prefix_step:
            result = llm.generate_with_prefix("prompt")

        self.assertEqual(result.content, "final answer")
        self.assertEqual(result.reasoning_content, "reasoning step")
        self.assertEqual(
            [
                call.kwargs["max_tokens"]
                for call in mock_generate_with_prefix_step.call_args_list
            ],
            [DEFAULT_THINKING_MAX_TOKENS, DEFAULT_CONTENT_MAX_TOKENS],
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_inspect_history_is_public(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        llm._last_conversations = [
            [
                {"role": "user", "content": "old prompt"},
                {"role": "assistant", "content": "old answer"},
            ],
            [
                {"role": "user", "content": "latest prompt"},
                {"role": "assistant", "content": "latest answer"},
            ],
        ]

        with patch("llm_utils.show_chat", return_value=["shown"]) as mock_show_chat:
            history = llm.inspect_history(idx=-1, k_last_messages=1)

        self.assertEqual(history, ["shown"])
        mock_show_chat.assert_called_once_with(
            [{"role": "assistant", "content": "latest answer"}]
        )


if __name__ == "__main__":
    unittest.main()
