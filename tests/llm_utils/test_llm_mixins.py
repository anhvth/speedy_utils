"""Unit tests for Qwen3LLM."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletionMessage
from openai.types.completion import CompletionChoice
from openai.types.completion_usage import CompletionUsage

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
        self.assertFalse(hasattr(Qwen3LLM, "generate_with_prefix"))

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
        usage = CompletionUsage(
            completion_tokens=7,
            prompt_tokens=11,
            total_tokens=18,
        )
        mock_choice = CompletionChoice(
            finish_reason=finish_reason,
            index=0,
            logprobs=None,
            text=content,
        )
        mock_choice.usage = usage
        return SimpleNamespace(choices=[mock_choice], usage=usage)

    @staticmethod
    def _make_completion_choice(
        text: str,
        finish_reason: str = "stop",
        *,
        completion_tokens: int = 7,
        prompt_tokens: int = 11,
        total_tokens: int = 18,
    ) -> CompletionChoice:
        choice = CompletionChoice(
            finish_reason=finish_reason,
            index=0,
            logprobs=None,
            text=text,
        )
        choice.usage = CompletionUsage(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )
        return choice

    @patch("llm_utils.lm.llm.get_base_client")
    def test_chat_completion_returns_reasoning_and_content(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_choice = self._make_completion_choice(
            "reasoning step</think> final answer",
            completion_tokens=5,
            prompt_tokens=13,
            total_tokens=18,
        )

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            side_effect=[completion_choice],
        ):
            result = llm.chat_completion(
                "prompt",
                assistant_prompt_prefix="<think>\nseed",
                thinking_max_tokens=32,
                content_max_tokens=64,
            )

        self.assertIsInstance(result, ChatCompletionMessage)
        self.assertEqual(result.role, "assistant")
        self.assertEqual(result.content, "final answer")
        self.assertEqual(result.reasoning_content, "seedreasoning step")
        self.assertEqual(result.call_count, 1)
        self.assertIs(result.usage, completion_choice.usage)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_chat_completion_records_structured_history(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt-template"
        usage = CompletionUsage(
            completion_tokens=4,
            prompt_tokens=9,
            total_tokens=13,
        )
        completion = SimpleNamespace(
            choices=[
                self._make_completion_choice(
                    "reasoning step</think> final answer",
                    completion_tokens=4,
                    prompt_tokens=9,
                    total_tokens=13,
                )
            ],
            usage=usage,
        )

        with (
            patch.object(llm, "_get_tokenizer", return_value=mock_tokenizer),
            patch.object(
                llm.client.completions,
                "create",
                return_value=completion,
            ),
        ):
            result = llm.chat_completion(
                "prompt",
                assistant_prompt_prefix="<think>\n",
                thinking_max_tokens=32,
                content_max_tokens=64,
            )

        self.assertEqual(result.content, "final answer")
        self.assertEqual(result.reasoning_content, "reasoning step")
        self.assertEqual(result.call_count, 1)
        self.assertIs(result.usage, usage)
        self.assertEqual(
            llm._last_conversations[-1],
            [
                {"role": "user", "content": "prompt"},
                {
                    "role": "assistant",
                    "content": "final answer",
                    "reasoning_content": "reasoning step",
                },
            ],
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_chat_completion_uses_default_think_prefix(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_choice = self._make_completion_choice(
            "reasoning step</think> final answer",
            completion_tokens=5,
            prompt_tokens=13,
            total_tokens=18,
        )

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            side_effect=[completion_choice],
        ) as mock_generate_with_prefix_step:
            result = llm.chat_completion(
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
        completion_choice = self._make_completion_choice(
            "reasoning step</think> final answer",
            completion_tokens=5,
            prompt_tokens=13,
            total_tokens=18,
        )

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            return_value=completion_choice,
        ) as mock_generate_with_prefix_step:
            state = llm.complete_reasoning(
                "prompt",
                assistant_prompt_prefix="<think>\nseed",
            )

        self.assertEqual(state.reasoning, "seedreasoning step")
        self.assertEqual(state.content, "final answer")
        self.assertTrue(state.think_done)
        self.assertEqual(state.stop_reason, "stop")
        self.assertEqual(state.call_count, 1)
        self.assertIs(state.usage, completion_choice.usage)
        self.assertEqual(
            mock_generate_with_prefix_step.call_args.kwargs["max_tokens"],
            DEFAULT_THINKING_MAX_TOKENS,
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_content_uses_reasoning_state(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        reasoning_choice = self._make_completion_choice(
            "still thinking",
            finish_reason="length",
            completion_tokens=3,
            prompt_tokens=8,
            total_tokens=11,
        )
        content_choice = self._make_completion_choice(
            "final answer",
            finish_reason="stop",
            completion_tokens=4,
            prompt_tokens=9,
            total_tokens=13,
        )

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            side_effect=[reasoning_choice, content_choice],
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
        self.assertEqual(reasoning_state.call_count, 1)
        self.assertIs(reasoning_state.usage, reasoning_choice.usage)
        self.assertEqual(result.content, "final answer")
        self.assertEqual(result.reasoning_content, "seedstill thinking")
        self.assertEqual(result.call_count, 2)
        self.assertIs(result.usage, content_choice.usage)
        self.assertEqual(result.usage.completion_tokens, 4)
        self.assertEqual(result.usage.prompt_tokens, 9)
        self.assertEqual(result.usage.total_tokens, 13)
        self.assertEqual(
            mock_generate_with_prefix_step.call_args_list[1].kwargs["max_tokens"],
            DEFAULT_CONTENT_MAX_TOKENS,
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_chat_completion_uses_default_token_budgets(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        reasoning_choice = self._make_completion_choice(
            "reasoning step</think>",
            finish_reason="stop",
            completion_tokens=3,
            prompt_tokens=8,
            total_tokens=11,
        )
        content_choice = self._make_completion_choice(
            "final answer",
            finish_reason="stop",
            completion_tokens=4,
            prompt_tokens=9,
            total_tokens=13,
        )

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            side_effect=[reasoning_choice, content_choice],
        ) as mock_generate_with_prefix_step:
            result = llm.chat_completion("prompt")

        self.assertEqual(result.content, "final answer")
        self.assertEqual(result.reasoning_content, "reasoning step")
        self.assertEqual(result.call_count, 2)
        self.assertIs(result.usage, content_choice.usage)
        self.assertEqual(result.usage.completion_tokens, 4)
        self.assertEqual(result.usage.prompt_tokens, 9)
        self.assertEqual(result.usage.total_tokens, 13)
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
