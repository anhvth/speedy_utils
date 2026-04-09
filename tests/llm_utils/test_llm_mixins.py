"""Unit tests for Qwen3LLM."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletionMessage
from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage

import llm_utils
from llm_utils import Qwen3LLM
from llm_utils.lm.llm import LLM
from llm_utils.lm.llm_qwen3 import (
    ASSISTANT_PREFIX,
    DEFAULT_CONTENT_MAX_TOKENS,
    DEFAULT_THINKING_MAX_TOKENS,
    THINK_END,
    THINK_START,
    _CustomPrefixCompletionState,
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
            finish_reason=finish_reason,  # type: ignore[arg-type]
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
            finish_reason=finish_reason,  # type: ignore[arg-type]
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
        completion_state = _CustomPrefixCompletionState(
            assistant_prompt_prefix=(
                f"{ASSISTANT_PREFIX}\n<think>\nseedreasoning step</think> final answer"
            ),
            generated_text="reasoning step</think> final answer",
            stop=THINK_END,
            stop_reason="stop",
            call_count=1,
            usage=CompletionUsage(
                completion_tokens=5,
                prompt_tokens=13,
                total_tokens=18,
            ),
        )

        with patch.object(
            llm,
            "complete_until",
            return_value=completion_state,
        ) as mock_complete_until:
            state = llm.complete_reasoning(
                "prompt",
                assistant_prompt_prefix="<think>\nseed",
            )

        self.assertEqual(state.reasoning, "seedreasoning step")
        self.assertEqual(state.content, "final answer")
        self.assertTrue(state.think_done)
        self.assertEqual(state.stop_reason, "stop")
        self.assertEqual(state.call_count, 1)
        self.assertIs(state.usage, completion_state.usage)
        self.assertEqual(
            mock_complete_until.call_args.kwargs["max_tokens"],
            DEFAULT_THINKING_MAX_TOKENS,
        )
        self.assertEqual(mock_complete_until.call_args.kwargs["stop"], THINK_END)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_reasoning_skips_generation_when_thinking_disabled(
        self, mock_get_client
    ):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM(enable_thinking=False)

        with patch.object(llm, "_generate_with_prefix_step") as mock_generate:
            state = llm.complete_reasoning(
                "prompt",
                assistant_prompt_prefix="<think>\nseed",
            )

        self.assertEqual(
            state.assistant_prompt_prefix,
            f"{ASSISTANT_PREFIX}\n{THINK_START}\n\n{THINK_END}",
        )
        self.assertEqual(state.reasoning, "")
        self.assertEqual(state.content, "")
        self.assertTrue(state.think_done)
        self.assertIsNone(state.stop_reason)
        self.assertEqual(state.call_count, 0)
        self.assertIsNone(state.usage)
        mock_generate.assert_not_called()

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_reasoning_auto_closes_incomplete_think_from_complete_until(
        self, mock_get_client
    ):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_state = _CustomPrefixCompletionState(
            assistant_prompt_prefix=f"{ASSISTANT_PREFIX}\n<think>\nseedstill thinking",
            generated_text="still thinking",
            stop=None,
            stop_reason="length",
            call_count=1,
            usage=CompletionUsage(
                completion_tokens=3,
                prompt_tokens=8,
                total_tokens=11,
            ),
        )

        with patch.object(llm, "complete_until", return_value=completion_state):
            state = llm.complete_reasoning(
                "prompt",
                assistant_prompt_prefix="<think>\nseed",
                thinking_max_tokens=32,
            )

        self.assertEqual(
            state.assistant_prompt_prefix,
            f"{ASSISTANT_PREFIX}\n<think>\nseedstill thinking\n</think>",
        )
        self.assertEqual(state.reasoning, "seedstill thinking")
        self.assertEqual(state.content, "")
        self.assertTrue(state.think_done)
        self.assertEqual(state.stop_reason, "length")
        self.assertEqual(state.call_count, 1)

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
    def test_chat_completion_skips_reasoning_when_thinking_disabled(
        self, mock_get_client
    ):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM(enable_thinking=False)
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
            side_effect=[content_choice],
        ) as mock_generate_with_prefix_step:
            result = llm.chat_completion(
                "prompt",
                assistant_prompt_prefix="<think>\nseed",
                thinking_max_tokens=32,
                content_max_tokens=64,
            )

        self.assertEqual(result.content, "final answer")
        self.assertFalse(hasattr(result, "reasoning_content"))
        self.assertEqual(result.call_count, 1)
        self.assertIs(result.usage, content_choice.usage)
        self.assertEqual(mock_generate_with_prefix_step.call_count, 1)
        self.assertEqual(
            mock_generate_with_prefix_step.call_args.args[1],
            f"{ASSISTANT_PREFIX}\n{THINK_START}\n\n{THINK_END}",
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_chat_completion_per_call_disable_overrides_constructor_default(
        self, mock_get_client
    ):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM(enable_thinking=True)
        content_choice = self._make_completion_choice("final answer")

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            side_effect=[content_choice],
        ) as mock_generate_with_prefix_step:
            result = llm.chat_completion(
                "prompt",
                enable_thinking=False,
            )

        self.assertEqual(result.content, "final answer")
        self.assertEqual(mock_generate_with_prefix_step.call_count, 1)
        self.assertEqual(
            mock_generate_with_prefix_step.call_args.args[1],
            f"{ASSISTANT_PREFIX}\n{THINK_START}\n\n{THINK_END}",
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_content_normalizes_string_prefix_when_thinking_disabled(
        self, mock_get_client
    ):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        content_choice = self._make_completion_choice("final answer")

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            side_effect=[content_choice],
        ) as mock_generate_with_prefix_step:
            result = llm.complete_content(
                "prompt",
                "<think>\nseed",
                enable_thinking=False,
            )

        self.assertEqual(result.content, "final answer")
        self.assertFalse(hasattr(result, "reasoning_content"))
        self.assertEqual(mock_generate_with_prefix_step.call_count, 1)
        self.assertEqual(
            mock_generate_with_prefix_step.call_args.args[1],
            f"{ASSISTANT_PREFIX}\n{THINK_START}\n\n{THINK_END}",
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
    def test_chat_completion_uses_constructor_token_budget_defaults(
        self, mock_get_client
    ):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM(thinking_max_tokens=12, content_max_tokens=34)
        reasoning_choice = self._make_completion_choice(
            "reasoning step</think>",
            finish_reason="stop",
        )
        content_choice = self._make_completion_choice(
            "final answer",
            finish_reason="stop",
        )

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            side_effect=[reasoning_choice, content_choice],
        ) as mock_generate_with_prefix_step:
            llm.chat_completion("prompt")

        self.assertEqual(
            [
                call.kwargs["max_tokens"]
                for call in mock_generate_with_prefix_step.call_args_list
            ],
            [12, 34],
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_until_uses_raw_prefix_and_appends_stop_token(
        self, mock_get_client
    ):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_choice = self._make_completion_choice(
            "memory text",
            finish_reason="stop",
            completion_tokens=4,
            prompt_tokens=9,
            total_tokens=13,
        )

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            return_value=completion_choice,
        ) as mock_generate_with_prefix_step:
            result = llm.complete_until(
                "prompt",
                "<memory>",
                stop="</memory>",
                max_tokens=32,
            )

        self.assertEqual(result.generated_text, "memory text</memory>")
        self.assertEqual(
            result.assistant_prompt_prefix,
            f"{ASSISTANT_PREFIX}\n<memory>memory text</memory>",
        )
        self.assertEqual(result.stop, "</memory>")
        self.assertEqual(result.stop_reason, "stop")
        self.assertEqual(result.call_count, 1)
        self.assertIs(result.usage, completion_choice.usage)
        self.assertIsNone(result.client_idx)
        self.assertEqual(
            mock_generate_with_prefix_step.call_args.args[1],
            f"{ASSISTANT_PREFIX}\n<memory>",
        )
        self.assertEqual(
            mock_generate_with_prefix_step.call_args.kwargs["stop"],
            ["</memory>"],
        )
        self.assertEqual(
            mock_generate_with_prefix_step.call_args.kwargs["prefix_mode"],
            "raw",
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_until_preserves_existing_assistant_prefix(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_choice = self._make_completion_choice(
            "\nfinal answer",
            finish_reason="stop",
            completion_tokens=4,
            prompt_tokens=9,
            total_tokens=13,
        )

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            return_value=completion_choice,
        ):
            result = llm.complete_until(
                "prompt",
                f"{ASSISTANT_PREFIX}\n<memory>m</memory>\n<think_efficient>",
                stop="</think_efficient>",
                max_tokens=32,
            )

        self.assertEqual(
            result.assistant_prompt_prefix,
            (
                f"{ASSISTANT_PREFIX}\n<memory>m</memory>\n"
                "<think_efficient>\nfinal answer</think_efficient>"
            ),
        )
        self.assertEqual(result.stop, "</think_efficient>")

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_until_uses_constructor_generation_defaults(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM(max_tokens=23, temperature=0.2)
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt-template"
        completion = SimpleNamespace(
            choices=[
                self._make_completion_choice(
                    "memory text",
                    finish_reason="stop",
                )
            ],
        )

        with (
            patch.object(llm, "_get_tokenizer", return_value=mock_tokenizer),
            patch.object(
                llm.client.completions,
                "create",
                return_value=completion,
            ) as mock_completion_create,
        ):
            llm.complete_until(
                "prompt",
                "<memory>",
                stop="</memory>",
            )

        self.assertEqual(mock_completion_create.call_args.kwargs["max_tokens"], 23)
        self.assertEqual(mock_completion_create.call_args.kwargs["temperature"], 0.2)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_until_state_inject_appends_prefix_text(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        state = _CustomPrefixCompletionState(
            assistant_prompt_prefix=f"{ASSISTANT_PREFIX}\n<memory>m</memory>",
            generated_text="m</memory>",
            stop="</memory>",
            stop_reason="stop",
            call_count=1,
            usage=None,
            client_idx=2,
        )

        injected = state.inject("\n<think_efficient>\n")

        self.assertEqual(
            injected.assistant_prompt_prefix,
            f"{ASSISTANT_PREFIX}\n<memory>m</memory>\n<think_efficient>\n",
        )
        self.assertEqual(injected.generated_text, "m</memory>\n<think_efficient>\n")
        self.assertEqual(injected.call_count, 1)
        self.assertEqual(injected.client_idx, 2)
        self.assertIsNone(injected.stop)
        self.assertIsNone(injected.stop_reason)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_until_reuses_bound_client_from_state(self, mock_get_client):
        client_a = self._make_mock_client()
        client_b = self._make_mock_client()
        mock_get_client.return_value = [client_a, client_b]
        llm = Qwen3LLM(client=["http://a/v1", "http://b/v1"])

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "prompt-template"
        client_b.completions.create.side_effect = [
            self._make_text_completion("memory", finish_reason="stop"),
            self._make_text_completion("thought", finish_reason="stop"),
        ]

        with (
            patch.object(llm, "_get_tokenizer", return_value=mock_tokenizer),
            patch.object(llm, "_select_client", return_value=client_b) as mock_select,
        ):
            state = llm.complete_until(
                "prompt",
                "<memory>",
                stop="</memory>",
                max_tokens=32,
            )
            state = state.inject("\n<think_efficient>\n")
            state = llm.complete_until(
                "prompt",
                state,
                stop="</think_efficient>",
                max_tokens=32,
            )

        self.assertEqual(mock_select.call_count, 1)
        client_a.completions.create.assert_not_called()
        self.assertEqual(client_b.completions.create.call_count, 2)
        self.assertEqual(state.client_idx, 1)
        self.assertEqual(state.call_count, 2)
        self.assertEqual(
            state.assistant_prompt_prefix,
            (
                f"{ASSISTANT_PREFIX}\n<memory>memory</memory>\n"
                "<think_efficient>\nthought</think_efficient>"
            ),
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_until_raises_for_missing_bound_client(self, mock_get_client):
        client_a = self._make_mock_client()
        mock_get_client.return_value = [client_a]
        llm = Qwen3LLM(client=["http://a/v1"])

        state = _CustomPrefixCompletionState(
            assistant_prompt_prefix=f"{ASSISTANT_PREFIX}\n<memory>m</memory>",
            generated_text="m</memory>",
            stop="</memory>",
            stop_reason="stop",
            call_count=1,
            usage=None,
            client_idx=5,
        )

        with self.assertRaisesRegex(RuntimeError, "out of range"):
            llm.complete_until(
                "prompt", state, stop="</think_efficient>", max_tokens=32
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

        with patch("llm_utils.show_chat", return_value=["shown"]):
            history = llm.inspect_history(idx=-1, k_last_messages=1)

        self.assertEqual(history, ["shown"])


class TestLLMRawCompletionStep(unittest.TestCase):
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
            finish_reason=finish_reason,  # type: ignore[arg-type]
            index=0,
            logprobs=None,
            text=content,
        )
        mock_choice.usage = usage
        return SimpleNamespace(choices=[mock_choice], usage=usage)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_raw_completion_step_returns_choice_with_usage(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = LLM()
        completion = self._make_text_completion("continued text")

        with patch.object(
            llm.client.completions,
            "create",
            return_value=completion,
        ) as mock_completion_create:
            result = llm._raw_completion_step(
                "seed prompt",
                max_tokens=9,
                temperature=0.3,
            )

        self.assertEqual(result.text, "continued text")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIs(result.usage, completion.usage)
        self.assertEqual(
            mock_completion_create.call_args.kwargs,
            {
                "model": "test-model",
                "prompt": "seed prompt",
                "max_tokens": 9,
                "temperature": 0.3,
            },
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_raw_completion_step_returns_bound_client_index(self, mock_get_client):
        client_a = self._make_mock_client()
        client_b = self._make_mock_client()
        mock_get_client.return_value = [client_a, client_b]
        llm = LLM(client=["http://a/v1", "http://b/v1"])
        completion = self._make_text_completion("continued text")

        with patch.object(
            client_b.completions,
            "create",
            return_value=completion,
        ):
            result = llm._raw_completion_step(
                "seed prompt",
                client_idx=1,
                return_client_idx=True,
            )

        choice, client_idx = result
        self.assertEqual(choice.text, "continued text")
        self.assertEqual(client_idx, 1)
        client_a.completions.create.assert_not_called()

    @patch("llm_utils.lm.llm.get_base_client")
    def test_raw_completion_step_defaults_max_tokens_to_one(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = LLM()
        completion = self._make_text_completion("x")

        with patch.object(
            llm.client.completions,
            "create",
            return_value=completion,
        ) as mock_completion_create:
            llm._raw_completion_step("seed prompt")

        self.assertEqual(mock_completion_create.call_args.kwargs["max_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
