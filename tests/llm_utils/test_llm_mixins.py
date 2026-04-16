"""Unit tests for Qwen3LLM."""

import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai.types.chat import ChatCompletionMessage
from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage

import llm_utils
from llm_utils import Qwen3LLM
from llm_utils.lm.llm import LLM
from llm_utils.lm.openai_memoize import MOpenAI
from llm_utils.lm.llm_qwen3 import (
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

    def test_qwen3_llm_is_available_from_top_level_package(self):
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
        self.assertEqual(result.reasoning, "seedreasoning step")
        self.assertEqual(result.call_count, 1)
        self.assertIs(result.usage, completion_choice.usage)

    def test_call_result_model_dump_raises_with_mocked_mopenai_client(self):
        counts = {"models": 0, "completions": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "GET" and request.url.path.endswith("/models"):
                counts["models"] += 1
                return httpx.Response(
                    200,
                    json={
                        "object": "list",
                        "data": [
                            {
                                "id": "test-model",
                                "object": "model",
                                "created": 0,
                                "owned_by": "openai",
                            }
                        ],
                    },
                )

            if request.method == "POST" and request.url.path.endswith("/completions"):
                counts["completions"] += 1
                return httpx.Response(
                    200,
                    json={
                        "id": "cmpl-123",
                        "object": "text_completion",
                        "created": 1710000000,
                        "model": "test-model",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "index": 0,
                                "logprobs": None,
                                "text": "reasoning step</think> final answer",
                            }
                        ],
                        "usage": {
                            "completion_tokens": 7,
                            "prompt_tokens": 11,
                            "total_tokens": 18,
                        },
                    },
                )

            raise AssertionError(f"Unexpected request: {request.method} {request.url}")

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = MOpenAI(
            api_key="dummy-key",
            base_url="http://test/v1",
            http_client=http_client,
            cache=True,
        )

        try:
            with patch("llm_utils.lm.llm.get_base_client", return_value=client):
                llm = Qwen3LLM(model="test-model")
                result = llm(f"hi-{uuid.uuid4().hex}")
        finally:
            client.close()

        self.assertEqual(counts["models"], 1)
        self.assertEqual(counts["completions"], 1)
        # model_dump() must work — Qwen3LLM must not break pydantic serialization
        dumped = result.model_dump()
        self.assertIsInstance(dumped, dict)
        self.assertEqual(dumped["role"], "assistant")

    @patch("llm_utils.lm.llm.get_base_client")
    def test_qwen3_call_result_uses_canonical_reasoning_field(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()

        reasoning_state = Qwen3LLM._build_prefix_state(
            "<think>\nchain of thought</think>hello",
            stop_reason="stop",
            call_count=1,
        )
        with patch.object(llm, "complete_reasoning", return_value=reasoning_state):
            result = llm.chat_completion(
                "prompt",
                thinking_max_tokens=32,
                content_max_tokens=64,
            )

        dumped = result.model_dump()
        self.assertIn("reasoning", dumped)
        self.assertNotIn("reasoning_content", dumped)
        self.assertEqual(dumped["reasoning"], "chain of thought")

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
        self.assertEqual(result.reasoning, "reasoning step")
        self.assertEqual(result.call_count, 1)
        self.assertIs(result.usage, usage)
        self.assertEqual(
            llm._last_conversations[-1],
            [
                {"role": "user", "content": "prompt"},
                {
                    "role": "assistant",
                    "content": "final answer",
                    "reasoning": "reasoning step",
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
            "<think>\n",
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_reasoning_returns_prefix_state(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_state = _CustomPrefixCompletionState(
            assistant_prompt_prefix="<think>\nseedreasoning step</think> final answer",
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
            f"{THINK_START}\n\n{THINK_END}",
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
            assistant_prompt_prefix="<think>\nseedstill thinking",
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
            "<think>\nseedstill thinking\n</think>",
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
        self.assertEqual(result.reasoning, "seedstill thinking")
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
            f"{THINK_START}\n\n{THINK_END}",
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
            f"{THINK_START}\n\n{THINK_END}",
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
            f"{THINK_START}\n\n{THINK_END}",
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
        self.assertEqual(result.reasoning, "reasoning step")
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

        # Stop token </memory> is included in generated_text (not ASSISTANT_END)
        self.assertEqual(result.generated_text, "memory text</memory>")
        self.assertEqual(
            result.assistant_prompt_prefix,
            "<memory>memory text</memory>",
        )
        self.assertEqual(result.stop, "</memory>")
        self.assertEqual(result.stop_reason, "stop")
        self.assertEqual(result.call_count, 1)
        self.assertIs(result.usage, completion_choice.usage)
        self.assertIsNone(result.client_idx)
        # Assistant-body text (no wrapper) is passed to _generate_with_prefix_step
        self.assertEqual(
            mock_generate_with_prefix_step.call_args.args[1],
            "<memory>",
        )
        self.assertEqual(
            mock_generate_with_prefix_step.call_args.kwargs["stop"],
            ["</memory>"],
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
                "<memory>m</memory>\n<think_efficient>",
                stop="</think_efficient>",
                max_tokens=32,
            )

        self.assertEqual(result.generated_text, "\nfinal answer</think_efficient>")
        self.assertEqual(
            result.assistant_prompt_prefix,
            "<memory>m</memory>\n<think_efficient>\nfinal answer</think_efficient>",
        )
        self.assertEqual(result.stop, "</think_efficient>")

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_until_preserves_prefix_whitespace_verbatim(
        self, mock_get_client
    ):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_choice = self._make_completion_choice(
            "reasoning step",
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
                "<think>\n",
                stop="</think>",
                max_tokens=32,
            )

        self.assertEqual(mock_generate_with_prefix_step.call_args.args[1], "<think>\n")
        self.assertEqual(result.generated_text, "reasoning step</think>")
        self.assertEqual(
            result.assistant_prompt_prefix,
            "<think>\nreasoning step</think>",
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_until_strips_assistant_end_from_output(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_choice = self._make_completion_choice(
            "\n\n\nHello! 👋\n\nHow can I help you today?<|im_end|>",
            finish_reason="stop",
            completion_tokens=8,
            prompt_tokens=10,
            total_tokens=18,
        )

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            return_value=completion_choice,
        ):
            result = llm.complete_until(
                "prompt",
                "<memory>",
                stop="</memory>",
                max_tokens=32,
                include_stop_in_prefix=False,
            )

        self.assertEqual(
            result.generated_text,
            "\n\n\nHello! 👋\n\nHow can I help you today?",
        )
        self.assertEqual(
            result.assistant_prompt_prefix,
            "<memory>\n\n\nHello! 👋\n\nHow can I help you today?",
        )
        self.assertEqual(result.stop, "</memory>")

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_until_strips_assistant_end_from_input_state(
        self, mock_get_client
    ):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_choice = self._make_completion_choice(
            "continued",
            finish_reason="stop",
            completion_tokens=4,
            prompt_tokens=10,
            total_tokens=14,
        )

        with patch.object(
            llm,
            "_generate_with_prefix_step",
            return_value=completion_choice,
        ) as mock_generate_with_prefix_step:
            result = llm.complete_until(
                "prompt",
                "<memory>seed</memory><|im_end|>",
                stop="</memory>",
                max_tokens=32,
                include_stop_in_prefix=False,
            )

        self.assertEqual(
            mock_generate_with_prefix_step.call_args.args[1],
            "<memory>seed</memory>",
        )
        self.assertEqual(result.generated_text, "continued")
        self.assertEqual(
            result.assistant_prompt_prefix,
            "<memory>seed</memory>continued",
        )
        self.assertEqual(result.stop, "</memory>")

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
    def test_generate_with_prefix_step_falls_back_without_transformers(
        self, mock_get_client
    ):
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_choice = self._make_completion_choice(
            "memory text",
            finish_reason="stop",
        )

        with (
            patch.object(Qwen3LLM, "_tokenizer", None),
            patch.object(Qwen3LLM, "_tokenizer_checked", False),
            patch.object(Qwen3LLM, "_tokenizer_import_error", None),
            patch(
                "llm_utils.lm.llm_qwen3._get_tokenizer",
                side_effect=ImportError("transformers unavailable"),
            ) as mock_load_tokenizer,
            patch.object(
                llm,
                "_raw_completion_step",
                return_value=completion_choice,
            ) as mock_raw_completion,
        ):
            llm._generate_with_prefix_step(
                [{"role": "user", "content": "prompt"}],
                "<memory>",
                max_tokens=32,
            )
            llm._generate_with_prefix_step(
                [{"role": "user", "content": "prompt"}],
                "<memory>",
                max_tokens=32,
            )

        self.assertEqual(mock_load_tokenizer.call_count, 1)
        self.assertEqual(
            [call.args[0] for call in mock_raw_completion.call_args_list],
            [
                "<|im_start|>user\nprompt<|im_end|>\n<|im_start|>assistant\n<memory>",
                "<|im_start|>user\nprompt<|im_end|>\n<|im_start|>assistant\n<memory>",
            ],
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_until_state_inject_appends_prefix_text(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        state = _CustomPrefixCompletionState(
            assistant_prompt_prefix="<memory>m</memory>",
            generated_text="m</memory>",
            stop="</memory>",
            stop_reason="stop",
            call_count=1,
            usage=None,
            client_idx=2,
        )

        # inject strips only wrapper tokens and preserves the rest verbatim
        injected = state.inject("\n<think_efficient>\n")

        self.assertEqual(
            injected.assistant_prompt_prefix,
            "<memory>m</memory>\n<think_efficient>\n",
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
            "<memory>memory</memory>\n<think_efficient>\nthought</think_efficient>",
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_until_raises_for_missing_bound_client(self, mock_get_client):
        client_a = self._make_mock_client()
        mock_get_client.return_value = [client_a]
        llm = Qwen3LLM(client=["http://a/v1"])

        state = _CustomPrefixCompletionState(
            assistant_prompt_prefix="<memory>m</memory>",
            generated_text="m</memory>",
            stop="</memory>",
            stop_reason="stop",
            call_count=1,
            usage=None,
            client_idx=5,
        )

        with patch.object(llm, "_build_completion_prompt") as mock_build_prompt:
            with self.assertRaisesRegex(RuntimeError, "out of range"):
                llm.complete_until(
                    "prompt", state, stop="</think_efficient>", max_tokens=32
                )

        mock_build_prompt.assert_not_called()

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

    @patch("llm_utils.lm.llm.get_base_client")
    def test_chat_completion_takes_two_step_path_when_only_thinking_max_tokens_given(
        self, mock_get_client
    ):
        """When thinking_max_tokens is given but content_max_tokens is omitted,
        the two-step reasoning path must be taken and the given thinking_max_tokens honored."""
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        reasoning_state = Qwen3LLM._build_prefix_state(
            "<think>\nsome reasoning</think>",
            stop_reason="stop",
            call_count=1,
        )
        expected_message = ChatCompletionMessage(role="assistant", content="the answer")
        fallback_state = _CustomPrefixCompletionState(
            assistant_prompt_prefix="<think>\nfallback</think> done",
            generated_text="fallback</think> done",
            stop=None,
            stop_reason="stop",
            call_count=1,
            usage=None,
        )

        with (
            patch.object(
                llm, "complete_reasoning", return_value=reasoning_state
            ) as mock_reasoning,
            patch.object(llm, "complete_content", return_value=expected_message),
            # Guard the fallback single-step path so the test fails at the
            # assertion level, not inside the plumbing.
            patch.object(llm, "complete_until", return_value=fallback_state),
        ):
            result = llm.chat_completion("hi", thinking_max_tokens=2)

        mock_reasoning.assert_called_once()
        self.assertEqual(mock_reasoning.call_args.kwargs["thinking_max_tokens"], 2)
        self.assertIs(result, expected_message)


class TestEarlyThinkingStopMessage(unittest.TestCase):
    """Regression tests for the early_thinking_stop_message feature.

    The feature allows callers to customize the message appended when the
    thinking budget is exhausted before the model naturally closes the
    <think> block.  It is wired through chat_completion -> complete_reasoning.
    """

    @staticmethod
    def _make_mock_client():
        mock_client = MagicMock()
        mock_model = MagicMock(id="test-model")
        mock_client.models.list.return_value = MagicMock(data=[mock_model])
        return mock_client

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
    def test_complete_reasoning_injects_custom_message_on_length_stop(
        self, mock_get_client
    ):
        """When thinking_max_tokens is exhausted (length stop), a custom
        early_thinking_stop_message is prepended before the </think> token."""
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        # Model stops mid-think, returning partial reasoning with finish_reason="length"
        completion_state = _CustomPrefixCompletionState(
            assistant_prompt_prefix="<think>\npartial reasoning",
            generated_text="partial reasoning",
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
                assistant_prompt_prefix="<think>\n",
                thinking_max_tokens=32,
                early_thinking_stop_message="\n[OVERRIDE] Stopped early.",
            )

        self.assertTrue(state.think_done)
        reasoning = state.reasoning
        assert reasoning is not None
        # The injected message becomes part of the reasoning (the <think> body)
        # because split_assistant_parts parses everything between <think> and </think> as reasoning
        self.assertTrue(reasoning.endswith("[OVERRIDE] Stopped early."))
        self.assertEqual(state.content, "")
        self.assertEqual(state.call_count, 1)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_reasoning_uses_default_message_when_true(
        self, mock_get_client
    ):
        """early_thinking_stop_message=True uses DEFAULT_EARLY_THINKING_STOP_MESSAGE."""
        from llm_utils.lm.llm_qwen3 import DEFAULT_EARLY_THINKING_STOP_MESSAGE

        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_state = _CustomPrefixCompletionState(
            assistant_prompt_prefix="<think>\npartial reasoning",
            generated_text="partial reasoning",
            stop=None,
            stop_reason="length",
            call_count=1,
            usage=None,
        )

        with patch.object(llm, "complete_until", return_value=completion_state):
            state = llm.complete_reasoning(
                "prompt",
                assistant_prompt_prefix="<think>\n",
                thinking_max_tokens=32,
                early_thinking_stop_message=True,
            )

        self.assertTrue(state.think_done)
        reasoning = state.reasoning
        assert reasoning is not None
        # True (bool) is truthy → the stringified bool value "True" is prepended to reasoning
        self.assertTrue(reasoning.endswith("True"))
        # Verify the DEFAULT message is NOT used (True → string "True", not DEFAULT)
        self.assertNotIn(
            DEFAULT_EARLY_THINKING_STOP_MESSAGE.strip(),
            reasoning,
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_reasoning_uses_minimal_close_when_disabled(
        self, mock_get_client
    ):
        """early_thinking_stop_message=False suppresses the custom message."""
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_state = _CustomPrefixCompletionState(
            assistant_prompt_prefix="<think>\npartial reasoning",
            generated_text="partial reasoning",
            stop=None,
            stop_reason="length",
            call_count=1,
            usage=None,
        )

        with patch.object(llm, "complete_until", return_value=completion_state):
            state_false = llm.complete_reasoning(
                "prompt",
                assistant_prompt_prefix="<think>\n",
                thinking_max_tokens=32,
                early_thinking_stop_message=False,
            )
            state_none = llm.complete_reasoning(
                "prompt",
                assistant_prompt_prefix="<think>\n",
                thinking_max_tokens=32,
                early_thinking_stop_message=None,
            )

        # Both False and None are falsy → minimal closing text is used.
        # split_assistant_parts strips the [/FOUND_REASONING] token from
        # the content, leaving nothing (empty string) for content field.
        # The key invariant is think_done=True with no injected message.
        self.assertTrue(state_false.think_done)
        self.assertEqual(state_false.content, "")
        self.assertTrue(state_none.think_done)
        self.assertEqual(state_none.content, "")

    @patch("llm_utils.lm.llm.get_base_client")
    def test_complete_reasoning_natural_think_completion_ignores_early_stop_msg(
        self, mock_get_client
    ):
        """When the model naturally closes <think> with stop_reason="stop",
        early_thinking_stop_message is never used."""
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        completion_state = _CustomPrefixCompletionState(
            assistant_prompt_prefix="<think>\ncomplete reasoning</think>",
            generated_text="complete reasoning</think>",
            stop=THINK_END,
            stop_reason="stop",
            call_count=1,
            usage=None,
        )

        with patch.object(llm, "complete_until", return_value=completion_state):
            state = llm.complete_reasoning(
                "prompt",
                assistant_prompt_prefix="<think>\n",
                thinking_max_tokens=32,
                early_thinking_stop_message="SHOULD NOT APPEAR",
            )

        self.assertTrue(state.think_done)
        self.assertEqual(state.reasoning, "complete reasoning")
        self.assertNotIn("SHOULD NOT APPEAR", state.assistant_prompt_prefix)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_chat_completion_passes_early_thinking_stop_message_to_reasoning(
        self, mock_get_client
    ):
        """chat_completion forwards early_thinking_stop_message to complete_reasoning."""
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        # Model naturally closes thinking on the first call to complete_reasoning
        reasoning_state = Qwen3LLM._build_prefix_state(
            "<think>\nreasoning step</think>",
            stop_reason="stop",
            call_count=1,
        )
        content_choice = self._make_completion_choice("final answer")

        with (
            patch.object(
                llm, "complete_reasoning", return_value=reasoning_state
            ) as mock_reasoning,
            patch.object(
                llm, "complete_content", return_value=ChatCompletionMessage(
                    role="assistant", content="final answer"
                ),
            ) as mock_content,
        ):
            llm.chat_completion(
                "prompt",
                thinking_max_tokens=32,
                content_max_tokens=64,
                early_thinking_stop_message="[CUSTOM] Stopped early.",
            )

        mock_reasoning.assert_called_once()
        self.assertEqual(
            mock_reasoning.call_args.kwargs["early_thinking_stop_message"],
            "[CUSTOM] Stopped early.",
        )
        # complete_content should still be called since think was naturally done
        mock_content.assert_called_once()

    @patch("llm_utils.lm.llm.get_base_client")
    def test_chat_completion_triggers_content_step_after_early_stop(
        self, mock_get_client
    ):
        """When complete_reasoning hits early stop (length), chat_completion must
        proceed to complete_content for the answer."""
        mock_get_client.return_value = self._make_mock_client()
        llm = Qwen3LLM()
        # Reasoning hits length limit mid-think
        reasoning_state = Qwen3LLM._build_prefix_state(
            "<think>\npartial reasoning\n[/FOUND_REASONING]\n\n",
            stop_reason="length",
            call_count=1,
        )
        expected_message = ChatCompletionMessage(role="assistant", content="the answer")

        with (
            patch.object(llm, "complete_reasoning", return_value=reasoning_state),
            patch.object(llm, "complete_content", return_value=expected_message),
        ):
            result = llm.chat_completion(
                "prompt",
                thinking_max_tokens=32,
                content_max_tokens=64,
                early_thinking_stop_message="[STOP] Thinking budget exhausted.",
            )

        # complete_content must have been called because think was artificially closed
        self.assertEqual(result.content, "the answer")


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
