import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_utils import Qwen38LLM
from llm_utils.lm.llm_qwen38 import ReasoningEffort, DEFAULT_REASONING_EFFORT, DEFAULT_QWEN38_MODEL


class TestQwen38LLM(unittest.TestCase):
    """Model-specific behavior checks for Qwen38LLM."""

    def test_qwen38_default_reasoning_effort_is_none(self) -> None:
        params = inspect.signature(Qwen38LLM.__init__).parameters
        self.assertIsNone(params["reasoning_effort"].default)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_qwen38_get_model_sampling_params_defaults(
        self,
        mock_get_client,
    ) -> None:
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client

        llm = Qwen38LLM(temperature=0.7, top_p=0.9, max_tokens=64)
        params = llm.get_model_sampling_params()

        self.assertEqual(params["reasoning_effort"], DEFAULT_REASONING_EFFORT)
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["top_p"], 0.9)
        self.assertEqual(params["max_tokens"], 64)

        llm_no_reasoning = Qwen38LLM(enable_thinking=False, reasoning_effort=None)
        no_reasoning = llm_no_reasoning.get_model_sampling_params()
        self.assertEqual(no_reasoning["reasoning_effort"], "none")

    @staticmethod
    def _make_mock_client():
        mock_client = MagicMock()
        mock_model = MagicMock(id=DEFAULT_QWEN38_MODEL)
        mock_client.models.list.return_value = SimpleNamespace(data=[mock_model])
        return mock_client

    @staticmethod
    def _make_chat_completion_result(
        content: str = "hi",
        role: str = "assistant",
    ):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(role=role, content=content),
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_qwen38_default_reasoning_effort_is_sent_with_chat_completion(
        self,
        mock_get_client,
    ) -> None:
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        completion = self._make_chat_completion_result()

        with patch.object(mock_client.chat.completions, "create", return_value=completion) as create:
            llm = Qwen38LLM()
            llm.chat_completion("hello")

        self.assertEqual(create.call_count, 1)
        self.assertEqual(
            create.call_args.kwargs.get("reasoning_effort"),
            DEFAULT_REASONING_EFFORT,
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_qwen38_chat_completion_accepts_reasoning_effort_override(
        self,
        mock_get_client,
    ) -> None:
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        completion = self._make_chat_completion_result()
        runtime_effort: ReasoningEffort = "low"

        with patch.object(mock_client.chat.completions, "create", return_value=completion) as create:
            llm = Qwen38LLM(reasoning_effort="xhigh")
            llm.chat_completion("hello", reasoning_effort=runtime_effort)

        self.assertEqual(create.call_args.kwargs.get("reasoning_effort"), runtime_effort)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_qwen38_enable_thinking_false_forces_no_reasoning_when_not_set(
        self,
        mock_get_client,
    ) -> None:
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        completion = self._make_chat_completion_result()

        with patch.object(mock_client.chat.completions, "create", return_value=completion) as create:
            llm = Qwen38LLM(reasoning_effort=None)
            llm.chat_completion("hello", enable_thinking=False)

        self.assertEqual(create.call_args.kwargs.get("reasoning_effort"), "none")

    @patch("llm_utils.lm.llm.get_base_client")
    def test_qwen38_constructor_no_reasoning_when_enable_thinking_false(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        completion = self._make_chat_completion_result()

        with patch.object(mock_client.chat.completions, "create", return_value=completion) as create:
            llm = Qwen38LLM(enable_thinking=False, reasoning_effort=None)
            llm.chat_completion("hello")

        self.assertEqual(create.call_args.kwargs.get("reasoning_effort"), "none")
