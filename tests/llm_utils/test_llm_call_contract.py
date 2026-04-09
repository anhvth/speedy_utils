"""Unit tests for the simplified sync LLM public API."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletionMessage
from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel

from llm_utils.lm.llm import LLM


class TestLLMCallContract(TestCase):
    """Verify the simplified sync LLM wrapper return shapes."""

    class ParsedOutput(BaseModel):
        answer: str

    @staticmethod
    def _make_mock_client():
        mock_client = MagicMock()
        mock_model = MagicMock(id="test-model")
        mock_client.models.list.return_value = MagicMock(data=[mock_model])
        return mock_client

    @staticmethod
    def _make_completion(message):
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice])

    @staticmethod
    def _make_text_completion(text, finish_reason="stop"):
        usage = CompletionUsage(
            completion_tokens=7,
            prompt_tokens=11,
            total_tokens=18,
        )
        choice = CompletionChoice(
            finish_reason=finish_reason,  # type: ignore[arg-type]
            index=0,
            logprobs=None,
            text=text,
        )
        return SimpleNamespace(choices=[choice], usage=usage)

    @staticmethod
    def _make_vllm_text_completion():
        usage = SimpleNamespace(
            completion_tokens=7,
            prompt_tokens=11,
            total_tokens=18,
        )
        choice = {
            "finish_reason": "stop",
            "index": 0,
            "text": "True",
            "logprobs": {
                "text_offset": [0, 4],
                "token_logprobs": [-0.11, -0.22],
                "tokens": ["True", "False"],
                "top_logprobs": [{"True": -0.11}, {"False": -0.22}],
                "extra_old_field": 123,
            },
            "prompt_logprobs": [{"True": -0.33}],
            "token_ids": [11, 22],
            "prompt_token_ids": [33, 44],
            "extra_choice_field": "vllm-extra",
        }
        return SimpleNamespace(choices=[choice], usage=usage)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_chat_completion_returns_first_message(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM()

        message = ChatCompletionMessage(role="assistant", content="hello")
        completion = self._make_completion(message)
        mock_client.chat.completions.create.return_value = completion

        result = llm.chat_completion("prompt")

        self.assertIs(result, message)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_chat_completion_returns_first_message_with_chat_messages(
        self, mock_get_client
    ):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM()

        message = ChatCompletionMessage(role="assistant", content="hello")
        completion = self._make_completion(message)
        mock_client.chat.completions.create.return_value = completion

        result = llm.chat_completion("prompt")

        self.assertIs(result, message)
        mock_client.chat.completions.create.assert_called_once()
        mock_client.completions.create.assert_not_called()

    @patch("llm_utils.lm.llm.get_base_client")
    def test_generate_returns_completion_choice_for_string_prompt(
        self, mock_get_client
    ):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM()

        completion = self._make_text_completion("hello")
        mock_client.completions.create.return_value = completion

        result = llm.generate("prompt")

        self.assertIsInstance(result, CompletionChoice)
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIs(result.usage, completion.usage)
        self.assertEqual(result.usage.total_tokens, 18)
        self.assertEqual(len(llm._last_conversations), 1)
        self.assertEqual(
            llm._last_conversations[0],
            [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "hello"},
            ],
        )
        self.assertEqual(
            mock_client.completions.create.call_args.kwargs["prompt"],
            "prompt",
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_generate_preserves_vllm_completion_metadata(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM(model="test-model")

        completion = self._make_vllm_text_completion()
        mock_client.completions.create.return_value = completion

        result = llm.generate("prompt")

        self.assertEqual(result.text, "True")
        self.assertEqual(result.finish_reason, "stop")
        assert result.logprobs is not None
        self.assertEqual(result.logprobs.token_logprobs, [-0.11, -0.22])
        self.assertEqual(result.logprobs.tokens, ["True", "False"])
        self.assertEqual(
            result.logprobs.top_logprobs,
            [{"True": -0.11}, {"False": -0.22}],
        )
        self.assertEqual(result.logprobs.extra_old_field, 123)  # type: ignore[attr-defined]
        self.assertEqual(result.prompt_logprobs, [{"True": -0.33}])  # type: ignore[attr-defined]
        self.assertEqual(result.token_ids, [11, 22])  # type: ignore[attr-defined]
        self.assertEqual(result.prompt_token_ids, [33, 44])  # type: ignore[attr-defined]
        self.assertEqual(result.extra_choice_field, "vllm-extra")  # type: ignore[attr-defined]

    @patch("llm_utils.lm.llm.get_base_client")
    def test_generate_uses_completions_api_not_chat_api(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM()

        completion = self._make_text_completion("hello")
        mock_client.completions.create.return_value = completion

        result = llm.generate("prompt", max_tokens=3)

        self.assertEqual(result.text, "hello")
        mock_client.completions.create.assert_called_once()
        mock_client.chat.completions.create.assert_not_called()

    @patch("llm_utils.lm.llm.get_base_client")
    def test_generate_smoke_with_vllm_logprobs_kwargs(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM(model="test-model")

        completion = self._make_vllm_text_completion()
        mock_client.completions.create.return_value = completion

        result = llm.generate(
            "prompt",
            max_tokens=0,
            echo=True,
            logprobs=1,
            temperature=0,
        )

        self.assertEqual(
            mock_client.completions.create.call_args.kwargs,
            {
                "model": "test-model",
                "prompt": "prompt",
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": 0,
            },
        )
        self.assertEqual(result.logprobs.token_logprobs, [-0.11, -0.22])  # type: ignore[union-attr]
        self.assertEqual(result.prompt_logprobs, [{"True": -0.33}])  # type: ignore[attr-defined]
        self.assertEqual(result.token_ids, [11, 22])  # type: ignore[attr-defined]
        self.assertEqual(result.prompt_token_ids, [33, 44])  # type: ignore[attr-defined]

    @patch("llm_utils.lm.llm.get_base_client")
    def test_constructor_common_kwargs_become_default_model_kwargs(
        self, mock_get_client
    ):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM(
            model="test-model",
            max_tokens=17,
            temperature=0.3,
            top_p=0.8,
        )

        completion = self._make_text_completion("hello")
        mock_client.completions.create.return_value = completion

        llm.generate("prompt")

        self.assertEqual(
            mock_client.completions.create.call_args.kwargs,
            {
                "model": "test-model",
                "prompt": "prompt",
                "max_tokens": 17,
                "temperature": 0.3,
                "top_p": 0.8,
            },
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_generate_requires_string_prompt(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = LLM()

        with self.assertRaises(TypeError):
            llm.generate({"prompt": "hello"})  # type: ignore[arg-type]

    @patch("llm_utils.lm.llm.get_base_client")
    def test_pydantic_parse_returns_model_instance(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM()

        parsed_message = SimpleNamespace(
            content='{"answer": "yes"}',
            parsed={"answer": "yes"},
        )
        completion = self._make_completion(parsed_message)
        mock_client.chat.completions.parse.return_value = completion

        result = llm.pydantic_parse(
            "prompt",
            response_model=self.ParsedOutput,
        )

        self.assertIsInstance(result, self.ParsedOutput)
        self.assertEqual(result.answer, "yes")

    @patch("llm_utils.lm.llm.get_base_client")
    def test_pydantic_parse_accepts_json_string_in_parsed_message(
        self, mock_get_client
    ):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM()

        parsed_message = SimpleNamespace(
            content='{"answer": "green"}',
            parsed='{"answer": "green"}',
        )
        completion = self._make_completion(parsed_message)
        mock_client.chat.completions.parse.return_value = completion

        result = llm.pydantic_parse(
            "prompt",
            response_model=self.ParsedOutput,
        )

        self.assertIsInstance(result, self.ParsedOutput)
        self.assertEqual(result.answer, "green")

    @patch("llm_utils.lm.llm.get_base_client")
    def test_call_can_return_dict_for_chat_completion(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM()

        message = ChatCompletionMessage(role="assistant", content="hello")
        completion = self._make_completion(message)
        mock_client.chat.completions.create.return_value = completion

        result = llm("prompt", return_dict=True)

        self.assertEqual(
            set(result.keys()),
            {"completion", "message", "messages", "parsed"},
        )
        self.assertIs(result["completion"], completion)
        self.assertIs(result["message"], message)
        self.assertEqual(result["parsed"], "hello")
        self.assertEqual(
            result["messages"],
            [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": "hello"},
            ],
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_call_can_return_dict_for_parsed_output(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM()

        parsed_message = SimpleNamespace(
            content='{"answer": "yes"}',
            parsed={"answer": "yes"},
        )
        completion = self._make_completion(parsed_message)
        mock_client.chat.completions.parse.return_value = completion

        result = llm(
            "prompt",
            response_model=self.ParsedOutput,
            return_dict=True,
        )

        self.assertEqual(
            set(result.keys()),
            {"completion", "message", "messages", "parsed"},
        )
        self.assertIs(result["completion"], completion)
        self.assertIs(result["message"], parsed_message)
        self.assertIsInstance(result["parsed"], self.ParsedOutput)
        self.assertEqual(result["parsed"].answer, "yes")
        self.assertEqual(
            result["messages"],
            [
                {"role": "user", "content": "prompt"},
                {"role": "assistant", "content": '{"answer": "yes"}'},
            ],
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_pydantic_parse_accepts_message_list(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = LLM()

        parsed_message = SimpleNamespace(
            content='{"answer": "yes"}',
            parsed={"answer": "yes"},
        )
        completion = self._make_completion(parsed_message)
        llm.client.chat.completions.parse.return_value = completion

        result = llm.pydantic_parse(
            [{"role": "user", "content": "prompt"}],
            response_model=self.ParsedOutput,
        )

        self.assertIsInstance(result, self.ParsedOutput)
        self.assertEqual(result.answer, "yes")

    @patch("llm_utils.lm.llm.get_base_client")
    def test_pydantic_parse_rejects_non_model_response_model(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = LLM()

        with self.assertRaises(SystemExit):
            llm.pydantic_parse("prompt", response_model=str)  # type: ignore[arg-type]

    @patch("llm_utils.lm.llm.get_base_client")
    def test_constructor_rejects_legacy_schema_kwargs(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        legacy_kwargs = (
            {"input_model": self.ParsedOutput},
            {"output_model": self.ParsedOutput},
            {"response_model": self.ParsedOutput},
            {"is_reasoning_model": True},
        )

        for kwargs in legacy_kwargs:
            label = next(iter(kwargs.keys()))  # Use just the key name for subTest
            with self.subTest(kwarg=label), self.assertRaises(TypeError):
                LLM(**kwargs)  # type: ignore[call-arg]

    @patch("llm_utils.lm.llm.get_base_client")
    def test_call_routes_to_chat_completion(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM()

        message = ChatCompletionMessage(role="assistant", content="hello")
        completion = self._make_completion(message)
        mock_client.chat.completions.create.return_value = completion

        result = llm("prompt")

        self.assertIs(result, message)
        mock_client.chat.completions.create.assert_called_once()
        mock_client.completions.create.assert_not_called()

    @patch("llm_utils.lm.llm.get_base_client")
    def test_call_requires_n_to_be_one(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = LLM()

        with self.assertRaises(ValueError):
            llm("prompt", n=2)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_call_rejects_return_dict_when_streaming(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = LLM()

        with self.assertRaises(ValueError):
            llm("prompt", stream=True, return_dict=True)

    @patch("llm_utils.lm.llm.get_base_client")
    def test_call_streams_from_chat_api(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLM()

        stream = object()
        mock_client.chat.completions.create.return_value = stream

        result = llm("prompt", stream=True)

        self.assertIs(result, stream)
        mock_client.chat.completions.create.assert_called_once()
        mock_client.completions.create.assert_not_called()
