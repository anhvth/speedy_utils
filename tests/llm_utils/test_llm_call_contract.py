"""Unit tests for the simplified sync LLM public API."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import MagicMock, patch

from openai.types.chat import ChatCompletionMessage
from openai.types.completion import CompletionChoice
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
            finish_reason=finish_reason,
            index=0,
            logprobs=None,
            text=text,
        )
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
        self.assertIs(llm.last_ai_response, completion)

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
        self.assertIs(llm.last_ai_response, completion)
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
        self.assertIs(llm.last_ai_response, completion)

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
    def test_call_can_return_dict_for_text_completion(self, mock_get_client):
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
        self.assertIs(llm.last_ai_response, completion)

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
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                LLM(**kwargs)

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
