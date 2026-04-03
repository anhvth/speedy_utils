"""Unit tests for LLMSignature."""

from types import SimpleNamespace
from typing import Annotated
from unittest import TestCase
from unittest.mock import MagicMock, patch

from pydantic import BaseModel

from llm_utils.lm.llm_signature import LLMSignature
from llm_utils.lm.signature import Input, Output, Signature


class TestLLMSignature(TestCase):
    """Verify signature-backed defaults and parsing behavior."""

    class JudgeSignature(Signature):
        """Judge whether the answer is correct."""

        question: Annotated[str, Input("Question text")]
        answer: Annotated[str, Input("Candidate answer")]
        verdict: Annotated[str, Output("One-word verdict")]

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

    @patch("llm_utils.lm.llm.get_base_client")
    def test_signature_exposes_derived_models(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = LLMSignature(signature=self.JudgeSignature)

        expected_input_model = self.JudgeSignature.get_input_model()
        expected_output_model = self.JudgeSignature.get_output_model()

        self.assertEqual(
            llm.input_model.model_json_schema(),
            expected_input_model.model_json_schema(),
        )
        self.assertEqual(
            llm.output_model.model_json_schema(),
            expected_output_model.model_json_schema(),
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_call_defaults_to_signature_output_model(
        self, mock_get_client
    ):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLMSignature(signature=self.JudgeSignature)

        parsed_message = SimpleNamespace(
            content='{"verdict": "yes"}',
            parsed={"verdict": "yes"},
        )
        completion = self._make_completion(parsed_message)
        mock_client.chat.completions.parse.return_value = completion

        result = llm("prompt", return_dict=True)

        self.assertIs(result["completion"], completion)
        self.assertIs(result["message"], parsed_message)
        self.assertIsInstance(result["parsed"], llm.output_model)
        self.assertEqual(result["parsed"].verdict, "yes")
        self.assertEqual(
            mock_client.chat.completions.parse.call_args.kwargs["messages"],
            [{"role": "user", "content": "prompt"}],
        )
        self.assertIs(
            mock_client.chat.completions.parse.call_args.kwargs["response_format"],
            llm.output_model,
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_pydantic_parse_defaults_to_signature_output_model(
        self, mock_get_client
    ):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLMSignature(signature=self.JudgeSignature)

        parsed_message = SimpleNamespace(
            content='{"verdict": "maybe"}',
            parsed={"verdict": "maybe"},
        )
        completion = self._make_completion(parsed_message)
        mock_client.chat.completions.parse.return_value = completion

        result = llm.pydantic_parse("prompt")

        self.assertIsInstance(result, llm.output_model)
        self.assertEqual(result.verdict, "maybe")
        self.assertIs(
            mock_client.chat.completions.parse.call_args.kwargs["response_format"],
            llm.output_model,
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_pydantic_parse_accepts_message_list(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()
        llm = LLMSignature(signature=self.JudgeSignature)

        parsed_message = SimpleNamespace(
            content='{"verdict": "yes"}',
            parsed={"verdict": "yes"},
        )
        completion = self._make_completion(parsed_message)
        mock_client = llm.client
        mock_client.chat.completions.parse.return_value = completion

        result = llm.pydantic_parse(
            [{"role": "user", "content": "prompt"}],
        )

        self.assertIsInstance(result, llm.output_model)
        self.assertEqual(result.verdict, "yes")

    @patch("llm_utils.lm.llm.get_base_client")
    def test_pydantic_parse_can_override_signature_output_model(
        self, mock_get_client
    ):
        mock_client = self._make_mock_client()
        mock_get_client.return_value = mock_client
        llm = LLMSignature(signature=self.JudgeSignature)

        parsed_message = SimpleNamespace(
            content='{"verdict": "no"}',
            parsed={"verdict": "no"},
        )
        completion = self._make_completion(parsed_message)
        mock_client.chat.completions.parse.return_value = completion

        class OverrideOutput(BaseModel):
            verdict: str

        result = llm.pydantic_parse("prompt", response_model=OverrideOutput)

        self.assertIsInstance(result, OverrideOutput)
        self.assertEqual(result.verdict, "no")
        self.assertIs(
            mock_client.chat.completions.parse.call_args.kwargs["response_format"],
            OverrideOutput,
        )

    @patch("llm_utils.lm.llm.get_base_client")
    def test_legacy_constructor_arguments_are_rejected(self, mock_get_client):
        mock_get_client.return_value = self._make_mock_client()

        legacy_kwargs = (
            {"input_model": BaseModel},
            {"output_model": BaseModel},
            {"response_model": BaseModel},
            {"is_reasoning_model": True},
        )

        for kwargs in legacy_kwargs:
            with self.subTest(kwargs=kwargs), self.assertRaises(TypeError):
                LLMSignature(signature=self.JudgeSignature, **kwargs)
