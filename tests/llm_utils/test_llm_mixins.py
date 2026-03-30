"""Unit tests for LLM mixins."""

import unittest
from unittest.mock import MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

try:
    from llm_utils.lm.llm import LLM
    from llm_utils.lm.mixins import TemperatureRangeMixin, TwoStepPydanticMixin
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    LLM = None
    TemperatureRangeMixin = None
    TwoStepPydanticMixin = None


pytestmark = pytest.mark.skipif(not LLM_AVAILABLE, reason="LLM not available")


class OutputModel(BaseModel):
    """Test Pydantic model."""

    text: str
    score: int


class TestTemperatureRangeMixin(unittest.TestCase):
    """Test TemperatureRangeMixin functionality."""

    @patch('speedy_utils.multi_worker.thread.multi_thread')
    @patch('llm_utils.lm.llm.get_base_client')
    def test_temperature_range_sampling(self, mock_get_client, mock_multi_thread):
        """Test temperature range sampling generates correct params."""
        # Setup mocks
        mock_get_client.return_value = MagicMock()
        mock_multi_thread.return_value = [
            {'parsed': f'response_{i}', 'messages': []} for i in range(5)
        ]

        # Create LLM instance
        llm = LLM()

        # Call temperature_range_sampling
        result = llm.temperature_range_sampling(
            'test input', temperature_ranges=(0.1, 1.0), n=5
        )

        # Verify multi_thread was called
        self.assertTrue(mock_multi_thread.called)

        # Check that we got the right number of results
        self.assertEqual(len(result), 5)

        # Verify the function was called with correct kwargs
        call_args = mock_multi_thread.call_args
        list_kwargs = call_args[0][1]  # Second positional arg

        # Check temperature values
        self.assertEqual(len(list_kwargs), 5)
        min_temp, max_temp = 0.1, 1.0
        step = (max_temp - min_temp) / (5 - 1)
        for i, kwargs in enumerate(list_kwargs):
            expected_temp = min_temp + i * step
            self.assertAlmostEqual(kwargs['temperature'], expected_temp, places=5)

    @patch('llm_utils.lm.llm.get_base_client')
    @patch('speedy_utils.multi_worker.thread.multi_thread')
    def test_temperature_range_via_call(self, mock_multi_thread, mock_get_client):
        """Test temperature_ranges parameter in __call__ method."""
        mock_get_client.return_value = MagicMock()
        mock_multi_thread.return_value = [
            {'parsed': f'response_{i}', 'messages': []} for i in range(3)
        ]

        llm = LLM()
        result = llm('test input', temperature_ranges=(0.1, 2.0), n=3)

        self.assertEqual(len(result), 3)
        self.assertTrue(mock_multi_thread.called)


class TestTwoStepPydanticMixin(unittest.TestCase):
    """Test TwoStepPydanticMixin functionality."""

    @patch('llm_utils.lm.llm.get_base_client')
    def test_two_step_parse_success(self, mock_get_client):
        """Test successful two-step parsing."""
        mock_get_client.return_value = MagicMock()
        llm = LLM(output_model=OutputModel)

        # Mock text_completion to return JSON string
        with patch.object(llm, 'text_completion') as mock_text:
            mock_text.return_value = [
                {
                    'parsed': '{"text": "hello", "score": 42}',
                    'messages': [{'role': 'user', 'content': 'test'}],
                }
            ]

            result = llm.two_step_pydantic_parse(
                'test input', response_model=OutputModel
            )

            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0]['parsed'], OutputModel)
            self.assertEqual(result[0]['parsed'].text, 'hello')
            self.assertEqual(result[0]['parsed'].score, 42)

    @patch('llm_utils.lm.llm.get_base_client')
    def test_two_step_parse_with_think_tags(self, mock_get_client):
        """Test parsing with reasoning <think> tags."""
        mock_get_client.return_value = MagicMock()
        llm = LLM(output_model=OutputModel)

        # Mock text_completion with <think> tags
        with patch.object(llm, 'text_completion') as mock_text:
            mock_text.return_value = [
                {
                    'parsed': '<think>reasoning here</think>{"text": "world", "score": 99}',
                    'messages': [{'role': 'user', 'content': 'test'}],
                }
            ]

            result = llm.two_step_pydantic_parse(
                'test input', response_model=OutputModel
            )

            # Should strip <think> and parse successfully
            self.assertEqual(result[0]['parsed'].text, 'world')
            self.assertEqual(result[0]['parsed'].score, 99)

    @patch('llm_utils.lm.llm.get_base_client')
    def test_two_step_parse_fallback(self, mock_get_client):
        """Test fallback to LLM extraction when JSON parsing fails."""
        mock_get_client.return_value = MagicMock()
        llm = LLM(output_model=OutputModel)

        # Mock text_completion with invalid JSON
        with patch.object(llm, 'text_completion') as mock_text:
            mock_text.return_value = [
                {
                    'parsed': 'This is not JSON at all',
                    'messages': [{'role': 'user', 'content': 'test'}],
                }
            ]

            # Mock pydantic_parse for fallback
            with patch.object(llm, 'pydantic_parse') as mock_parse:
                mock_parse.return_value = [
                    {'parsed': OutputModel(text='extracted', score=1), 'messages': []}
                ]

                result = llm.two_step_pydantic_parse(
                    'test input', response_model=OutputModel
                )

                # Should have called fallback
                self.assertTrue(mock_parse.called)
                self.assertEqual(result[0]['parsed'].text, 'extracted')

    @patch('llm_utils.lm.llm.get_base_client')
    def test_two_step_via_call(self, mock_get_client):
        """Test two_step_parse_pydantic parameter in __call__ method."""
        mock_get_client.return_value = MagicMock()
        llm = LLM(output_model=OutputModel)

        with patch.object(llm, 'text_completion') as mock_text:
            mock_text.return_value = [
                {
                    'parsed': '{"text": "test", "score": 5}',
                    'messages': [{'role': 'user', 'content': 'test'}],
                }
            ]

            result = llm('test input', two_step_parse_pydantic=True)

            self.assertTrue(mock_text.called)
            self.assertIsInstance(result[0]['parsed'], OutputModel)


class TestMixinIntegration(unittest.TestCase):
    """Test mixin integration with LLM class."""

    def test_llm_inherits_mixins(self):
        """Test that LLM inherits from both mixins."""
        self.assertTrue(issubclass(LLM, TemperatureRangeMixin))
        self.assertTrue(issubclass(LLM, TwoStepPydanticMixin))

    @patch('llm_utils.lm.llm.get_base_client')
    def test_llm_has_mixin_methods(self, mock_get_client):
        """Test that LLM instances have mixin methods."""
        mock_get_client.return_value = MagicMock()
        llm = LLM()
        self.assertTrue(hasattr(llm, 'temperature_range_sampling'))
        self.assertTrue(hasattr(llm, 'two_step_pydantic_parse'))
        self.assertTrue(callable(llm.temperature_range_sampling))
        self.assertTrue(callable(llm.two_step_pydantic_parse))


class TestLLMTimeout(unittest.TestCase):
    """Verify LLM respects configured OpenAI timeout defaults."""

    @patch('llm_utils.lm.llm.get_base_client')
    def test_timeout_applied_to_completion(self, mock_get_client):
        mock_client = MagicMock()
        mock_model = MagicMock(id='test-model')
        mock_client.models.list.return_value = MagicMock(data=[mock_model])
        mock_choice = MagicMock()
        mock_choice.message.content = 'hello'
        mock_completion = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_completion
        mock_get_client.return_value = mock_client

        llm = LLM(timeout=1.0)
        llm.text_completion('prompt')

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertIn('timeout', kwargs)
        self.assertEqual(kwargs['timeout'], 1.0)


class TestLLMEnableThinking(unittest.TestCase):
    """Verify sync LLM thinking-control merges into extra_body correctly."""

    class _StrictCompletions:
        def __init__(self, completion):
            self.calls = []
            self._completion = completion

        def create(
            self,
            *,
            model,
            messages,
            extra_body=None,
            timeout=None,
            n=1,
        ):
            self.calls.append(
                {
                    'model': model,
                    'messages': messages,
                    'extra_body': extra_body,
                    'timeout': timeout,
                    'n': n,
                }
            )
            return self._completion

    class _StrictChat:
        def __init__(self, completion):
            self.completions = TestLLMEnableThinking._StrictCompletions(completion)

    class _StrictClient:
        def __init__(self, completion, model='test-model'):
            self.chat = TestLLMEnableThinking._StrictChat(completion)
            self.models = MagicMock()
            self.models.list.return_value = MagicMock(
                data=[MagicMock(id=model)]
            )
            self.base_url = 'http://worker-10:8002/v1'

    @staticmethod
    def _make_mock_client():
        mock_client = MagicMock()
        mock_model = MagicMock(id='test-model')
        mock_client.models.list.return_value = MagicMock(data=[mock_model])
        return mock_client

    @staticmethod
    def _make_text_completion():
        mock_choice = MagicMock()
        mock_choice.message.content = 'hello'
        return MagicMock(choices=[mock_choice])

    @staticmethod
    def _make_parse_completion():
        mock_choice = MagicMock()
        mock_choice.message.content = '{"text": "hello", "score": 7}'
        mock_choice.message.parsed = OutputModel(text='hello', score=7)
        return MagicMock(choices=[mock_choice])

    @patch('llm_utils.lm.llm.get_base_client')
    def test_constructor_level_enable_thinking_false_sets_extra_body(
        self, mock_get_client
    ):
        mock_client = self._make_mock_client()
        mock_client.chat.completions.create.return_value = self._make_text_completion()
        mock_get_client.return_value = mock_client

        llm = LLM(enable_thinking=False)
        llm('prompt')

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(
            kwargs['extra_body'],
            {'chat_template_kwargs': {'enable_thinking': False}},
        )
        self.assertNotIn('enable_thinking', kwargs)

    @patch('llm_utils.lm.llm.get_base_client')
    def test_notebook_style_client_url_call_uses_extra_body_without_leaking_flag(
        self, mock_get_client
    ):
        client_url = 'http://worker-10:8002/v1'
        model = '/tmp/scratch/models/checkpoint-48'
        strict_client = self._StrictClient(
            self._make_text_completion(),
            model=model,
        )
        mock_get_client.return_value = strict_client
        llm_kwargs = {
            'client': client_url,
            'cache': False,
            'is_reasoning_model': True,
            'enable_thinking': False,
            'model': model,
        }

        llm = LLM(**llm_kwargs)
        self.assertEqual(llm.model, model)

        llm('hi')

        mock_get_client.assert_called_once_with(
            client_url,
            cache=False,
            api_key='abc',
            vllm_cmd=None,
            vllm_process=None,
        )
        self.assertEqual(len(strict_client.chat.completions.calls), 1)
        kwargs = strict_client.chat.completions.calls[0]
        self.assertEqual(kwargs['model'], model)
        self.assertEqual(
            kwargs['extra_body'],
            {'chat_template_kwargs': {'enable_thinking': False}},
        )
        self.assertNotIn('enable_thinking', kwargs)

    @patch('llm_utils.lm.llm.get_base_client')
    def test_runtime_enable_thinking_overrides_instance_default(
        self, mock_get_client
    ):
        mock_client = self._make_mock_client()
        mock_client.chat.completions.create.return_value = self._make_text_completion()
        mock_get_client.return_value = mock_client

        llm = LLM(enable_thinking=True)
        llm.text_completion('prompt', enable_thinking=False)

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(
            kwargs['extra_body'],
            {'chat_template_kwargs': {'enable_thinking': False}},
        )
        self.assertNotIn('enable_thinking', kwargs)

    @patch('llm_utils.lm.llm.get_base_client')
    def test_stale_enable_thinking_in_model_kwargs_is_filtered(
        self, mock_get_client
    ):
        mock_client = self._make_mock_client()
        mock_client.chat.completions.create.return_value = self._make_text_completion()
        mock_get_client.return_value = mock_client

        llm = LLM()
        llm.model_kwargs['enable_thinking'] = False

        llm.text_completion('prompt')

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertNotIn('enable_thinking', kwargs)
        self.assertNotIn('extra_body', kwargs)

    @patch('llm_utils.lm.llm.get_base_client')
    def test_existing_extra_body_keys_are_preserved(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_client.chat.completions.create.return_value = self._make_text_completion()
        mock_get_client.return_value = mock_client

        llm = LLM()
        llm.text_completion(
            'prompt',
            enable_thinking=False,
            extra_body={'top_k': 20},
        )

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(
            kwargs['extra_body'],
            {
                'top_k': 20,
                'chat_template_kwargs': {'enable_thinking': False},
            },
        )

    @patch('llm_utils.lm.llm.get_base_client')
    def test_existing_chat_template_kwargs_are_preserved(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_client.chat.completions.create.return_value = self._make_text_completion()
        mock_get_client.return_value = mock_client

        llm = LLM()
        llm.text_completion(
            'prompt',
            enable_thinking=False,
            extra_body={
                'chat_template_kwargs': {'foo': 'bar'},
                'top_p': 0.9,
            },
        )

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(
            kwargs['extra_body'],
            {
                'chat_template_kwargs': {
                    'foo': 'bar',
                    'enable_thinking': False,
                },
                'top_p': 0.9,
            },
        )

    @patch('llm_utils.lm.llm.get_base_client')
    def test_runtime_extra_body_enable_thinking_wins(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_client.chat.completions.create.return_value = self._make_text_completion()
        mock_get_client.return_value = mock_client

        llm = LLM(enable_thinking=False)
        llm.text_completion(
            'prompt',
            enable_thinking=True,
            extra_body={'chat_template_kwargs': {'enable_thinking': False}},
        )

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(
            kwargs['extra_body'],
            {'chat_template_kwargs': {'enable_thinking': False}},
        )

    @patch('llm_utils.lm.llm.get_base_client')
    def test_stream_text_completion_uses_same_merge_logic(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_client.chat.completions.create.return_value = iter([])
        mock_get_client.return_value = mock_client

        llm = LLM(enable_thinking=False)
        llm.stream_text_completion(
            'prompt',
            extra_body={'top_k': 10},
        )

        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertTrue(kwargs['stream'])
        self.assertEqual(
            kwargs['extra_body'],
            {
                'top_k': 10,
                'chat_template_kwargs': {'enable_thinking': False},
            },
        )

    @patch('llm_utils.lm.llm.get_base_client')
    def test_pydantic_parse_uses_same_merge_logic(self, mock_get_client):
        mock_client = self._make_mock_client()
        mock_client.chat.completions.parse.return_value = self._make_parse_completion()
        mock_get_client.return_value = mock_client

        llm = LLM(enable_thinking=False, output_model=OutputModel)
        llm.pydantic_parse(
            'prompt',
            extra_body={'top_k': 5},
        )

        _, kwargs = mock_client.chat.completions.parse.call_args
        self.assertEqual(
            kwargs['extra_body'],
            {
                'top_k': 5,
                'chat_template_kwargs': {'enable_thinking': False},
            },
        )


if __name__ == '__main__':
    unittest.main()
