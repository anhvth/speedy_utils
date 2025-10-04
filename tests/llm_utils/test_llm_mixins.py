"""Unit tests for LLM mixins."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from pydantic import BaseModel

from llm_utils.lm.llm import LLM
from llm_utils.lm.mixins import TemperatureRangeMixin, TwoStepPydanticMixin


class TestOutput(BaseModel):
    """Test Pydantic model."""

    text: str
    score: int


class TestTemperatureRangeMixin(unittest.TestCase):
    """Test TemperatureRangeMixin functionality."""

    @patch("llm_utils.lm.mixins.multi_thread")
    def test_temperature_range_sampling(self, mock_multi_thread):
        """Test temperature range sampling generates correct params."""
        # Setup mock
        mock_multi_thread.return_value = [{"parsed": f"response_{i}", "messages": []} for i in range(5)]

        # Create LLM instance
        llm = LLM()

        # Call temperature_range_sampling
        result = llm.temperature_range_sampling("test input", n=5, max_temperature=1.0)

        # Verify multi_thread was called
        self.assertTrue(mock_multi_thread.called)

        # Check that we got the right number of results
        self.assertEqual(len(result), 5)

        # Verify the function was called with correct kwargs
        call_args = mock_multi_thread.call_args
        list_kwargs = call_args[0][1]  # Second positional arg

        # Check temperature values
        self.assertEqual(len(list_kwargs), 5)
        for i, kwargs in enumerate(list_kwargs):
            expected_temp = 0.1 + i * (1.0 / 5)
            self.assertAlmostEqual(kwargs["temperature"], expected_temp, places=2)

    def test_temperature_range_via_call(self):
        """Test temperature_ranges parameter in __call__ method."""
        with patch("llm_utils.lm.mixins.multi_thread") as mock_multi_thread:
            mock_multi_thread.return_value = [{"parsed": f"response_{i}", "messages": []} for i in range(3)]

            llm = LLM()
            result = llm("test input", temperature_ranges=3, max_temperature=2.0)

            self.assertEqual(len(result), 3)
            self.assertTrue(mock_multi_thread.called)


class TestTwoStepPydanticMixin(unittest.TestCase):
    """Test TwoStepPydanticMixin functionality."""

    def test_two_step_parse_success(self):
        """Test successful two-step parsing."""
        llm = LLM(output_model=TestOutput)

        # Mock text_completion to return JSON string
        with patch.object(llm, "text_completion") as mock_text:
            mock_text.return_value = [
                {"parsed": '{"text": "hello", "score": 42}', "messages": [{"role": "user", "content": "test"}]}
            ]

            result = llm.two_step_pydantic_parse("test input", response_model=TestOutput)

            self.assertEqual(len(result), 1)
            self.assertIsInstance(result[0]["parsed"], TestOutput)
            self.assertEqual(result[0]["parsed"].text, "hello")
            self.assertEqual(result[0]["parsed"].score, 42)

    def test_two_step_parse_with_think_tags(self):
        """Test parsing with reasoning <think> tags."""
        llm = LLM(output_model=TestOutput)

        # Mock text_completion with <think> tags
        with patch.object(llm, "text_completion") as mock_text:
            mock_text.return_value = [
                {
                    "parsed": '<think>reasoning here</think>{"text": "world", "score": 99}',
                    "messages": [{"role": "user", "content": "test"}],
                }
            ]

            result = llm.two_step_pydantic_parse("test input", response_model=TestOutput)

            # Should strip <think> and parse successfully
            self.assertEqual(result[0]["parsed"].text, "world")
            self.assertEqual(result[0]["parsed"].score, 99)

    def test_two_step_parse_fallback(self):
        """Test fallback to LLM extraction when JSON parsing fails."""
        llm = LLM(output_model=TestOutput)

        # Mock text_completion with invalid JSON
        with patch.object(llm, "text_completion") as mock_text:
            mock_text.return_value = [
                {"parsed": "This is not JSON at all", "messages": [{"role": "user", "content": "test"}]}
            ]

            # Mock pydantic_parse for fallback
            with patch.object(llm, "pydantic_parse") as mock_parse:
                mock_parse.return_value = [{"parsed": TestOutput(text="extracted", score=1), "messages": []}]

                result = llm.two_step_pydantic_parse("test input", response_model=TestOutput)

                # Should have called fallback
                self.assertTrue(mock_parse.called)
                self.assertEqual(result[0]["parsed"].text, "extracted")

    def test_two_step_via_call(self):
        """Test two_step_parse_pydantic parameter in __call__ method."""
        llm = LLM(output_model=TestOutput)

        with patch.object(llm, "text_completion") as mock_text:
            mock_text.return_value = [
                {"parsed": '{"text": "test", "score": 5}', "messages": [{"role": "user", "content": "test"}]}
            ]

            result = llm("test input", two_step_parse_pydantic=True)

            self.assertTrue(mock_text.called)
            self.assertIsInstance(result[0]["parsed"], TestOutput)


class TestMixinIntegration(unittest.TestCase):
    """Test mixin integration with LLM class."""

    def test_llm_inherits_mixins(self):
        """Test that LLM inherits from both mixins."""
        self.assertTrue(issubclass(LLM, TemperatureRangeMixin))
        self.assertTrue(issubclass(LLM, TwoStepPydanticMixin))

    def test_llm_has_mixin_methods(self):
        """Test that LLM instances have mixin methods."""
        llm = LLM()
        self.assertTrue(hasattr(llm, "temperature_range_sampling"))
        self.assertTrue(hasattr(llm, "two_step_pydantic_parse"))
        self.assertTrue(callable(llm.temperature_range_sampling))
        self.assertTrue(callable(llm.two_step_pydantic_parse))


if __name__ == "__main__":
    unittest.main()
