import inspect
import unittest

import llm_utils
from llm_utils.chat_format.display import display_chat_messages_as_html, show_chat
from llm_utils.lm.llm_qwen3 import Qwen3LLM
from llm_utils.lm.llm_signature import LLMSignature
from llm_utils.lm.openai_memoize import MAsyncOpenAI, MOpenAI
from llm_utils.utils import get_one_turn_conv as utils_get_one_turn_conv
from llm_utils.utils import msgs_turns as utils_msgs_turns
from llm_utils.utils import turn as utils_turn


class TestPublicApi(unittest.TestCase):
    def test_top_level_exports_include_light_helpers(self):
        self.assertIs(llm_utils.get_one_turn_conv, utils_get_one_turn_conv)
        self.assertIs(llm_utils.turn, utils_turn)
        self.assertIs(llm_utils.msgs_turns, utils_msgs_turns)
        self.assertIs(llm_utils.LLM, llm_utils.lm.LLM)
        self.assertIs(llm_utils.Qwen3LLM, llm_utils.lm.Qwen3LLM)
        self.assertIs(llm_utils.MOpenAI, llm_utils.lm.MOpenAI)
        self.assertNotIn("__getattr__", llm_utils.__dict__)

    def test_heavy_lm_exports_remain_available_from_lm_package(self):
        self.assertTrue(hasattr(llm_utils.lm, "LLM"))
        self.assertTrue(hasattr(llm_utils.lm, "MOpenAI"))

    def test_conversation_helpers_build_chatml_messages(self):
        self.assertEqual(
            llm_utils.get_one_turn_conv("sys", "user", "assistant"),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "assistant"},
            ],
        )

    def test_display_chat_messages_as_html_matches_show_chat_signature(self):
        self.assertEqual(
            inspect.signature(display_chat_messages_as_html),
            inspect.signature(show_chat),
        )

    def test_llm_signature_constructor_surfaces_main_llm_options(self):
        params = inspect.signature(LLMSignature.__init__).parameters

        self.assertEqual(
            list(params.keys())[:8],
            [
                "self",
                "signature",
                "client",
                "cache",
                "verbose",
                "timeout",
                "enable_thinking",
                "model",
            ],
        )
        self.assertIn("max_tokens", params)
        self.assertIn("temperature", params)
        self.assertIn("frequency_penalty", params)
        self.assertEqual(params["model_kwargs"].kind, inspect.Parameter.VAR_KEYWORD)

    def test_qwen3_constructor_surfaces_prefix_and_llm_options(self):
        params = inspect.signature(Qwen3LLM.__init__).parameters

        self.assertEqual(
            list(params.keys())[:7],
            [
                "self",
                "client",
                "cache",
                "verbose",
                "timeout",
                "enable_thinking",
                "model",
            ],
        )
        self.assertIn("thinking_max_tokens", params)
        self.assertIn("content_max_tokens", params)
        self.assertEqual(params["model_kwargs"].kind, inspect.Parameter.VAR_KEYWORD)

    def test_mopenai_factories_surface_explicit_keyword_signatures(self):
        sync_params = inspect.signature(MOpenAI).parameters
        async_params = inspect.signature(MAsyncOpenAI).parameters

        expected_prefix = [
            "api_key",
            "organization",
            "project",
            "webhook_secret",
            "base_url",
            "websocket_base_url",
            "timeout",
            "max_retries",
            "default_headers",
            "default_query",
            "http_client",
            "_strict_response_validation",
            "cache",
        ]
        self.assertEqual(list(sync_params.keys())[: len(expected_prefix)], expected_prefix)
        self.assertEqual(
            list(async_params.keys())[: len(expected_prefix)], expected_prefix
        )
        self.assertEqual(sync_params["kwargs"].kind, inspect.Parameter.VAR_KEYWORD)
        self.assertEqual(async_params["kwargs"].kind, inspect.Parameter.VAR_KEYWORD)
        self.assertEqual(llm_utils.turn("assistant", "hello"), utils_turn("assistant", "hello"))
        self.assertEqual(
            llm_utils.msgs_turns(("s", "sys"), ("u", "user"), ("a", "assistant")),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "assistant"},
            ],
        )
