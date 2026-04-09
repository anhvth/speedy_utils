import unittest

import llm_utils
from llm_utils.utils import get_one_turn_conv as utils_get_one_turn_conv
from llm_utils.utils import msgs_turns as utils_msgs_turns
from llm_utils.utils import turn as utils_turn


class TestPublicApi(unittest.TestCase):
    def test_top_level_exports_are_direct_reexports(self):
        self.assertIs(llm_utils.get_one_turn_conv, utils_get_one_turn_conv)
        self.assertIs(llm_utils.turn, utils_turn)
        self.assertIs(llm_utils.msgs_turns, utils_msgs_turns)
        self.assertNotIn("__getattr__", llm_utils.__dict__)

    def test_conversation_helpers_build_chatml_messages(self):
        self.assertEqual(
            llm_utils.get_one_turn_conv("sys", "user", "assistant"),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "assistant"},
            ],
        )
        self.assertEqual(llm_utils.turn("assistant", "hello"), utils_turn("assistant", "hello"))
        self.assertEqual(
            llm_utils.msgs_turns(("s", "sys"), ("u", "user"), ("a", "assistant")),
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "user"},
                {"role": "assistant", "content": "assistant"},
            ],
        )