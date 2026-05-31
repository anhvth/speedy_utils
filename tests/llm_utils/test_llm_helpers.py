"""Examples-as-tests for LLM normalization helpers (no live endpoint)."""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import MagicMock, patch

import pytest

from llm_utils.lm.llm import LLM


class TestPrepareInput(TestCase):
    def test_str_becomes_single_user_message(self):
        assert LLM._prepare_input("hi") == [{"role": "user", "content": "hi"}]

    def test_message_list_passes_through(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert LLM._prepare_input(msgs) is msgs


class TestRequireSingleChoice(TestCase):
    def test_default_n_is_allowed(self):
        LLM._require_single_choice({})

    def test_n_greater_than_one_is_rejected(self):
        with pytest.raises(ValueError):
            LLM._require_single_choice({"n": 2})


class TestBuildApiKwargs(TestCase):
    @staticmethod
    def _make_llm():
        client = MagicMock()
        client.models.list.return_value = MagicMock(data=[MagicMock(id="m")])
        with patch("llm_utils.lm.llm.get_base_client", return_value=client):
            return LLM()

    def test_enable_thinking_sets_chat_template_kwargs(self):
        llm = self._make_llm()
        _, api_kwargs = llm._build_api_kwargs({}, enable_thinking=True)
        chat_kwargs = api_kwargs["extra_body"]["chat_template_kwargs"]
        assert chat_kwargs["enable_thinking"] is True

    def test_enable_thinking_does_not_mutate_caller_extra_body(self):
        llm = self._make_llm()
        extra_body: dict = {}
        llm._build_api_kwargs({"extra_body": extra_body}, enable_thinking=True)
        assert extra_body == {}

    def test_model_name_is_pulled_out_of_api_kwargs(self):
        llm = self._make_llm()
        model_name, api_kwargs = llm._build_api_kwargs({"model": "custom"})
        assert model_name == "custom"
        assert "model" not in api_kwargs
