"""Test Qwen3 prefix/tokenizer helper behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from llm_utils.lm.llm_qwen3 import (
    ASSISTANT_END,
    ASSISTANT_PREFIX,
    THINK_END,
    THINK_START,
    Qwen3LLM,
    build_assistant_prefix,
    split_assistant_parts,
)


def _make_mock_client():
    mock_client = MagicMock()
    mock_model = MagicMock(id="test-model")
    mock_client.models.list.return_value = SimpleNamespace(data=[mock_model])
    return mock_client


def test_split_assistant_parts_with_reasoning_and_content():
    text = (
        f"{ASSISTANT_PREFIX}\n{THINK_START}\n"
        "seed reasoning\n"
        f"{THINK_END} final answer {ASSISTANT_END}"
    )

    reasoning, content, think_done = split_assistant_parts(text)

    assert reasoning == "seed reasoning"
    assert content == "final answer"
    assert think_done is True


def test_split_assistant_parts_with_content_only():
    reasoning, content, think_done = split_assistant_parts("hello world")

    assert reasoning is None
    assert content == "hello world"
    assert think_done is True


def test_build_assistant_prefix_normalizes_both_phases():
    assert build_assistant_prefix("seed", "final", True) == (
        f"{ASSISTANT_PREFIX}\n{THINK_START}\nseed\n{THINK_END}final"
    )
    assert build_assistant_prefix("seed", None, False) == (
        f"{ASSISTANT_PREFIX}\n{THINK_START}\nseed"
    )


@patch("llm_utils.lm.llm.get_base_client")
def test_generate_with_prefix_step_uses_tokenizer_and_generate_response(
    mock_get_client,
):
    mock_get_client.return_value = _make_mock_client()
    llm = Qwen3LLM()

    class FakeTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            self.calls.append((messages, tokenize, add_generation_prompt))
            return "TOK_PROMPT"

    fake_tokenizer = FakeTokenizer()

    with (
        patch.object(Qwen3LLM, "_get_tokenizer", return_value=fake_tokenizer),
        patch.object(
            llm,
            "_generate_response",
            return_value={"text": "reasoning step", "stop": "stop"},
        ) as mock_generate_response,
    ):
        text, stop_reason = llm._generate_with_prefix_step(
            [{"role": "user", "content": "hi"}],
            "<think>\nseed",
            temperature=0.2,
            max_tokens=99,
            extra_body={"ignored": True},
        )

    assert fake_tokenizer.calls == [
        (
            [{"role": "user", "content": "hi"}],
            False,
            False,
        )
    ]
    assert mock_generate_response.call_args.args == ("TOK_PROMPT<think>\nseed",)
    assert mock_generate_response.call_args.kwargs == {
        "temperature": 0.2,
        "max_tokens": 99,
    }
    assert text == "reasoning step"
    assert stop_reason == "stop"
