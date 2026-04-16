"""Test Qwen3 prefix/tokenizer helper behavior."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from openai.types.completion_choice import CompletionChoice
from openai.types.completion_usage import CompletionUsage

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
    # build_assistant_prefix returns wrapper-free text (no ASSISTANT_PREFIX)
    assert build_assistant_prefix("seed", "final", True) == (
        f"{THINK_START}\nseed\n{THINK_END}final"
    )
    assert build_assistant_prefix("seed", None, False) == (
        f"{THINK_START}\nseed"
    )


@patch("llm_utils.lm.llm.get_base_client")
def test_generate_with_prefix_step_uses_tokenizer_and_completion_api(
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
    choice = CompletionChoice(
        finish_reason="stop",
        index=0,
        logprobs=None,
        text="reasoning step",
    )
    usage = CompletionUsage(
        completion_tokens=4,
        prompt_tokens=9,
        total_tokens=13,
    )
    completion = SimpleNamespace(choices=[choice], usage=usage)

    with (
        patch.object(Qwen3LLM, "_get_tokenizer", return_value=fake_tokenizer),
        patch.object(
            llm.client.completions, "create", return_value=completion
        ) as mock_completion_create,
    ):
        result = llm._generate_with_prefix_step(
            [{"role": "user", "content": "hi"}],
            "<think>\nseed",
            temperature=0.2,
            max_tokens=99,
        )

    assert fake_tokenizer.calls == [
        (
            [{"role": "user", "content": "hi"}],
            False,
            False,
        )
    ]
    assert mock_completion_create.call_args.kwargs == {
        "model": "test-model",
        "prompt": "TOK_PROMPT<|im_start|>assistant\n<think>\nseed",
        "temperature": 0.2,
        "max_tokens": 99,
    }
    assert result.text == "reasoning step"
    assert result.finish_reason == "stop"
    assert result.usage is usage


@patch("llm_utils.lm.llm.get_base_client")
def test_generate_with_prefix_step_uses_assistant_body_verbatim(
    mock_get_client,
):
    mock_get_client.return_value = _make_mock_client()
    llm = Qwen3LLM(enable_thinking=False)

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hi"}]
            assert tokenize is False
            assert add_generation_prompt is False
            return "TOK_PROMPT"

    choice = CompletionChoice(
        finish_reason="stop",
        index=0,
        logprobs=None,
        text="final answer",
    )
    usage = CompletionUsage(
        completion_tokens=4,
        prompt_tokens=9,
        total_tokens=13,
    )
    completion = SimpleNamespace(choices=[choice], usage=usage)

    with (
        patch.object(Qwen3LLM, "_get_tokenizer", return_value=FakeTokenizer()),
        patch.object(
            llm.client.completions, "create", return_value=completion
        ) as mock_completion_create,
    ):
        # _generate_with_prefix_step takes assistant-body text and wraps it
        llm._generate_with_prefix_step(
            [{"role": "user", "content": "hi"}],
            "<think>\nseed",
            max_tokens=1,
        )

    # enable_thinking is consumed by caller; body is used verbatim
    assert mock_completion_create.call_args.kwargs == {
        "model": "test-model",
        "prompt": "TOK_PROMPT<|im_start|>assistant\n<think>\nseed",
        "max_tokens": 1,
    }
