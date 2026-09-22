import inspect
from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock, patch

import pytest

from llm_utils import DS41LLM
from llm_utils.lm.llm_ds41 import (
    DEFAULT_DS41_MODEL,
    DEFAULT_REASONING_EFFORT,
    ReasoningEffortName,
)


def make_mock_client():
    client = MagicMock()
    client.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id=DEFAULT_DS41_MODEL)]
    )
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content="hi"),
                index=0,
                finish_reason="stop",
            )
        ],
        usage=None,
    )
    return client


def test_ds41_constructor_does_not_assume_port_7788():
    params = inspect.signature(DS41LLM.__init__).parameters

    assert params["client"].default is None
    assert params["model"].default == DEFAULT_DS41_MODEL


def test_ds41_supported_named_reasoning_efforts_are_typed_as_literal():
    assert get_args(ReasoningEffortName) == ("low", "high", "max")
    assert DEFAULT_REASONING_EFFORT == "high"
    assert DS41LLM.REASONING_EFFORT_MAPPINGS == {
        "low": 50,
        "high": 75,
        "max": 100,
    }


def test_ds41_wires_the_checkpoint_tokenizer_to_the_class():
    assert DS41LLM.TOKENIZER_NAME == "deepseek-ai/DeepSeek-V4.1-Flash"


@pytest.mark.parametrize("reasoning_effort", ["low", "high", "max", 1, 50, 100])
def test_ds41_accepts_every_tokenizer_supported_reasoning_effort(reasoning_effort):
    client = make_mock_client()
    with patch("llm_utils.lm.llm.get_base_client", return_value=client):
        llm = DS41LLM(reasoning_effort=reasoning_effort)

    assert llm.get_model_sampling_params()["reasoning_effort"] == reasoning_effort


@patch("llm_utils.lm.llm.get_base_client")
def test_ds41_enables_reasoning_with_tokenizer_default(mock_get_client):
    client = make_mock_client()
    mock_get_client.return_value = client

    DS41LLM().chat_completion("hello")

    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["extra_body"]["chat_template_kwargs"] == {
        "thinking": True,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
    }
    assert "reasoning_effort" not in kwargs


@patch("llm_utils.lm.llm.get_base_client")
def test_ds41_accepts_per_call_numeric_reasoning_effort(mock_get_client):
    client = make_mock_client()
    mock_get_client.return_value = client

    DS41LLM(reasoning_effort=75).chat_completion(
        "hello",
        reasoning_effort=42,
    )

    chat_kwargs = client.chat.completions.create.call_args.kwargs["extra_body"][
        "chat_template_kwargs"
    ]
    assert chat_kwargs == {"thinking": True, "reasoning_effort": 42}


@patch("llm_utils.lm.llm.get_base_client")
def test_ds41_disabling_reasoning_omits_effort(mock_get_client):
    client = make_mock_client()
    mock_get_client.return_value = client

    DS41LLM(reasoning_effort=100).chat_completion(
        "hello",
        enable_thinking=False,
    )

    chat_kwargs = client.chat.completions.create.call_args.kwargs["extra_body"][
        "chat_template_kwargs"
    ]
    assert chat_kwargs == {"thinking": False}


@pytest.mark.parametrize("response_field", ["reasoning", "reasoning_content"])
@patch("llm_utils.lm.llm.get_base_client")
def test_ds41_raises_if_disabled_model_emits_reasoning(
    mock_get_client,
    response_field,
):
    client = make_mock_client()
    setattr(
        client.chat.completions.create.return_value.choices[0].message,
        response_field,
        "unexpected reasoning",
    )
    mock_get_client.return_value = client

    with pytest.raises(RuntimeError, match="reasoning was disabled"):
        DS41LLM(enable_thinking=False).chat_completion("hello")


@patch("llm_utils.lm.llm.get_base_client")
def test_ds41_per_call_disable_raises_on_reasoning_in_return_dict(mock_get_client):
    client = make_mock_client()
    client.chat.completions.create.return_value.choices[0].message.reasoning = (
        "unexpected reasoning"
    )
    mock_get_client.return_value = client

    with pytest.raises(RuntimeError, match="reasoning was disabled"):
        DS41LLM()("hello", return_dict=True, enable_thinking=False)


@patch("llm_utils.lm.llm.get_base_client")
def test_ds41_constructor_can_disable_and_call_can_reenable_reasoning(
    mock_get_client,
):
    client = make_mock_client()
    mock_get_client.return_value = client
    llm = DS41LLM(enable_thinking=False, reasoning_effort="max")

    llm.chat_completion("off")
    disabled_kwargs = client.chat.completions.create.call_args.kwargs
    assert disabled_kwargs["extra_body"]["chat_template_kwargs"] == {
        "thinking": False
    }

    llm.chat_completion("on", enable_thinking=True)
    enabled_kwargs = client.chat.completions.create.call_args.kwargs
    assert enabled_kwargs["extra_body"]["chat_template_kwargs"] == {
        "thinking": True,
        "reasoning_effort": "max",
    }


@patch("llm_utils.lm.llm.get_base_client")
def test_ds41_return_dict_path_uses_reasoning_settings(mock_get_client):
    client = make_mock_client()
    mock_get_client.return_value = client

    DS41LLM()("hello", return_dict=True, reasoning_effort="high")

    chat_kwargs = client.chat.completions.create.call_args.kwargs["extra_body"][
        "chat_template_kwargs"
    ]
    assert chat_kwargs == {"thinking": True, "reasoning_effort": "high"}


@patch("llm_utils.lm.llm.get_base_client")
def test_ds41_normalizes_deepseek_reasoning_content(mock_get_client):
    client = make_mock_client()
    client.chat.completions.create.return_value.choices[
        0
    ].message.reasoning_content = "worked it out"
    mock_get_client.return_value = client

    result = DS41LLM()("hello", return_dict=True)

    assert result["reasoning"] == "worked it out"
    assert result["messages"][-1]["reasoning"] == "worked it out"


@pytest.mark.parametrize(
    "reasoning_effort",
    [-1, 0, 101, True, 1.5, "minimal", "medium", "xhigh", "none", []],
)
def test_ds41_rejects_every_unsupported_reasoning_effort(reasoning_effort):
    with pytest.raises(ValueError, match="reasoning_effort"):
        DS41LLM(reasoning_effort=reasoning_effort)


@patch("llm_utils.lm.llm.get_base_client")
def test_ds41_rejects_unsupported_per_call_effort_before_request(mock_get_client):
    client = make_mock_client()
    mock_get_client.return_value = client
    llm = DS41LLM()

    with pytest.raises(ValueError, match="reasoning_effort"):
        llm.chat_completion("hello", reasoning_effort="xhigh")

    client.chat.completions.create.assert_not_called()


def test_ds41_sampling_params_follow_reasoning_state():
    client = make_mock_client()
    with patch("llm_utils.lm.llm.get_base_client", return_value=client):
        llm = DS41LLM(
            enable_thinking=True,
            reasoning_effort=60,
            temperature=1.0,
            top_p=0.95,
        )

    assert llm.get_model_sampling_params() == {
        "reasoning_effort": 60,
        "temperature": 1.0,
        "top_p": 0.95,
    }
    assert llm.get_model_sampling_params(enable_thinking=False) == {
        "temperature": 1.0,
        "top_p": 0.95,
    }
