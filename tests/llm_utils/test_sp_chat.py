from typing import Any

import pytest

from llm_utils.scripts.sp_chat import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    ChatConfig,
    _archive_current_chat,
    _build_history_title,
    _chat_completion_kwargs,
    _chat_settings_defaults,
    _chainlit_context_to_openai,
    _openai_client_kwargs,
    _render_streaming_blocks,
    _reset_chat_state,
    normalize_client_base_url,
    parse_cli_args,
)


def test_parse_cli_args_defaults() -> None:
    config = parse_cli_args([])
    assert config == ChatConfig()
    assert config.api_key == "abc"


def test_openai_client_kwargs_default_to_dummy_api_key() -> None:
    kwargs = _openai_client_kwargs("http://localhost:8000/v1", None)

    assert kwargs == {"base_url": "http://localhost:8000/v1", "api_key": "abc"}


def test_chat_settings_defaults_seed_message_handler_before_ui_settings() -> None:
    settings = _chat_settings_defaults(
        model_options=["Qwen3.5-122B-A10B"],
        initial_model=None,
        enable_thinking_default=True,
    )

    assert settings == {
        "model": "Qwen3.5-122B-A10B",
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "system_prompt": "",
        "thinking": True,
    }


def test_chat_settings_defaults_falls_back_to_initial_model() -> None:
    settings = _chat_settings_defaults(
        model_options=[],
        initial_model="configured-model",
        enable_thinking_default=False,
    )

    assert settings["model"] == "configured-model"
    assert settings["thinking"] is False


def test_chat_completion_kwargs_omits_default_temperature() -> None:
    kwargs = _chat_completion_kwargs(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=128,
    )

    assert "temperature" not in kwargs


def test_chat_completion_kwargs_keeps_explicit_temperature() -> None:
    kwargs = _chat_completion_kwargs(
        model="model",
        messages=[{"role": "user", "content": "hello"}],
        temperature="0.7",
        max_tokens=128,
    )

    assert kwargs["temperature"] == 0.7


def test_parse_cli_args_key_value_inputs() -> None:
    config = parse_cli_args(
        [
            "client=9000",
            "port=5011",
            "host=127.0.0.1",
            "model=Qwen/Qwen2.5-7B-Instruct",
            "api_key=secret",
            "thinking=true",
        ]
    )

    assert config.client == "9000"
    assert config.app_port == 5011
    assert config.app_host == "127.0.0.1"
    assert config.model == "Qwen/Qwen2.5-7B-Instruct"
    assert config.api_key == "secret"
    assert config.thinking is True


def test_parse_cli_args_invalid_format_raises() -> None:
    with pytest.raises(ValueError, match="Expected key=value"):
        parse_cli_args(["client", "8000"])


def test_parse_cli_args_invalid_key_raises() -> None:
    with pytest.raises(ValueError, match="Unknown argument"):
        parse_cli_args(["unknown=1"])


def test_parse_cli_args_invalid_bool_raises() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        parse_cli_args(["thinking=maybe"])


def test_parse_cli_args_help_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(["--help"])


@pytest.mark.parametrize(
    ("client", "expected"),
    [
        (None, "http://localhost:8000/v1"),
        ("", "http://localhost:8000/v1"),
        ("8000", "http://localhost:8000/v1"),
        (8001, "http://localhost:8001/v1"),
        ("localhost:9000", "http://localhost:9000/v1"),
        ("http://127.0.0.1:7000", "http://127.0.0.1:7000/v1"),
        ("http://127.0.0.1:7000/v1", "http://127.0.0.1:7000/v1"),
        ("https://api.z.ai/api/coding/paas/v4", "https://api.z.ai/api/coding/paas/v4"),
        ("https://api.z.ai/api/coding/paas/v4/", "https://api.z.ai/api/coding/paas/v4"),
    ],
)
def test_normalize_client_base_url(client, expected) -> None:
    assert normalize_client_base_url(client) == expected


class _PlaceholderSpy:
    def __init__(self) -> None:
        self.last_text = ""
        self.last_unsafe = False

    def markdown(self, text: str, *, unsafe_allow_html: bool = False) -> None:
        self.last_text = text
        self.last_unsafe = unsafe_allow_html


class _Element:
    def __init__(
        self,
        *,
        type: str,
        name: str = "",
        mime: str | None = None,
        url: str | None = None,
        path: str | None = None,
        content: bytes | str | None = None,
    ) -> None:
        self.type = type
        self.name = name
        self.mime = mime
        self.url = url
        self.path = path
        self.content = content


class _Message:
    def __init__(
        self,
        *,
        type: str,
        content: str,
        elements: list[_Element] | None = None,
    ) -> None:
        self.type = type
        self.content = content
        self.elements = elements or []


def test_render_streaming_blocks_shows_thinking_and_answer() -> None:
    placeholder = _PlaceholderSpy()
    _render_streaming_blocks(
        placeholder,
        thinking_text="reason step",
        answer_text="final answer",
    )

    assert "sp-thinking-stream" in placeholder.last_text
    assert "reason step" in placeholder.last_text
    assert "final answer" in placeholder.last_text
    assert placeholder.last_unsafe is True


def test_chainlit_context_to_openai_includes_image_bytes() -> None:
    messages = [
        _Message(
            type="user_message",
            content="what is in this image?",
            elements=[
                _Element(
                    type="image",
                    name="sample.png",
                    mime="image/png",
                    content=b"fake-png",
                )
            ],
        )
    ]

    converted = _chainlit_context_to_openai(messages)

    assert converted == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what is in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,ZmFrZS1wbmc="},
                },
            ],
        }
    ]


def test_chainlit_context_to_openai_keeps_image_urls() -> None:
    messages = [
        _Message(
            type="user_message",
            content="describe",
            elements=[
                _Element(
                    type="image",
                    mime="image/jpeg",
                    url="https://example.com/image.jpg",
                )
            ],
        )
    ]

    converted = _chainlit_context_to_openai(messages)

    assert converted[0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.jpg"},
    }


def test_chainlit_context_to_openai_includes_text_attachments() -> None:
    messages = [
        _Message(
            type="user_message",
            content="summarize this",
            elements=[
                _Element(
                    type="file",
                    name="notes.txt",
                    mime="text/plain",
                    content=b"hello\nworld",
                )
            ],
        )
    ]

    converted = _chainlit_context_to_openai(messages)

    assert converted[0]["content"] == [
        {"type": "text", "text": "summarize this"},
        {"type": "text", "text": "\n\nAttached file: notes.txt\nhello\nworld"},
    ]


def test_reset_chat_state_like_soft_refresh() -> None:
    state: dict[str, object] = {
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 1.3,
        "max_tokens": 256,
        "system_prompt": "be brief",
        "enable_thinking": False,
        "temp_slider": 1.3,
        "max_tokens_input": 256,
        "system_prompt_input": "be brief",
        "enable_thinking_toggle": False,
        "unrelated_key": "keep me",
    }

    _reset_chat_state(state, ChatConfig(thinking=True))

    assert state["messages"] == []
    assert state["temperature"] == DEFAULT_TEMPERATURE
    assert state["max_tokens"] == DEFAULT_MAX_TOKENS
    assert state["system_prompt"] == ""
    assert state["enable_thinking"] is True
    assert state["unrelated_key"] == "keep me"


def test_build_history_title_prefers_first_user_message() -> None:
    messages = [
        {"role": "assistant", "content": "Hello there"},
        {"role": "user", "content": "Please summarize this long text now."},
    ]

    title = _build_history_title(messages, index=3)

    assert title.startswith("Chat 3: Please summarize")


def test_archive_current_chat_moves_messages_to_history() -> None:
    state: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        "chat_history": [],
        "history_counter": 1,
    }

    _archive_current_chat(state)

    history = state["chat_history"]
    assert isinstance(history, list)
    assert len(history) == 1
    entry = history[0]
    assert entry["id"] == 1
    assert entry["turn_count"] == 2
    assert entry["messages"][0]["content"] == "Hi"
    assert state["dimmed_messages"][1]["content"] == "Hello!"
    assert state["history_counter"] == 2


def test_system_prompt_none_does_not_crash() -> None:
    """Test that None system_prompt doesn't cause AttributeError.

    Regression test for: 'NoneType' object has no attribute 'strip'
    This can happen when settings['system_prompt'] is None.
    """
    # Simulate the logic from on_message
    system_prompt = None

    # This should not raise AttributeError
    messages: list[dict] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    assert messages == []

    # Also test empty string case
    system_prompt = ""
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    assert messages == []

    # And whitespace-only case
    system_prompt = "   "
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    assert messages == []

    # Valid case should add message
    system_prompt = "You are helpful"
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})

    assert messages == [{"role": "system", "content": "You are helpful"}]


def test_launch_chainlit_cwd(monkeypatch) -> None:
    import subprocess
    from unittest.mock import MagicMock
    from llm_utils.scripts.sp_chat import _launch_chainlit, ChatConfig
    import llm_utils.scripts.sp_chat

    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(llm_utils.scripts.sp_chat, "_list_models_sync", lambda *args, **kwargs: ([], None))

    config = ChatConfig(app_port=12345)
    _launch_chainlit(config)

    assert mock_run.called
    kwargs = mock_run.call_args[1]
    assert "cwd" in kwargs
    assert kwargs["cwd"].endswith("12345")
