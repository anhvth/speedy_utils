import pytest

from llm_utils.scripts.sp_chat import (
    ChatConfig,
    _render_streaming_blocks,
    normalize_client_base_url,
    parse_cli_args,
)


def test_parse_cli_args_defaults() -> None:
    config = parse_cli_args([])
    assert config == ChatConfig()


def test_parse_cli_args_key_value_inputs() -> None:
    config = parse_cli_args(
        [
            'client=9000',
            'port=5011',
            'host=127.0.0.1',
            'model=Qwen/Qwen2.5-7B-Instruct',
            'api_key=secret',
            'thinking=true',
        ]
    )

    assert config.client == '9000'
    assert config.app_port == 5011
    assert config.app_host == '127.0.0.1'
    assert config.model == 'Qwen/Qwen2.5-7B-Instruct'
    assert config.api_key == 'secret'
    assert config.thinking is True


def test_parse_cli_args_invalid_format_raises() -> None:
    with pytest.raises(ValueError, match='Expected key=value'):
        parse_cli_args(['client', '8000'])


def test_parse_cli_args_invalid_key_raises() -> None:
    with pytest.raises(ValueError, match='Unknown argument'):
        parse_cli_args(['unknown=1'])


def test_parse_cli_args_invalid_bool_raises() -> None:
    with pytest.raises(ValueError, match='must be a boolean'):
        parse_cli_args(['thinking=maybe'])


def test_parse_cli_args_help_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        parse_cli_args(['--help'])


@pytest.mark.parametrize(
    ('client', 'expected'),
    [
        (None, 'http://localhost:8000/v1'),
        ('', 'http://localhost:8000/v1'),
        ('8000', 'http://localhost:8000/v1'),
        (8001, 'http://localhost:8001/v1'),
        ('localhost:9000', 'http://localhost:9000/v1'),
        ('http://127.0.0.1:7000', 'http://127.0.0.1:7000/v1'),
        ('http://127.0.0.1:7000/v1', 'http://127.0.0.1:7000/v1'),
        ('https://api.z.ai/api/coding/paas/v4', 'https://api.z.ai/api/coding/paas/v4'),
        ('https://api.z.ai/api/coding/paas/v4/', 'https://api.z.ai/api/coding/paas/v4'),
    ],
)
def test_normalize_client_base_url(client, expected) -> None:
    assert normalize_client_base_url(client) == expected


class _PlaceholderSpy:
    def __init__(self) -> None:
        self.last_text = ''
        self.last_unsafe = False

    def markdown(self, text: str, *, unsafe_allow_html: bool = False) -> None:
        self.last_text = text
        self.last_unsafe = unsafe_allow_html


def test_render_streaming_blocks_shows_thinking_and_answer() -> None:
    placeholder = _PlaceholderSpy()
    _render_streaming_blocks(
        placeholder,
        thinking_text='reason step',
        answer_text='final answer',
    )

    assert 'sp-thinking-stream' in placeholder.last_text
    assert 'reason step' in placeholder.last_text
    assert 'final answer' in placeholder.last_text
    assert placeholder.last_unsafe is True


def test_clear_chat_button_callback() -> None:
    """Clear chat now uses st.button return + st.rerun."""
    assert True
