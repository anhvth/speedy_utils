import pytest

from llm_utils.scripts.sp_chat import (
    ChatConfig,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    _archive_current_chat,
    _build_history_title,
    _render_streaming_blocks,
    _reset_chat_state,
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


def test_reset_chat_state_like_soft_refresh() -> None:
    state: dict[str, object] = {
        'messages': [{'role': 'user', 'content': 'hello'}],
        'temperature': 1.3,
        'max_tokens': 256,
        'system_prompt': 'be brief',
        'enable_thinking': False,
        'temp_slider': 1.3,
        'max_tokens_input': 256,
        'system_prompt_input': 'be brief',
        'enable_thinking_toggle': False,
        'unrelated_key': 'keep me',
    }

    _reset_chat_state(state, ChatConfig(thinking=True))

    assert state['messages'] == []
    assert state['temperature'] == DEFAULT_TEMPERATURE
    assert state['max_tokens'] == DEFAULT_MAX_TOKENS
    assert state['system_prompt'] == ''
    assert state['enable_thinking'] is True
    assert state['unrelated_key'] == 'keep me'


def test_build_history_title_prefers_first_user_message() -> None:
    messages = [
        {'role': 'assistant', 'content': 'Hello there'},
        {'role': 'user', 'content': 'Please summarize this long text now.'},
    ]

    title = _build_history_title(messages, index=3)

    assert title.startswith('Chat 3: Please summarize')


def test_archive_current_chat_moves_messages_to_history() -> None:
    state: dict[str, object] = {
        'messages': [
            {'role': 'user', 'content': 'Hi'},
            {'role': 'assistant', 'content': 'Hello!'},
        ],
        'chat_history': [],
        'history_counter': 1,
    }

    _archive_current_chat(state)

    history = state['chat_history']
    assert isinstance(history, list)
    assert len(history) == 1
    entry = history[0]
    assert entry['id'] == 1
    assert entry['turn_count'] == 2
    assert entry['messages'][0]['content'] == 'Hi'
    assert state['dimmed_messages'][1]['content'] == 'Hello!'
    assert state['history_counter'] == 2
