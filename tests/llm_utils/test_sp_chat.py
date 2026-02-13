import pytest

from llm_utils.scripts.sp_chat import (
    ChatConfig,
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
        ]
    )

    assert config.client == '9000'
    assert config.app_port == 5011
    assert config.app_host == '127.0.0.1'
    assert config.model == 'Qwen/Qwen2.5-7B-Instruct'
    assert config.api_key == 'secret'


def test_parse_cli_args_invalid_format_raises() -> None:
    with pytest.raises(ValueError, match='Expected key=value'):
        parse_cli_args(['client', '8000'])


def test_parse_cli_args_invalid_key_raises() -> None:
    with pytest.raises(ValueError, match='Unknown argument'):
        parse_cli_args(['unknown=1'])


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
    ],
)
def test_normalize_client_base_url(client, expected) -> None:
    assert normalize_client_base_url(client) == expected
