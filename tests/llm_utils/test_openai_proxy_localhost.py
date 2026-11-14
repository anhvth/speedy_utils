"""A loopback endpoint must ignore proxies without mutating the process env."""

import os

from llm_utils.lm.openai_memoize import MOpenAI


def _set_proxy_env(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("https_proxy", "http://proxy.example:3128")


def test_localhost_client_ignores_proxy_env(monkeypatch):
    _set_proxy_env(monkeypatch)

    client = MOpenAI(api_key="dummy-key", base_url="http://localhost:8000/v1")
    try:
        assert client._client._trust_env is False
        assert os.environ["https_proxy"] == "http://proxy.example:3128"
        assert os.environ["http_proxy"] == "http://proxy.example:3128"
    finally:
        client.close()


def test_remote_client_keeps_proxy_env(monkeypatch):
    _set_proxy_env(monkeypatch)

    client = MOpenAI(api_key="dummy-key", base_url="https://api.openai.com/v1")
    try:
        assert client._client._trust_env is True
        assert os.environ["http_proxy"] == "http://proxy.example:3128"
    finally:
        client.close()
