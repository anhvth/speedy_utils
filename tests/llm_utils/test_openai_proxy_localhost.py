import importlib


def test_unset_proxy_env_for_localhost(monkeypatch):
    mod = importlib.import_module("llm_utils.lm.openai_memoize")

    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("https_proxy", "http://proxy.example:3128")

    removed = mod._unset_proxy_env_for_localhost("http://localhost:8000/v1")

    assert set(removed) == {
        "http_proxy",
        "HTTP_PROXY",
    }
    for env_name in removed:
        assert env_name not in mod.os.environ
    assert mod.os.environ["https_proxy"] == "http://proxy.example:3128"


def test_keep_proxy_env_for_non_localhost(monkeypatch):
    mod = importlib.import_module("llm_utils.lm.openai_memoize")
    monkeypatch.setenv("http_proxy", "http://proxy.example:3128")

    removed = mod._unset_proxy_env_for_localhost("https://api.openai.com/v1")

    assert removed == []
    assert mod.os.environ["http_proxy"] == "http://proxy.example:3128"
