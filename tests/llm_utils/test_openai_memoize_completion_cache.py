"""Regression tests for memoized completion responses."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import httpx

from llm_utils.lm.llm import LLM
from llm_utils.lm.openai_memoize import MOpenAI


def _make_models_list_payload() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": "test-model",
                "object": "model",
                "created": 0,
                "owned_by": "openai",
            }
        ],
    }


def _make_completion_payload() -> dict:
    return {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 1710000000,
        "model": "test-model",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "text": "True",
                "logprobs": {
                    "text_offset": [0, 4],
                    "token_logprobs": [-0.11, -0.22],
                    "tokens": ["True", "False"],
                    "top_logprobs": [{"True": -0.11}, {"False": -0.22}],
                    "extra_old_field": 123,
                },
                "prompt_logprobs": [{"True": -0.33}],
                "token_ids": [11, 22],
                "prompt_token_ids": [33, 44],
                "extra_choice_field": "vllm-extra",
            }
        ],
        "usage": {
            "completion_tokens": 7,
            "prompt_tokens": 11,
            "total_tokens": 18,
        },
    }


def test_generate_uses_cached_mopenai_completion_and_preserves_logprobs():
    counts = {"models": 0, "completions": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.endswith("/models"):
            counts["models"] += 1
            return httpx.Response(200, json=_make_models_list_payload())

        if request.method == "POST" and request.url.path.endswith("/completions"):
            counts["completions"] += 1
            return httpx.Response(200, json=_make_completion_payload())

        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = MOpenAI(
        api_key="dummy-key",
        base_url="http://test/v1",
        http_client=http_client,
        cache=True,
    )

    try:
        with patch("llm_utils.lm.llm.get_base_client", return_value=client):
            llm = LLM(model="test-model")
            prompt = f"prompt-{uuid.uuid4().hex}"
            first = llm.generate(
                prompt,
                max_tokens=0,
                echo=True,
                logprobs=1,
                temperature=0,
            )
            second = llm.generate(
                prompt,
                max_tokens=0,
                echo=True,
                logprobs=1,
                temperature=0,
            )
    finally:
        client.close()

    assert counts["models"] == 1
    assert counts["completions"] == 1

    for choice in (first, second):
        assert choice.text == "True"
        assert choice.logprobs is not None
        assert choice.logprobs.token_logprobs == [-0.11, -0.22]
        assert choice.logprobs.tokens == ["True", "False"]
        assert choice.logprobs.top_logprobs == [{"True": -0.11}, {"False": -0.22}]
        assert choice.logprobs.extra_old_field == 123  # type: ignore[attr-defined]
        assert choice.prompt_logprobs == [{"True": -0.33}]  # type: ignore[attr-defined]
        assert choice.token_ids == [11, 22]  # type: ignore[attr-defined]
        assert choice.prompt_token_ids == [33, 44]  # type: ignore[attr-defined]
        assert choice.extra_choice_field == "vllm-extra"  # type: ignore[attr-defined]
