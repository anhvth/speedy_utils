from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import BadRequestError

from llm_utils import LLM


def bad_request(body):
    return BadRequestError(
        str(body),
        response=httpx.Response(400, request=httpx.Request("POST", "http://test/v1/chat/completions")),
        body=body,
    )


@pytest.mark.parametrize("body", [
    {"error": {"message": "Unexpected reasoning effort high. Supported types are xhigh (default), medium, and low.", "code": 400}},
    {"message": "Invalid JSON schema", "code": 400},
    "Unsupported parameter",
])
def test_chat_validation_error_is_sent_once_without_sleep(body):
    client = MagicMock()
    client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="Qwen3.8-27B")])
    client.chat.completions.create.side_effect = bad_request(body)
    with patch("llm_utils.lm.llm.get_base_client", return_value=client), patch("llm_utils.lm.llm.time.sleep") as sleep:
        llm = LLM(model="Qwen3.8-27B", cache=False)
        with pytest.raises(Exception, match="BadRequestError"):
            llm.chat_completion("Hello", reasoning_effort="high")
    assert client.chat.completions.create.call_count == 1
    sleep.assert_not_called()
    assert client.chat.completions.create.call_args.kwargs["reasoning_effort"] == "high"


@pytest.mark.parametrize("body", [
    "Invalid HTTP request received.",
    {"error": {"message": "Invalid HTTP request received.", "code": 400}},
])
def test_vllm_transient_bad_request_still_retries(body):
    client = MagicMock()
    client.models.list.return_value = SimpleNamespace(data=[SimpleNamespace(id="test-model")])
    message = SimpleNamespace(role="assistant", content="ok")
    client.chat.completions.create.side_effect = [
        bad_request(body), SimpleNamespace(choices=[SimpleNamespace(message=message)]),
    ]
    with patch("llm_utils.lm.llm.get_base_client", return_value=client), patch("llm_utils.lm.llm.time.sleep") as sleep:
        result = LLM(model="test-model", cache=False).chat_completion("Hello")
    assert result.content == "ok"
    assert client.chat.completions.create.call_count == 2
    sleep.assert_called_once()
