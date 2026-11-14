"""Native audio input and provider transport contract."""

import base64
import io
import wave
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessage

from llm_utils import LLM, AudioLLM


def make_llm(provider="openai", **kwargs):
    client = MagicMock()
    client.base_url = (
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        if provider == "dashscope"
        else "http://audio-test/v1"
    )
    client.models.list.return_value = SimpleNamespace(
        data=[SimpleNamespace(id="audio-test")]
    )
    with patch("llm_utils.lm.llm.get_base_client", return_value=client):
        llm = AudioLLM(
            model="audio-test", cache=False, audio_provider=provider, **kwargs
        )
    return llm, client


def wav_bytes():
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * 32)
    return buf.getvalue()


def reply(client):
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=ChatCompletionMessage(role="assistant", content="heard")
            )
        ],
        usage=None,
    )


def test_public_subclass_and_path_messages_preserve_history(tmp_path):
    llm, client = make_llm()
    assert isinstance(llm, LLM)
    path = tmp_path / "audio.wav"
    path.write_bytes(wav_bytes())
    messages = [
        {"role": "system", "content": "listen"},
        {"role": "user", "content": "previous"},
        {"role": "assistant", "content": "ok"},
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"path": str(path)}},
                {"type": "text", "text": "question"},
            ],
        },
    ]
    original = deepcopy(messages)
    reply(client)
    assert llm(messages).content == "heard"
    sent = client.chat.completions.create.call_args.kwargs["messages"]
    assert sent[:3] == messages[:3]
    payload = sent[-1]["content"][0]["input_audio"]
    assert base64.b64decode(payload["data"]) == path.read_bytes()
    assert "path" not in payload
    assert messages == original


@pytest.mark.parametrize(
    "kind", ["path", "bytes", "bytearray", "memoryview", "waveform"]
)
def test_audio_convenience_encodes_wav(tmp_path, kind):
    llm, client = make_llm()
    path = tmp_path / "input.wav"
    path.write_bytes(wav_bytes())
    value = {
        "path": path,
        "bytes": wav_bytes(),
        "bytearray": bytearray(wav_bytes()),
        "memoryview": memoryview(wav_bytes()),
        "waveform": np.zeros(32, dtype=np.float32),
    }[kind]
    reply(client)
    llm(
        "question",
        audio=value,
        **({"sample_rate": 16000} if kind == "waveform" else {}),
    )
    payload = client.chat.completions.create.call_args.kwargs["messages"][-1][
        "content"
    ][-1]["input_audio"]
    with wave.open(io.BytesIO(base64.b64decode(payload["data"]))) as wav:
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 32


@pytest.mark.parametrize(
    "audio,kwargs,error",
    [
        (np.zeros(4), {}, "sample_rate"),
        (b"", {}, "empty"),
        (b"not pcm", {}, "audio_format"),
        (np.array([float("nan")]), {"sample_rate": 16000}, "finite"),
    ],
)
def test_invalid_audio_fails_before_request(audio, kwargs, error):
    llm, client = make_llm()
    with pytest.raises(Exception, match=error):
        llm("question", audio=audio, **kwargs)
    client.chat.completions.create.assert_not_called()


def test_message_bytes_and_preencoded_data_do_not_mutate():
    llm, _ = make_llm()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": wav_bytes(), "format": "wav"},
                }
            ],
        }
    ]
    converted = llm._prepare_input(messages)
    assert (
        base64.b64decode(converted[0]["content"][0]["input_audio"]["data"])
        == wav_bytes()
    )
    assert isinstance(messages[0]["content"][0]["input_audio"]["data"], bytes)
    assert llm._prepare_input(converted) == converted


def chunks():
    return [
        ChatCompletionChunk(
            id="id",
            created=1,
            model="audio-test",
            object="chat.completion.chunk",
            choices=[{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
        )
        for text, finish in [("hel", None), ("lo", "stop")]
    ] + [
        ChatCompletionChunk(
            id="id",
            created=1,
            model="audio-test",
            object="chat.completion.chunk",
            choices=[],
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )
    ]


def test_dashscope_nonstream_aggregates_and_preserves_usage():
    llm, client = make_llm("dashscope")
    stream = MagicMock()
    stream.__iter__.return_value = iter(chunks())
    client.chat.completions.create.return_value = stream
    result = llm("question", audio=wav_bytes(), return_dict=True)
    assert isinstance(result["message"], ChatCompletionMessage)
    assert result["message"].content == "hello"
    assert result["completion"].usage.total_tokens == 12
    assert llm.last_usage["total_tokens"] == 12
    sent = client.chat.completions.create.call_args.kwargs
    assert sent["stream"] is True
    assert sent["modalities"] == ["text"]
    assert sent["stream_options"] == {"include_usage": True}
    assert sent["messages"][0]["content"][-1]["input_audio"]["data"].startswith(
        "data:audio/wav;base64,"
    )
    stream.close.assert_called_once()


def test_explicit_stream_preserves_llm_chunks_and_audio_payload():
    llm, client = make_llm("dashscope")
    client.chat.completions.create.return_value = iter(chunks())
    assert (
        "".join(
            chunk.choices[0].delta.content
            for chunk in llm("q", audio=wav_bytes(), stream=True)
            if chunk.choices
        )
        == "hello"
    )
    assert client.chat.completions.create.call_args.kwargs["messages"][0]["content"][
        -1
    ]["input_audio"]["data"].startswith("data:audio/wav;base64,")


def test_structured_audio_uses_native_parse():
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    llm, client = make_llm()
    client.chat.completions.parse.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"answer":"yes"}', parsed=Answer(answer="yes")
                )
            )
        ]
    )
    assert llm("q", audio=wav_bytes(), response_model=Answer).answer == "yes"
    assert (
        client.chat.completions.parse.call_args.kwargs["messages"][0]["content"][-1][
            "type"
        ]
        == "input_audio"
    )


def test_environment_path_expansion_and_empty_files(tmp_path, monkeypatch):
    llm, client = make_llm()
    monkeypatch.setenv("AUDIO_TEST_ROOT", str(tmp_path))
    (tmp_path / "valid.wav").write_bytes(wav_bytes())
    reply(client)
    assert llm("q", audio="${AUDIO_TEST_ROOT}/valid.wav").content == "heard"
    (tmp_path / "empty.wav").write_bytes(b"")
    client.chat.completions.create.reset_mock()
    with pytest.raises(Exception, match="empty"):
        llm("q", audio=tmp_path / "empty.wav")
    client.chat.completions.create.assert_not_called()


def test_dashscope_structured_output_fails_explicitly():
    from pydantic import BaseModel

    class Answer(BaseModel):
        answer: str

    llm, client = make_llm("dashscope")
    with pytest.raises(Exception, match="does not support response_model"):
        llm("q", audio=wav_bytes(), response_model=Answer)
    client.chat.completions.create.assert_not_called()
    client.chat.completions.parse.assert_not_called()


def test_truncated_dashscope_stream_fails_and_closes():
    llm, client = make_llm("dashscope")
    stream = MagicMock()
    stream.__iter__.return_value = iter(chunks()[:1])
    client.chat.completions.create.return_value = stream
    with pytest.raises(Exception, match="without a finish reason"):
        llm("q", audio=wav_bytes())
    stream.close.assert_called_once()


def test_url_and_preencoded_message_parts_pass_through_without_mutation():
    llm, _ = make_llm("dashscope")
    original = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "https://example.com/audio.wav",
                        "format": "wav",
                    },
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": "data:audio/wav;base64,AAAA",
                        "format": "wav",
                    },
                },
            ],
        }
    ]
    assert llm._prepare_input(original) == original


def test_explicit_bytes_format_is_preserved():
    llm, client = make_llm()
    reply(client)
    llm("q", audio=b"encoded-mp3", audio_format="mp3")
    payload = client.chat.completions.create.call_args.kwargs["messages"][0]["content"][
        -1
    ]["input_audio"]
    assert payload["format"] == "mp3"
    assert base64.b64decode(payload["data"]) == b"encoded-mp3"


def test_message_memoryview_is_encoded_without_mutating_input():
    llm, _ = make_llm()
    view = memoryview(wav_bytes())
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": view, "format": "wav"}}
            ],
        }
    ]
    result = llm._prepare_input(messages)
    assert (
        base64.b64decode(result[0]["content"][0]["input_audio"]["data"]) == wav_bytes()
    )
    assert messages[0]["content"][0]["input_audio"]["data"] is view


def test_conflicting_path_and_bytes_rejected_before_request():
    llm, client = make_llm()
    with pytest.raises(Exception, match="cannot be combined"):
        llm(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "path": "/unused.wav",
                                "data": wav_bytes(),
                                "format": "wav",
                            },
                        }
                    ],
                }
            ]
        )
    client.chat.completions.create.assert_not_called()
