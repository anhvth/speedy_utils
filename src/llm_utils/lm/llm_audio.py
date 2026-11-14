"""Native audio inputs for OpenAI-compatible chat endpoints."""

import base64
import io
import os
import wave
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlparse

from .llm import LLM


class AudioLLM(LLM):
    """LLM with native audio messages and optional ``audio=`` convenience input.

    DashScope is detected by its hostname; ``audio_provider='dashscope'`` also
    supports private gateways. Other endpoints use OpenAI input_audio payloads.
    """

    def __init__(self, *args, audio_provider=None, **kwargs):
        if audio_provider not in (None, "openai", "dashscope"):
            raise ValueError("audio_provider must be 'openai' or 'dashscope'")
        self.audio_provider = audio_provider
        super().__init__(*args, **kwargs)

    def _is_dashscope(self, client=None):
        if self.audio_provider is not None:
            return self.audio_provider == "dashscope"
        client = client or self.client
        host = urlparse(str(client.base_url)).hostname or ""
        return (
            host == "dashscope.aliyuncs.com"
            or host.startswith("dashscope-")
            and host.endswith(".aliyuncs.com")
        )

    @staticmethod
    def _encode_audio(audio, sample_rate=None, audio_format=None):
        if isinstance(audio, (str, os.PathLike)):
            path = Path(os.path.expandvars(os.fspath(audio))).expanduser()
            audio_format = path.suffix.lower().lstrip(".")
            if audio_format not in ("wav", "mp3", "flac", "ogg", "m4a", "aac"):
                raise ValueError("Unsupported audio file extension")
            data = path.read_bytes()
            if not data:
                raise ValueError("Audio file cannot be empty")
        elif isinstance(audio, (bytes, bytearray, memoryview)):
            data = bytes(audio)
            if not data:
                raise ValueError("Audio bytes cannot be empty")
            if audio_format is None and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
                audio_format = "wav"
            if audio_format not in ("wav", "mp3", "flac", "ogg", "m4a", "aac"):
                raise ValueError(
                    "Encoded audio bytes require a supported audio_format; raw PCM is unsupported"
                )
        else:
            import numpy as np

            if not isinstance(audio, np.ndarray):
                raise TypeError(
                    "audio must be a file path, encoded bytes, or numpy.ndarray"
                )
            if (
                not isinstance(sample_rate, int)
                or isinstance(sample_rate, bool)
                or sample_rate <= 0
            ):
                raise ValueError(
                    "A positive integer sample_rate is required for waveform audio"
                )
            if audio.ndim not in (1, 2) or audio.size == 0:
                raise ValueError(
                    "audio waveform must be nonempty (frames,) or (frames, channels)"
                )
            if audio.dtype.kind == "f":
                if not np.isfinite(audio).all():
                    raise ValueError("audio waveform must contain finite samples")
                pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
            elif audio.dtype == np.int16:
                pcm = audio.astype("<i2")
            else:
                raise TypeError("audio waveform must use float or int16 samples")
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1 if audio.ndim == 1 else audio.shape[1])
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(pcm.tobytes())
            data, audio_format = buffer.getvalue(), "wav"
        return {"data": base64.b64encode(data).decode("ascii"), "format": audio_format}

    def _audio_messages(
        self, input_data, audio=None, sample_rate=None, audio_format=None
    ):
        original = super()._prepare_input(input_data)
        # memoryview cannot be deep-copied; materialize only encoded byte inputs.
        memo = {}
        for message in original:
            if isinstance(message.get("content"), list):
                for part in message["content"]:
                    data = part.get("input_audio", {}).get("data")
                    if isinstance(data, memoryview):
                        memo[id(data)] = bytes(data)
        messages = deepcopy(original, memo)
        if audio is not None:
            for message in reversed(messages):
                if message["role"] == "user":
                    content = message.get("content")
                    if isinstance(content, str):
                        content = [{"type": "text", "text": content}]
                    if not isinstance(content, list):
                        raise ValueError(
                            "Audio requires a user message with text or content parts"
                        )
                    message["content"] = content + [
                        {
                            "type": "input_audio",
                            "input_audio": self._encode_audio(
                                audio, sample_rate, audio_format
                            ),
                        }
                    ]
                    break
            else:
                raise ValueError("Audio requires a user message")
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") != "input_audio":
                    continue
                payload = part.get("input_audio")
                if not isinstance(payload, dict):
                    continue
                if "path" in payload and ("data" in payload or "audio_url" in payload):
                    raise ValueError(
                        "input_audio.path cannot be combined with data or audio_url"
                    )
                if isinstance(payload.get("data"), (bytes, bytearray, memoryview)):
                    payload = self._encode_audio(
                        payload["data"], audio_format=payload.get("format")
                    )
                    part["input_audio"] = payload
                if "path" in payload:
                    payload = self._encode_audio(payload["path"])
                    part["input_audio"] = payload
        if self._is_dashscope():
            for message in messages:
                if isinstance(message.get("content"), list):
                    for part in message["content"]:
                        payload = part.get("input_audio", {})
                        data = payload.get("data")
                        if data and not data.startswith(
                            ("data:", "http://", "https://")
                        ):
                            payload["data"] = (
                                f"data:audio/{payload.get('format', 'wav')};base64,{data}"
                            )
        return messages

    def _prepare_input(self, input_data):
        return self._audio_messages(input_data)

    def _chat_completion_result(
        self, input_data, *, audio=None, sample_rate=None, audio_format=None, **kwargs
    ):
        effective_cache = kwargs.get("cache")
        if effective_cache is None:
            effective_cache = self.cache
        if self._is_dashscope() and effective_cache:
            from loguru import logger

            logger.warning(
                "AudioLLM disables caching for DashScope streaming transport"
            )
        return super()._chat_completion_result(
            self._audio_messages(input_data, audio, sample_rate, audio_format), **kwargs
        )

    def _stream_chat(
        self, input_data, *, audio=None, sample_rate=None, audio_format=None, **kwargs
    ):
        return super()._stream_chat(
            self._audio_messages(input_data, audio, sample_rate, audio_format), **kwargs
        )

    def _pydantic_completion(
        self, input_data, *, audio=None, sample_rate=None, audio_format=None, **kwargs
    ):
        if self._is_dashscope():
            raise ValueError(
                "DashScope AudioLLM does not support response_model; request JSON text and validate locally"
            )
        return super()._pydantic_completion(
            self._audio_messages(input_data, audio, sample_rate, audio_format), **kwargs
        )

    def _build_api_kwargs(self, effective_kwargs, **kwargs):
        model, api_kwargs = super()._build_api_kwargs(effective_kwargs, **kwargs)
        if self._is_dashscope():
            api_kwargs.setdefault("modalities", ["text"])
        return model, api_kwargs

    def _create_chat_completion(self, client, *, model, messages, **kwargs):
        if not self._is_dashscope(client):
            return super()._create_chat_completion(
                client, model=model, messages=messages, **kwargs
            )
        from openai.types.chat import ChatCompletion

        previous_cache = self.cache
        self._set_cache(False, client=client)
        try:
            kwargs.pop("stream", None)
            options = dict(kwargs.pop("stream_options", {}) or {})
            options["include_usage"] = True
            stream = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                stream_options=options,
                **kwargs,
            )
            chunks, reasoning, usage = [], [], None
            completion_id, created, finish_reason = "audio", 0, None
            try:
                for chunk in stream:
                    completion_id, created = chunk.id, chunk.created
                    if chunk.usage is not None:
                        usage = chunk.usage
                    for choice in chunk.choices:
                        if choice.delta.content:
                            chunks.append(choice.delta.content)
                        detail = getattr(choice.delta, "reasoning_content", None)
                        if detail:
                            reasoning.append(detail)
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason
            finally:
                stream.close()
        finally:
            self._set_cache(previous_cache, client=client)
        if finish_reason is None:
            raise ValueError("Audio stream ended without a finish reason")
        message = {"role": "assistant", "content": "".join(chunks)}
        if reasoning:
            message["reasoning"] = "".join(reasoning)
        return ChatCompletion(
            id=completion_id,
            created=created,
            model=model,
            object="chat.completion",
            choices=[{"index": 0, "finish_reason": finish_reason, "message": message}],
            usage=usage,
        )
