from __future__ import annotations

"""An **asynchronous** drop‑in replacement for the original `LM` class.

Usage example (Python ≥3.8):

    from async_lm import AsyncLM
    import asyncio

    async def main():
        lm = AsyncLM(model="gpt-4o-mini")
        reply: str = await lm(prompt="Hello, world!")
        print(reply)

    asyncio.run(main())
"""

import asyncio
import base64
import hashlib
import json
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
    cast,
)

from httpx import URL
from openai import AsyncOpenAI, AuthenticationError, RateLimitError

# from openai.pagination import AsyncSyncPage
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from openai.types.model import Model
from pydantic import BaseModel
from loguru import logger
from openai.pagination import AsyncPage as AsyncSyncPage

# --------------------------------------------------------------------------- #
# type helpers
# --------------------------------------------------------------------------- #
TModel = TypeVar("TModel", bound=BaseModel)
Messages = List[ChatCompletionMessageParam]
LegacyMsgs = List[Dict[str, str]]
RawMsgs = Union[Messages, LegacyMsgs]

# --------------------------------------------------------------------------- #
# color helpers (unchanged)
# --------------------------------------------------------------------------- #


def _color(code: int, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m"


_red = lambda t: _color(31, t)
_green = lambda t: _color(32, t)
_blue = lambda t: _color(34, t)
_yellow = lambda t: _color(33, t)


class AsyncLM:
    """Unified **async** language‑model wrapper with optional JSON parsing."""

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2_000,
        host: str = "localhost",
        port: Optional[int | str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: bool = True,
        ports: Optional[List[int]] = None,
        **openai_kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.port = port
        self.host = host
        self.base_url = base_url or (f"http://{host}:{port}/v1" if port else None)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "abc")
        self.openai_kwargs = openai_kwargs
        self.do_cache = cache
        self.ports = ports

        # Async client

    @property
    def client(self) -> AsyncOpenAI:
        # if have multiple ports
        if self.ports:
            import random
            port = random.choice(self.ports)
            api_base = f"http://{self.host}:{port}/v1"
            logger.debug(f"Using port: {port}")
        else:
            api_base = self.base_url or f"http://{self.host}:{self.port}/v1"
        client = AsyncOpenAI(
            api_key=self.api_key, base_url=api_base, **self.openai_kwargs
        )
        return client

    # ------------------------------------------------------------------ #
    # Public API – typed overloads
    # ------------------------------------------------------------------ #
    @overload
    async def __call__(
        self,
        *,
        prompt: str | None = ...,
        messages: RawMsgs | None = ...,
        response_format: type[str] = str,
        return_openai_response: bool = ...,
        **kwargs: Any,
    ) -> str: ...

    @overload
    async def __call__(
        self,
        *,
        prompt: str | None = ...,
        messages: RawMsgs | None = ...,
        response_format: Type[TModel],
        return_openai_response: bool = ...,
        **kwargs: Any,
    ) -> TModel: ...

    async def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[RawMsgs] = None,
        response_format: Union[type[str], Type[BaseModel]] = str,
        cache: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        return_openai_response: bool = False,
        **kwargs: Any,
    ):
        if (prompt is None) == (messages is None):
            raise ValueError("Provide *either* `prompt` or `messages` (but not both).")

        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        assert messages is not None
        # assert self.model is not None, "Model must be set before calling."
        if not self.model:
            models = await self.list_models(port=self.port, host=self.host)
            self.model = models[0] if models else None
            logger.info(
                f"No model specified. Using the first available model. {self.model}"
            )
        openai_msgs: Messages = (
            self._convert_messages(cast(LegacyMsgs, messages))
            if isinstance(messages[0], dict)
            else cast(Messages, messages)
        )

        kw = dict(
            self.openai_kwargs,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        kw.update(kwargs)
        use_cache = self.do_cache if cache is None else cache

        raw_response = await self._call_raw(
            openai_msgs,
            response_format=response_format,
            use_cache=use_cache,
            **kw,
        )

        if return_openai_response:
            response = raw_response
        else:
            response = self._parse_output(raw_response, response_format)

        self.last_log = [prompt, messages, raw_response]
        return response

    # ------------------------------------------------------------------ #
    # Model invocation (async)
    # ------------------------------------------------------------------ #
    async def _call_raw(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        response_format: Union[type[str], Type[BaseModel]],
        use_cache: bool,
        **kw: Any,
    ):
        assert self.model is not None, "Model must be set before making a call."
        model: str = self.model

        cache_key = (
            self._cache_key(messages, kw, response_format) if use_cache else None
        )
        if cache_key and (hit := self._load_cache(cache_key)) is not None:
            return hit

        try:
            if response_format is not str and issubclass(response_format, BaseModel):
                openai_response = await self.client.beta.chat.completions.parse(
                    model=model,
                    messages=list(messages),
                    response_format=response_format,  # type: ignore[arg-type]
                    **kw,
                )
            else:
                openai_response = await self.client.chat.completions.create(
                    model=model,
                    messages=list(messages),
                    **kw,
                )

        except (AuthenticationError, RateLimitError) as exc:
            logger.error(exc)
            raise

        if cache_key:
            self._dump_cache(cache_key, openai_response)

        return openai_response

    # ------------------------------------------------------------------ #
    # Utilities below are unchanged (sync I/O is acceptable)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _convert_messages(msgs: LegacyMsgs) -> Messages:
        converted: Messages = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                converted.append(
                    ChatCompletionUserMessageParam(role="user", content=content)
                )
            elif role == "assistant":
                converted.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant", content=content
                    )
                )
            elif role == "system":
                converted.append(
                    ChatCompletionSystemMessageParam(role="system", content=content)
                )
            elif role == "tool":
                converted.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        content=content,
                        tool_call_id=msg.get("tool_call_id") or "",
                    )
                )
            else:
                converted.append({"role": role, "content": content})  # type: ignore[arg-type]
        return converted

    @staticmethod
    def _parse_output(
        raw_response: Any, response_format: Union[type[str], Type[BaseModel]]
    ) -> str | BaseModel:
        if hasattr(raw_response, "model_dump"):
            raw_response = raw_response.model_dump()

        if response_format is str:
            if isinstance(raw_response, dict) and "choices" in raw_response:
                message = raw_response["choices"][0]["message"]
                return message.get("content", "") or ""
            return cast(str, raw_response)

        model_cls = cast(Type[BaseModel], response_format)

        if isinstance(raw_response, dict) and "choices" in raw_response:
            message = raw_response["choices"][0]["message"]
            if "parsed" in message:
                return model_cls.model_validate(message["parsed"])
            content = message.get("content")
            if content is None:
                raise ValueError("Model returned empty content")
            try:
                data = json.loads(content)
                return model_cls.model_validate(data)
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse model output as JSON:\n{content}"
                ) from exc

        if isinstance(raw_response, model_cls):
            return raw_response
        if isinstance(raw_response, dict):
            return model_cls.model_validate(raw_response)

        try:
            data = json.loads(raw_response)
            return model_cls.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"Model did not return valid JSON:\n---\n{raw_response}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Simple disk cache (sync)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cache_key(
        messages: Any, kw: Any, response_format: Union[type[str], Type[BaseModel]]
    ) -> str:
        tag = response_format.__name__ if response_format is not str else "text"
        blob = json.dumps([messages, kw, tag], sort_keys=True).encode()
        return base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).decode()[:22]

    @staticmethod
    def _cache_path(key: str) -> str:
        return os.path.expanduser(f"~/.cache/lm/{key}.json")

    def _dump_cache(self, key: str, val: Any) -> None:
        try:
            path = self._cache_path(key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                if isinstance(val, BaseModel):
                    json.dump(val.model_dump(mode="json"), fh)
                else:
                    json.dump(val, fh)
        except Exception as exc:
            logger.debug(f"cache write skipped: {exc}")

    def _load_cache(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    async def inspect_history(self) -> None:
        if not hasattr(self, "last_log"):
            raise ValueError("No history available. Please call the model first.")

        prompt, messages, response = self.last_log
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        if not messages:
            messages = [{"role": "user", "content": prompt}]

        print("\n\n")
        print(_blue("[Conversation History]") + "\n")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            print(_red(f"{role.capitalize()}:"))
            if isinstance(content, str):
                print(content.strip())
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        print(item["text"].strip())
                    elif item.get("type") == "image_url":
                        image_url = item["image_url"]["url"]
                        if "base64" in image_url:
                            len_base64 = len(image_url.split("base64,")[1])
                            print(_blue(f"<IMAGE BASE64 ENCODED({len_base64})>"))
                        else:
                            print(_blue(f"<image_url: {image_url}>"))
            print("\n")

        print(_red("Response:"))
        if isinstance(response, dict) and response.get("choices"):
            message = response["choices"][0].get("message", {})
            reasoning = message.get("reasoning_content")
            parsed = message.get("parsed")
            content = message.get("content")
            if reasoning:
                print(_yellow("<think>"))
                print(reasoning.strip())
                print(_yellow("</think>\n"))
            if parsed:
                print(
                    json.dumps(
                        (
                            parsed.model_dump()
                            if hasattr(parsed, "model_dump")
                            else parsed
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            elif content:
                print(content.strip())
            else:
                print(_green("[No content]"))
            if len(response["choices"]) > 1:
                print(
                    _blue(f"\n(Plus {len(response['choices']) - 1} other completions)")
                )
        else:
            print(_yellow("Warning: Not a standard OpenAI response object"))
            if isinstance(response, str):
                print(_green(response.strip()))
            elif isinstance(response, dict):
                print(_green(json.dumps(response, indent=2)))
            else:
                print(_green(str(response)))

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #
    def set_model(self, model: str) -> None:
        self.model = model

    @staticmethod
    async def list_models(port=None, host="localhost") -> List[str]:
        try:
            client: AsyncOpenAI = AsyncLM(port=port, host=host).client  # type: ignore[arg-type]
            base_url: URL = client.base_url
            logger.debug(f"Base URL: {base_url}")
            models: AsyncSyncPage[Model] = await client.models.list()  # type: ignore[assignment]
            return [model.id for model in models.data]
        except Exception as exc:
            logger.error(f"Failed to list models: {exc}")
            return []
