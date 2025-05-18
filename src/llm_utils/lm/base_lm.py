from __future__ import annotations

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
from loguru import logger
from openai import OpenAI, AuthenticationError, RateLimitError
from openai.pagination import SyncPage
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
import warnings

# --------------------------------------------------------------------------- #
# type helpers
# --------------------------------------------------------------------------- #
TModel = TypeVar("TModel", bound=BaseModel)
Messages = List[ChatCompletionMessageParam]  # final, already-typed messages
LegacyMsgs = List[Dict[str, str]]  # old “…role/content…” dicts
RawMsgs = Union[Messages, LegacyMsgs]  # what __call__ accepts


class LM:
    """
    Unified language-model wrapper.

    • `response_format=str`               → returns `str`
    • `response_format=YourPydanticModel` → returns that model instance
    """

    # --------------------------------------------------------------------- #
    # ctor / plumbing
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2_000,
        host: str = "localhost",
        port: Optional[int] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: bool = True,
        **openai_kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or (f"http://{host}:{port}/v1" if port else None)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "abc")
        self.openai_kwargs = openai_kwargs
        self.do_cache = cache

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def set_model(self, model: str) -> None:
        """Set the model name after initialization."""
        self.model = model

    # --------------------------------------------------------------------- #
    # public API – typed overloads
    # --------------------------------------------------------------------- #
    @overload
    def __call__(
        self,
        *,
        prompt: str | None = ...,
        messages: RawMsgs | None = ...,
        response_format: type[str] = str,
        **kwargs: Any,
    ) -> str: ...

    @overload
    def __call__(
        self,
        *,
        prompt: str | None = ...,
        messages: RawMsgs | None = ...,
        response_format: Type[TModel],
        **kwargs: Any,
    ) -> TModel: ...

    # single implementation
    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[RawMsgs] = None,
        response_format: Union[type[str], Type[BaseModel]] = str,
        cache: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        # argument validation ------------------------------------------------
        if (prompt is None) == (messages is None):
            raise ValueError("Provide *either* `prompt` or `messages` (but not both).")

        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        assert messages is not None  # for type-checker
        assert self.model is not None, "Model must be set before calling."
        openai_msgs: Messages = (
            self._convert_messages(cast(LegacyMsgs, messages))
            if isinstance(messages[0], dict)  # legacy style
            else cast(Messages, messages)  # already typed
        )

        kw = dict(
            self.openai_kwargs,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs,
        )
        use_cache = self.do_cache if cache is None else cache

        raw = self._call_raw(
            openai_msgs,
            response_format=response_format,
            use_cache=use_cache,
            **kw,
        )
        return self._parse_output(raw, response_format)

    # --------------------------------------------------------------------- #
    # low-level OpenAI call
    # --------------------------------------------------------------------- #
    def _call_raw(
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
            # structured mode
            if response_format is not str and issubclass(response_format, BaseModel):
                rsp: ParsedChatCompletion[BaseModel] = (
                    self.client.beta.chat.completions.parse(
                        model=model,
                        messages=list(messages),
                        response_format=response_format,  # type: ignore[arg-type]
                        **kw,
                    )
                )
                result: Any = rsp.choices[0].message.parsed  # already a model
            # plain-text mode
            else:
                rsp = self.client.chat.completions.create(
                    model=model,
                    messages=list(messages),
                    **kw,
                )
                result = rsp.choices[0].message.content  # str
        except (AuthenticationError, RateLimitError) as exc:  # pragma: no cover
            logger.error(exc)
            raise

        if cache_key:
            self._dump_cache(cache_key, result)

        return result

    # --------------------------------------------------------------------- #
    # legacy → typed messages
    # --------------------------------------------------------------------- #
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
                        tool_call_id=msg.get("tool_call_id") or "",  # str, never None
                    )
                )
            else:
                # fall back to raw dict for unknown roles
                converted.append({"role": role, "content": content})  # type: ignore[arg-type]
        return converted

    # --------------------------------------------------------------------- #
    # final parse (needed for plain-text or cache hits only)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _parse_output(
        raw: Any,
        response_format: Union[type[str], Type[BaseModel]],
    ) -> str | BaseModel:
        if response_format is str:
            return cast(str, raw)

        # For the type-checker: we *know* it's a BaseModel subclass here.
        model_cls = cast(Type[BaseModel], response_format)

        if isinstance(raw, model_cls):
            return raw
        if isinstance(raw, dict):
            return model_cls.model_validate(raw)
        try:
            data = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Model did not return JSON:\n---\n{raw}") from exc
        return model_cls.model_validate(data)

    # --------------------------------------------------------------------- #
    # tiny disk cache
    # --------------------------------------------------------------------- #
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
        except Exception as exc:  # pragma: no cover
            logger.debug(f"cache write skipped: {exc}")

    def _load_cache(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:  # pragma: no cover
            return None

    @staticmethod
    def list_models(port=None) -> List[str]:
        """
        List available models.
        """
        try:
            client: OpenAI = LM(port=port).client
            base_url: URL = client.base_url
            logger.debug(f"Base URL: {base_url}")
            models: SyncPage[Model] = client.models.list()
            return [model.id for model in models.data]
        except Exception as exc:
            logger.error(f"Failed to list models: {exc}")
            return []
