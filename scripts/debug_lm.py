# src/llm_utils/lm/base_lm.py
from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
    cast,
)

from loguru import logger
from openai import OpenAI, AuthenticationError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Type helpers
# --------------------------------------------------------------------------- #
TModel = TypeVar("TModel", bound=BaseModel)
Messages = List[ChatCompletionMessageParam]  # convenience alias


class LM:
    """
    Unified language-model wrapper.

    * `response_format=str`  –» returns `str`
    * `response_format=YourPydanticModel` –» returns an instance of that model
    """

    # --------------------------------------------------------------------- #
    # construction / plumbing
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        model: str,
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

    # --------------------------------------------------------------------- #
    # public API – typed overloads for correct inference
    # --------------------------------------------------------------------- #
    @overload
    def __call__(
        self,
        *,
        prompt: str | None = ...,
        messages: Messages | None = ...,
        response_format: type[str] = str,
        **kwargs: Any,
    ) -> str: ...

    @overload
    def __call__(
        self,
        *,
        prompt: str | None = ...,
        messages: Messages | None = ...,
        response_format: Type[TModel],
        **kwargs: Any,
    ) -> TModel: ...

    def __call__(  # noqa: D401 – keep one consolidated docstring
        self,
        *,
        prompt: Optional[str] = None,
        messages: Optional[Messages] = None,
        response_format: Union[type[str], Type[BaseModel]] = str,
        cache: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Entry point.

        Examples
        --------
        >>> lm = LM("gpt-4o-mini")
        >>> text: str = lm(prompt="Hello")              # plain text

        >>> class Person(BaseModel): name: str; age: int
        >>> p: Person = lm(
        ...     prompt="Give me a JSON person.",
        ...     response_format=Person
        ... )                                           # Pydantic
        """
        # ---------- argument validation ---------------------------------- #
        if (prompt is None) == (messages is None):
            raise ValueError("Provide *either* `prompt` or `messages` (but not both).")

        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        # after the checks: messages is guaranteed to be not-None
        assert messages is not None
        msgs: Messages = cast(Messages, messages)

        use_cache = self.do_cache if cache is None else cache
        kw = dict(
            self.openai_kwargs,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs,
        )

        raw_msg = self._call_raw(msgs, use_cache=use_cache, **kw)
        return self._parse_output(raw_msg, response_format)

    # --------------------------------------------------------------------- #
    # internals – low-level OpenAI call with tiny on-disk cache
    # --------------------------------------------------------------------- #
    def _call_raw(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        *,
        use_cache: bool,
        **kw: Any,
    ):
        cache_key = self._cache_key(messages, kw) if use_cache else None
        if cache_key and (hit := self._load_cache(cache_key)) is not None:
            return hit

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=list(messages),  # Iterable[Any] → List for runtime
                **kw,
            )
            result = response.choices[0].message
        except (AuthenticationError, RateLimitError) as exc:  # pragma: no cover
            logger.error(exc)
            raise

        if cache_key:
            self._dump_cache(cache_key, result)

        return result

    # --------------------------------------------------------------------- #
    # parse depending on response_format
    # --------------------------------------------------------------------- #
    def _parse_output(
        self,
        message: Any,
        response_format: Union[type[str], Type[BaseModel]],
    ):
        content = getattr(message, "content", message)

        # plain-text mode
        if response_format is str:
            return str(content)

        # Pydantic mode
        if issubclass(response_format, BaseModel):
            if isinstance(content, response_format):  # already parsed
                return content
            if isinstance(content, dict):
                return response_format.model_validate(content)
            try:
                data = json.loads(content)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"Model did not return JSON:\n---\n{content}") from exc
            return response_format.model_validate(data)

        raise TypeError(
            "`response_format` must be `str` or a subclass of `pydantic.BaseModel`"
        )

    # --------------------------------------------------------------------- #
    # very small (~one-liner) on-disk cache
    # --------------------------------------------------------------------- #
    @staticmethod
    def _cache_key(messages: Any, kw: Any) -> str:
        blob = json.dumps([messages, kw], sort_keys=True).encode()
        return base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).decode()[:22]

    @staticmethod
    def _cache_path(key: str) -> str:
        return os.path.expanduser(f"~/.cache/lm/{key}.json")

    def _dump_cache(self, key: str, val: Any) -> None:
        try:
            path = self._cache_path(key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                json.dump(
                    (
                        val
                        if isinstance(val, dict)
                        else getattr(val, "__dict__", str(val))
                    ),
                    fh,
                )
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
