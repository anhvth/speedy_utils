from __future__ import annotations

import inspect
import os
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse

from loguru import logger


if TYPE_CHECKING:
    import httpx
    from httpx import Timeout
    from openai import AsyncOpenAI, OpenAI


_LOCAL_PROXY_VARS = (
    "http_proxy",
    "HTTP_PROXY",
)
_localhost_proxy_notice_shown = False


def _unset_proxy_env_for_localhost(base_url: Any) -> list[str]:
    """Unset proxy env vars when base_url points to localhost/loopback."""
    global _localhost_proxy_notice_shown

    if not base_url:
        return []

    base_url_str = str(base_url)
    parsed = urlparse(base_url_str)
    host = parsed.hostname
    if host not in {"localhost", "127.0.0.1", "::1"}:
        return []

    removed_vars: list[str] = []
    for var_name in _LOCAL_PROXY_VARS:
        if os.environ.pop(var_name, None) is not None:
            removed_vars.append(var_name)

    if removed_vars and not _localhost_proxy_notice_shown:
        logger.warning(
            "Localhost base_url detected ({}). Unset proxy env vars for local LLM "
            "connectivity: {}",
            base_url_str,
            ", ".join(removed_vars),
        )
        _localhost_proxy_notice_shown = True

    return removed_vars


def _get_mopenai_class():
    """Lazily create MOpenAI class to avoid importing openai at module load."""
    from openai import OpenAI

    class MOpenAI(OpenAI):
        """OpenAI client wrapper that memoizes the bound `post` method."""

        def __init__(
            self,
            *,
            api_key: str | None | Callable[[], str] = None,
            organization: str | None = None,
            project: str | None = None,
            webhook_secret: str | None = None,
            base_url: str | httpx.URL | None = None,
            websocket_base_url: str | httpx.URL | None = None,
            timeout: float | Timeout | None | Any = None,
            max_retries: int = 2,
            default_headers: Mapping[str, str] | None = None,
            default_query: Mapping[str, object] | None = None,
            http_client: httpx.Client | None = None,
            _strict_response_validation: bool = False,
            cache: bool = True,
            **kwargs: Any,
        ) -> None:
            _unset_proxy_env_for_localhost(base_url)
            super().__init__(
                api_key=api_key,
                organization=organization,
                project=project,
                webhook_secret=webhook_secret,
                base_url=base_url,
                websocket_base_url=websocket_base_url,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                default_query=default_query,
                http_client=http_client,
                _strict_response_validation=_strict_response_validation,
                **kwargs,
            )
            self._orig_post = self.post
            if cache:
                self.set_cache(cache)

        def set_cache(self, cache: bool) -> None:
            """Enable or disable caching of the post method."""
            from speedy_utils.common.utils_cache import memoize

            if cache and self.post == self._orig_post:
                self.post = memoize(self._orig_post)  # type: ignore
            elif not cache and self.post != self._orig_post:
                self.post = self._orig_post

    cast(Any, MOpenAI).__signature__ = _OPENAI_FACTORY_SIGNATURE
    return MOpenAI


def _get_masyncopenai_class():
    """Lazily create MAsyncOpenAI class to avoid importing openai at module load."""
    from openai import AsyncOpenAI

    class MAsyncOpenAI(AsyncOpenAI):
        """Async OpenAI client wrapper that memoizes the bound `post` method."""

        def __init__(
            self,
            *,
            api_key: str | Callable[[], Awaitable[str]] | None = None,
            organization: str | None = None,
            project: str | None = None,
            webhook_secret: str | None = None,
            base_url: str | httpx.URL | None = None,
            websocket_base_url: str | httpx.URL | None = None,
            timeout: float | Timeout | None | Any = None,
            max_retries: int = 2,
            default_headers: Mapping[str, str] | None = None,
            default_query: Mapping[str, object] | None = None,
            http_client: httpx.AsyncClient | None = None,
            _strict_response_validation: bool = False,
            cache: bool = True,
            **kwargs: Any,
        ) -> None:
            _unset_proxy_env_for_localhost(base_url)
            super().__init__(
                api_key=api_key,
                organization=organization,
                project=project,
                webhook_secret=webhook_secret,
                base_url=base_url,
                websocket_base_url=websocket_base_url,
                timeout=timeout,
                max_retries=max_retries,
                default_headers=default_headers,
                default_query=default_query,
                http_client=http_client,
                _strict_response_validation=_strict_response_validation,
                **kwargs,
            )
            self._orig_post = self.post
            if cache:
                self.set_cache(cache)

        def set_cache(self, cache: bool) -> None:
            """Enable or disable caching of the post method."""
            from speedy_utils.common.utils_cache import memoize

            if cache and self.post == self._orig_post:
                self.post = memoize(self._orig_post)  # type: ignore
            elif not cache and self.post != self._orig_post:
                self.post = self._orig_post

    cast(Any, MAsyncOpenAI).__signature__ = _ASYNC_OPENAI_FACTORY_SIGNATURE
    return MAsyncOpenAI


# Cache the classes so they're only created once
_MOpenAI_class = None
_MAsyncOpenAI_class = None


_OPENAI_FACTORY_SIGNATURE = inspect.Signature(
    [
        inspect.Parameter(
            "api_key",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | None | Callable[[], str]",
        ),
        inspect.Parameter(
            "organization",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | None",
        ),
        inspect.Parameter(
            "project",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | None",
        ),
        inspect.Parameter(
            "webhook_secret",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | None",
        ),
        inspect.Parameter(
            "base_url",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | httpx.URL | None",
        ),
        inspect.Parameter(
            "websocket_base_url",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | httpx.URL | None",
        ),
        inspect.Parameter(
            "timeout",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="float | Timeout | None | Any",
        ),
        inspect.Parameter(
            "max_retries",
            inspect.Parameter.KEYWORD_ONLY,
            default=2,
            annotation="int",
        ),
        inspect.Parameter(
            "default_headers",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="Mapping[str, str] | None",
        ),
        inspect.Parameter(
            "default_query",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="Mapping[str, object] | None",
        ),
        inspect.Parameter(
            "http_client",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="httpx.Client | None",
        ),
        inspect.Parameter(
            "_strict_response_validation",
            inspect.Parameter.KEYWORD_ONLY,
            default=False,
            annotation="bool",
        ),
        inspect.Parameter(
            "cache",
            inspect.Parameter.KEYWORD_ONLY,
            default=True,
            annotation="bool",
        ),
        inspect.Parameter(
            "kwargs",
            inspect.Parameter.VAR_KEYWORD,
            annotation="Any",
        ),
    ]
)

_ASYNC_OPENAI_FACTORY_SIGNATURE = inspect.Signature(
    [
        inspect.Parameter(
            "api_key",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | Callable[[], Awaitable[str]] | None",
        ),
        inspect.Parameter(
            "organization",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | None",
        ),
        inspect.Parameter(
            "project",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | None",
        ),
        inspect.Parameter(
            "webhook_secret",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | None",
        ),
        inspect.Parameter(
            "base_url",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | httpx.URL | None",
        ),
        inspect.Parameter(
            "websocket_base_url",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="str | httpx.URL | None",
        ),
        inspect.Parameter(
            "timeout",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="float | Timeout | None | Any",
        ),
        inspect.Parameter(
            "max_retries",
            inspect.Parameter.KEYWORD_ONLY,
            default=2,
            annotation="int",
        ),
        inspect.Parameter(
            "default_headers",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="Mapping[str, str] | None",
        ),
        inspect.Parameter(
            "default_query",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="Mapping[str, object] | None",
        ),
        inspect.Parameter(
            "http_client",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="httpx.AsyncClient | None",
        ),
        inspect.Parameter(
            "_strict_response_validation",
            inspect.Parameter.KEYWORD_ONLY,
            default=False,
            annotation="bool",
        ),
        inspect.Parameter(
            "cache",
            inspect.Parameter.KEYWORD_ONLY,
            default=True,
            annotation="bool",
        ),
        inspect.Parameter(
            "kwargs",
            inspect.Parameter.VAR_KEYWORD,
            annotation="Any",
        ),
    ]
)


def MOpenAI(
    *,
    api_key: str | None | Callable[[], str] = None,
    organization: str | None = None,
    project: str | None = None,
    webhook_secret: str | None = None,
    base_url: str | httpx.URL | None = None,
    websocket_base_url: str | httpx.URL | None = None,
    timeout: float | Timeout | None | Any = None,
    max_retries: int = 2,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.Client | None = None,
    _strict_response_validation: bool = False,
    cache: bool = True,
    **kwargs: Any,
):
    """Lazily construct a memoized OpenAI client with an editor-friendly signature."""
    global _MOpenAI_class
    if _MOpenAI_class is None:
        _MOpenAI_class = _get_mopenai_class()
    return _MOpenAI_class(
        api_key=api_key,
        organization=organization,
        project=project,
        webhook_secret=webhook_secret,
        base_url=base_url,
        websocket_base_url=websocket_base_url,
        timeout=timeout,
        max_retries=max_retries,
        default_headers=default_headers,
        default_query=default_query,
        http_client=http_client,
        _strict_response_validation=_strict_response_validation,
        cache=cache,
        **kwargs,
    )


def MAsyncOpenAI(
    *,
    api_key: str | Callable[[], Awaitable[str]] | None = None,
    organization: str | None = None,
    project: str | None = None,
    webhook_secret: str | None = None,
    base_url: str | httpx.URL | None = None,
    websocket_base_url: str | httpx.URL | None = None,
    timeout: float | Timeout | None | Any = None,
    max_retries: int = 2,
    default_headers: Mapping[str, str] | None = None,
    default_query: Mapping[str, object] | None = None,
    http_client: httpx.AsyncClient | None = None,
    _strict_response_validation: bool = False,
    cache: bool = True,
    **kwargs: Any,
):
    """Lazily construct a memoized AsyncOpenAI client with an explicit signature."""
    global _MAsyncOpenAI_class
    if _MAsyncOpenAI_class is None:
        _MAsyncOpenAI_class = _get_masyncopenai_class()
    return _MAsyncOpenAI_class(
        api_key=api_key,
        organization=organization,
        project=project,
        webhook_secret=webhook_secret,
        base_url=base_url,
        websocket_base_url=websocket_base_url,
        timeout=timeout,
        max_retries=max_retries,
        default_headers=default_headers,
        default_query=default_query,
        http_client=http_client,
        _strict_response_validation=_strict_response_validation,
        cache=cache,
        **kwargs,
    )

