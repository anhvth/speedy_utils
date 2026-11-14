from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from loguru import logger


if TYPE_CHECKING:
    import httpx
    from httpx import Timeout
    from openai import AsyncOpenAI, OpenAI


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


__all__ = ["MOpenAI", "MAsyncOpenAI"]


def _is_loopback_url(base_url: Any) -> bool:
    """True when base_url points at the local machine."""
    return bool(base_url) and urlparse(str(base_url)).hostname in _LOOPBACK_HOSTS


def _loopback_http_client(is_async: bool):
    """Build a client that ignores proxy env vars, for loopback endpoints."""
    import httpx

    cls = httpx.AsyncClient if is_async else httpx.Client
    return cls(trust_env=False)


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
            if http_client is None and _is_loopback_url(base_url):
                http_client = _loopback_http_client(is_async=False)
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
            self._install_cached_post(cache)

        def _install_cached_post(self, cache: bool) -> None:
            """Wrap ``post`` once; the flag decides per call, so cached resource
            handles (``chat.completions._post``) never go stale."""
            from speedy_utils.common.utils_cache import memoize

            self._cache_enabled = cache
            memoized_post = memoize(self._orig_post)

            @functools.wraps(self._orig_post)
            def post(*args, **kwargs):
                if self._cache_enabled:
                    return memoized_post(*args, **kwargs)
                return self._orig_post(*args, **kwargs)

            self.post = post

        def set_cache(self, cache: bool) -> None:
            """Enable or disable caching of the post method."""
            self._cache_enabled = cache

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
            if http_client is None and _is_loopback_url(base_url):
                http_client = _loopback_http_client(is_async=True)
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
            self._install_cached_post(cache)

        def _install_cached_post(self, cache: bool) -> None:
            """Wrap ``post`` once; the flag decides per call, so cached resource
            handles (``chat.completions._post``) never go stale."""
            from speedy_utils.common.utils_cache import memoize

            self._cache_enabled = cache
            memoized_post = memoize(self._orig_post)

            @functools.wraps(self._orig_post)
            async def post(*args, **kwargs):
                if self._cache_enabled:
                    return await memoized_post(*args, **kwargs)
                return await self._orig_post(*args, **kwargs)

            self.post = post

        def set_cache(self, cache: bool) -> None:
            """Enable or disable caching of the post method."""
            self._cache_enabled = cache

    return MAsyncOpenAI


# Cache the classes so they're only created once
_MOpenAI_class = None
_MAsyncOpenAI_class = None


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

