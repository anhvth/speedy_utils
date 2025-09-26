from openai import OpenAI, AsyncOpenAI
from typing import Any, Callable

from speedy_utils.common.utils_cache import memoize


class MOpenAI(OpenAI):
    """
    MOpenAI(*args, **kwargs)

    Subclass of OpenAI that transparently memoizes the instance's `post` method.

    This class forwards all constructor arguments to the OpenAI base class and then
    replaces the instance's `post` method with a memoized wrapper:

    Behavior
    - The memoized `post` caches responses based on the arguments with which it is
        invoked, preventing repeated identical requests from invoking the underlying
        OpenAI API repeatedly.
    - Because `post` is replaced on the instance, the cache is by-default tied to
        the MOpenAI instance (per-instance cache).
    - Any initialization arguments are passed unchanged to OpenAI.__init__.

    Notes and cautions
    - The exact semantics of caching (cache key construction, expiry, max size,
        persistence) depend on the implementation of `memoize`. Ensure that the
        provided `memoize` supports the desired behavior (e.g., hashing of mutable
        inputs, thread-safety, TTL, cache invalidation).
    - If the original `post` method has important side effects or relies on
        non-deterministic behavior, memoization may change program behavior.
    - If you need a shared cache across instances, or more advanced cache controls,
        modify `memoize` or wrap at a class/static level instead of assigning to the
        bound method.
    - Type information is now fully preserved by the memoize decorator, eliminating
        the need for type casting.

    Example
            m = MOpenAI(api_key="...", model="gpt-4")
            r1 = m.post("Hello")         # executes API call and caches result
            r2 = m.post("Hello")         # returns cached result (no API call)
    """

    def __init__(self, *args, cache=True, **kwargs):
        super().__init__(*args, **kwargs)
        if cache:
            # Create a memoized wrapper for the instance's post method.
            # The memoize decorator now preserves exact type information,
            # so no casting is needed.
            orig_post = self.post
            memoized = memoize(orig_post)
            self.post = memoized


class MAsyncOpenAI(AsyncOpenAI):
    """
    MAsyncOpenAI(*args, **kwargs)

    Async subclass of AsyncOpenAI that transparently memoizes the instance's `post` method.

    This class forwards all constructor arguments to the AsyncOpenAI base class and then
    replaces the instance's `post` method with a memoized wrapper:

    Behavior
    - The memoized `post` caches responses based on the arguments with which it is
        invoked, preventing repeated identical requests from invoking the underlying
        OpenAI API repeatedly.
    - Because `post` is replaced on the instance, the cache is by-default tied to
        the MAsyncOpenAI instance (per-instance cache).
    - Any initialization arguments are passed unchanged to AsyncOpenAI.__init__.

    Example
            m = MAsyncOpenAI(api_key="...", model="gpt-4")
            r1 = await m.post("Hello")    # executes API call and caches result
            r2 = await m.post("Hello")    # returns cached result (no API call)
    """

    def __init__(self, *args, cache=True, **kwargs):
        super().__init__(*args, **kwargs)
        if cache:
            self.post = memoize(self.post) # type: ignore
