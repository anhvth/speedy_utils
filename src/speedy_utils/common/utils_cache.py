import asyncio
import functools
import inspect
import json
import os
import os.path as osp
import pickle
import uuid
import weakref
from threading import Lock
from typing import Any, Awaitable, Callable, Literal, Optional, TypeVar, overload

try:
    # Python 3.10+
    from typing import ParamSpec
except ImportError:  # pragma: no cover
    from typing_extensions import ParamSpec  # type: ignore

import cachetools
import pandas as pd
import xxhash
from loguru import logger
from pydantic import BaseModel

from speedy_utils.common.utils_io import dump_json_or_pickle, load_json_or_pickle
from speedy_utils.common.utils_misc import mkdir_or_exist

# --------------------------------------------------------------------------------------
# Defaults / Globals
# --------------------------------------------------------------------------------------

SPEED_CACHE_DIR = osp.join(osp.expanduser("~"), ".cache/speedy_cache")

# Thread locks for safety
disk_lock = Lock()
mem_lock = Lock()

# Quick identifier cache for big objects that support weakref
# (prevents recomputing expensive keys for the same object instance)
_QUICK_ID_MAP: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()

# Per-function memory caches (so different functions can have different LRU sizes)
_MEM_CACHES: "weakref.WeakKeyDictionary[Callable[..., Any], cachetools.LRUCache]" = (
    weakref.WeakKeyDictionary()
)

# Backward-compat global symbol (internal only; not exported)
LRU_MEM_CACHE = cachetools.LRUCache(maxsize=256)

# Typing helpers
P = ParamSpec("P")
R = TypeVar("R")
AsyncFunc = Callable[P, Awaitable[R]]

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def fast_serialize(x: Any) -> bytes:
    """Serialize x quickly; JSON if possible (stable), else pickle."""
    try:
        return json.dumps(x, sort_keys=True, default=str).encode("utf-8")
    except (TypeError, ValueError):
        return pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)


def identify_uuid(x: Any) -> str:
    data = fast_serialize(x)
    hash_obj = xxhash.xxh128(data, seed=0)
    return str(uuid.UUID(bytes=hash_obj.digest()))


def get_source(func: Callable[..., Any]) -> str:
    """Minified function source; falls back to module + qualname for builtins/lambdas."""
    try:
        code = inspect.getsource(func)
    except OSError:
        # source not available (e.g., builtins, some C extensions)
        mod = getattr(func, "__module__", "unknown")
        qn = getattr(func, "__qualname__", getattr(func, "__name__", "unknown"))
        code = f"{mod}.{qn}"
    # normalize whitespace to make it stable
    for r in (" ", "\n", "\t", "\r"):
        code = code.replace(r, "")
    return code


def _try_get_quick_id(obj: Any) -> Optional[str]:
    """Return a quick identifier if obj is weakref-able and cached."""
    try:
        return _QUICK_ID_MAP.get(obj)  # type: ignore[arg-type]
    except TypeError:
        # not weakref-able (e.g., list/dict); cannot use WeakKeyDictionary
        return None


def _try_store_quick_id(obj: Any, ident: str) -> None:
    """Store quick identifier if obj is weakref-able."""
    try:
        _QUICK_ID_MAP[obj] = ident  # type: ignore[index]
    except TypeError:
        # not weakref-able
        pass


def identify(obj: Any, depth: int = 0, max_depth: int = 2) -> str:
    """
    Produce a stable, content-based identifier string for arbitrary Python objects.
    Includes a quick path using a weakref cache for large, user-defined objects.
    """
    # Quick-path for user-defined objects (weakref-able)
    if depth == 0:
        quick = _try_get_quick_id(obj)
        if quick is not None:
            return quick

    if isinstance(obj, (list, tuple)):
        x = [identify(x, depth + 1, max_depth) for x in obj]
        x = "\n".join(x)
        out = identify(x, depth + 1, max_depth)
        if depth == 0:
            _try_store_quick_id(obj, out)
        return out
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        x = str(obj.to_dict())
        out = identify(x, depth + 1, max_depth)
        if depth == 0:
            _try_store_quick_id(obj, out)
        return out
    elif hasattr(obj, "__code__"):
        out = identify(get_source(obj), depth + 1, max_depth)
        if depth == 0:
            _try_store_quick_id(obj, out)
        return out
    elif isinstance(obj, BaseModel):
        out = identify(obj.model_dump(), depth + 1, max_depth)
        if depth == 0:
            _try_store_quick_id(obj, out)
        return out
    elif isinstance(obj, dict):
        ks = sorted(obj.keys())
        vs = [identify(obj[k], depth + 1, max_depth) for k in ks]
        out = identify([ks, vs], depth + 1, max_depth)
        if depth == 0:
            _try_store_quick_id(obj, out)
        return out
    elif obj is None:
        out = identify("None", depth + 1, max_depth)
        if depth == 0:
            _try_store_quick_id(obj, out)
        return out
    else:
        # primitives / everything else
        out = xxhash.xxh64_hexdigest(fast_serialize(obj), seed=0)
        if depth == 0:
            _try_store_quick_id(obj, out)
        return out


def _build_named_keys(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    keys: list[str],
) -> list[Any]:
    """Extract named parameters in order from args/kwargs for keying."""
    arg_spec = inspect.getfullargspec(func).args
    used_args = {arg_spec[i]: arg for i, arg in enumerate(args[: len(arg_spec)])}
    used_args.update(kwargs)
    values = [used_args[k] for k in keys if k in used_args]
    if not values:
        raise ValueError(f"Keys {keys} not found in function arguments")
    return values


def _compute_cache_components(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    ignore_self: bool,
    keys: Optional[list[str]],
    key_fn: Optional[Callable[..., Any]],
):
    """
    Return (func_source, sub_dir, key_id) for disk paths and memory keying.
    - If key_fn provided, it determines the cache key content.
    - Else if keys list provided, use those argument names.
    - Else use full (args, kwargs), optionally ignoring 'self' for methods.
    """
    func_source = get_source(func)

    # Custom key function (most explicit & fastest when user knows what's important)
    if key_fn is not None:
        try:
            custom_val = key_fn(*args, **kwargs)
        except Exception as e:
            raise ValueError(f"key function for {func.__name__} raised: {e}") from e
        sub_dir = "custom"
        key_id = f"{identify(custom_val)}.pkl"
        return func_source, sub_dir, key_id

    # Named keys (back-compat)
    if keys:
        values = _build_named_keys(func, args, kwargs, keys)
        param_hash = identify(values)
        dir_path = f"{func.__name__}_{identify(func_source)}"
        key_id = f"{'_'.join(keys)}_{param_hash}.pkl"
        return func_source, dir_path, key_id

    # Default: full argument identity (optionally ignoring 'self')
    if (
        inspect.getfullargspec(func).args
        and inspect.getfullargspec(func).args[0] == "self"
        and ignore_self
    ):
        fid = (func_source, args[1:], kwargs)
    else:
        fid = (func_source, args, kwargs)

    return func_source, "funcs", f"{identify(fid)}.pkl"


def _mem_cache_for(func: Callable[..., Any], size: int) -> cachetools.LRUCache:
    """Get or create a per-function LRU cache with the given size."""
    # Keep a per-function cache to avoid cross-talk of maxsize across functions
    with mem_lock:
        cache = _MEM_CACHES.get(func)
        if cache is None or cache.maxsize != size:
            cache = cachetools.LRUCache(maxsize=size)
            _MEM_CACHES[func] = cache
    # Keep global symbol backwards-compatible internally
    global LRU_MEM_CACHE
    LRU_MEM_CACHE = cache
    return cache


# --------------------------------------------------------------------------------------
# Memory-only memoize (sync / async)
# --------------------------------------------------------------------------------------


def _memory_memoize(
    func: Callable[P, R],
    size: int,
    keys: Optional[list[str]],
    ignore_self: bool,
    key_fn: Optional[Callable[..., Any]],
) -> Callable[P, R]:
    mem_cache = _mem_cache_for(func, size)

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        func_source, sub_dir, key_id = _compute_cache_components(
            func, args, kwargs, ignore_self, keys, key_fn
        )
        name = identify((func_source, sub_dir, key_id))

        with mem_lock:
            if name in mem_cache:
                return mem_cache[name]  # type: ignore[return-value]

        result = func(*args, **kwargs)

        with mem_lock:
            if name not in mem_cache:
                mem_cache[name] = result  # type: ignore[index]
        return result

    return wrapper


def _async_memory_memoize(
    func: AsyncFunc[P, R],
    size: int,
    keys: Optional[list[str]],
    ignore_self: bool,
    key_fn: Optional[Callable[..., Any]],
) -> AsyncFunc[P, R]:
    mem_cache = _mem_cache_for(func, size)

    # Avoid duplicate in-flight computations for the same key
    inflight: dict[str, asyncio.Task[R]] = {}
    alock = asyncio.Lock()

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        func_source, sub_dir, key_id = _compute_cache_components(
            func, args, kwargs, ignore_self, keys, key_fn
        )
        name = identify((func_source, sub_dir, key_id))

        async with alock:
            if name in mem_cache:
                return mem_cache[name]  # type: ignore[return-value]
            task = inflight.get(name)
            if task is None:
                task = asyncio.create_task(func(*args, **kwargs))  # type: ignore[arg-type]
                inflight[name] = task

        try:
            result = await task
        finally:
            async with alock:
                inflight.pop(name, None)

        with mem_lock:
            mem_cache[name] = result  # type: ignore[index]
        return result

    return wrapper


# --------------------------------------------------------------------------------------
# Disk-only memoize (sync / async)
# --------------------------------------------------------------------------------------


def _disk_memoize(
    func: Callable[P, R],
    keys: Optional[list[str]],
    cache_dir: str,
    ignore_self: bool,
    verbose: bool,
    key_fn: Optional[Callable[..., Any]],
) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            func_source, sub_dir, key_id = _compute_cache_components(
                func, args, kwargs, ignore_self, keys, key_fn
            )
            if sub_dir == "funcs":
                cache_path = osp.join(cache_dir, sub_dir, func.__name__, key_id)
            else:
                cache_path = osp.join(cache_dir, sub_dir, key_id)
            mkdir_or_exist(osp.dirname(cache_path))

            with disk_lock:
                if osp.exists(cache_path):
                    try:
                        return load_json_or_pickle(cache_path)
                    except Exception as e:
                        if osp.exists(cache_path):
                            os.remove(cache_path)
                        if verbose:
                            logger.opt(depth=1).warning(
                                f"Error loading cache: {str(e)[:100]}, recomputing"
                            )

            result = func(*args, **kwargs)

            with disk_lock:
                if not osp.exists(cache_path):
                    dump_json_or_pickle(result, cache_path)
            return result
        except Exception as e:
            if verbose:
                logger.opt(depth=1).warning(
                    f"Failed to cache {func.__name__}: {e}, executing without cache"
                )
            return func(*args, **kwargs)

    return wrapper


def _async_disk_memoize(
    func: AsyncFunc[P, R],
    keys: Optional[list[str]],
    cache_dir: str,
    ignore_self: bool,
    verbose: bool,
    key_fn: Optional[Callable[..., Any]],
) -> AsyncFunc[P, R]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            func_source, sub_dir, key_id = _compute_cache_components(
                func, args, kwargs, ignore_self, keys, key_fn
            )
            if sub_dir == "funcs":
                cache_path = osp.join(cache_dir, sub_dir, func.__name__, key_id)
            else:
                cache_path = osp.join(cache_dir, sub_dir, key_id)
            mkdir_or_exist(osp.dirname(cache_path))

            def check_cache() -> Optional[R]:
                with disk_lock:
                    if osp.exists(cache_path):
                        try:
                            return load_json_or_pickle(cache_path)
                        except Exception as e:
                            if osp.exists(cache_path):
                                os.remove(cache_path)
                            if verbose:
                                logger.opt(depth=1).warning(
                                    f"Error loading cache: {str(e)[:100]}, recomputing"
                                )
                    return None

            loop = asyncio.get_event_loop()
            cached_result = await loop.run_in_executor(None, check_cache)
            if cached_result is not None:
                return cached_result

            result = await func(*args, **kwargs)

            def write_cache() -> None:
                with disk_lock:
                    if not osp.exists(cache_path):
                        dump_json_or_pickle(result, cache_path)

            await loop.run_in_executor(None, write_cache)
            return result
        except Exception as e:
            if verbose:
                logger.opt(depth=1).warning(
                    f"Failed to cache {func.__name__}: {e}, executing without cache"
                )
            return await func(*args, **kwargs)

    return wrapper


# --------------------------------------------------------------------------------------
# Memory+Disk (sync / async)
# --------------------------------------------------------------------------------------


def both_memoize(
    func: Callable[P, R],
    keys: Optional[list[str]],
    cache_dir: str,
    ignore_self: bool,
    size: int,
    key_fn: Optional[Callable[..., Any]],
) -> Callable[P, R]:
    mem_cache = _mem_cache_for(func, size)

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        func_source, sub_dir, key_id = _compute_cache_components(
            func, args, kwargs, ignore_self, keys, key_fn
        )
        mem_key = identify((func_source, sub_dir, key_id))

        # Memory first
        with mem_lock:
            if mem_key in mem_cache:
                return mem_cache[mem_key]  # type: ignore[return-value]

        # Disk next
        if sub_dir == "funcs":
            cache_path = osp.join(cache_dir, sub_dir, func.__name__, key_id)
        else:
            cache_path = osp.join(cache_dir, sub_dir, key_id)
        mkdir_or_exist(osp.dirname(cache_path))

        disk_result: Optional[R] = None
        with disk_lock:
            if osp.exists(cache_path):
                try:
                    disk_result = load_json_or_pickle(cache_path)
                except Exception:
                    if osp.exists(cache_path):
                        os.remove(cache_path)
                    disk_result = None

        if disk_result is not None:
            with mem_lock:
                mem_cache[mem_key] = disk_result  # type: ignore[index]
            return disk_result

        # Miss: compute, then write both
        result = func(*args, **kwargs)
        with disk_lock:
            if not osp.exists(cache_path):
                dump_json_or_pickle(result, cache_path)
        with mem_lock:
            mem_cache[mem_key] = result  # type: ignore[index]
        return result

    return wrapper


def _async_both_memoize(
    func: AsyncFunc[P, R],
    keys: Optional[list[str]],
    cache_dir: str,
    ignore_self: bool,
    size: int,
    key_fn: Optional[Callable[..., Any]],
) -> AsyncFunc[P, R]:
    mem_cache = _mem_cache_for(func, size)

    inflight: dict[str, asyncio.Task[R]] = {}
    alock = asyncio.Lock()

    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        func_source, sub_dir, key_id = _compute_cache_components(
            func, args, kwargs, ignore_self, keys, key_fn
        )
        mem_key = identify((func_source, sub_dir, key_id))

        # Memory
        async with alock:
            if mem_key in mem_cache:
                return mem_cache[mem_key]  # type: ignore[return-value]

        # Disk
        if sub_dir == "funcs":
            cache_path = osp.join(cache_dir, sub_dir, func.__name__, key_id)
        else:
            cache_path = osp.join(cache_dir, sub_dir, key_id)
        mkdir_or_exist(osp.dirname(cache_path))

        def check_disk_cache() -> Optional[R]:
            with disk_lock:
                if osp.exists(cache_path):
                    return load_json_or_pickle(cache_path)
            return None

        loop = asyncio.get_event_loop()
        disk_result = await loop.run_in_executor(None, check_disk_cache)

        if disk_result is not None:
            with mem_lock:
                mem_cache[mem_key] = disk_result  # type: ignore[index]
            return disk_result

        # Avoid duplicate async work for same key
        async with alock:
            task = inflight.get(mem_key)
            if task is None:
                task = asyncio.create_task(func(*args, **kwargs))  # type: ignore[arg-type]
                inflight[mem_key] = task

        try:
            result = await task
        finally:
            async with alock:
                inflight.pop(mem_key, None)

        def write_disk_cache() -> None:
            with disk_lock:
                if not osp.exists(cache_path):
                    dump_json_or_pickle(result, cache_path)

        await loop.run_in_executor(None, write_disk_cache)

        with mem_lock:
            mem_cache[mem_key] = result  # type: ignore[index]
        return result

    return wrapper


# --------------------------------------------------------------------------------------
# Public decorator (only export memoize)
# --------------------------------------------------------------------------------------


@overload
def memoize(
    _func: Callable[P, R | Awaitable[R]],
    *,
    keys: Optional[list[str]] = ...,
    key: Optional[Callable[..., Any]] = ...,
    cache_dir: str = ...,
    cache_type: Literal["memory", "disk", "both"] = ...,
    size: int = ...,
    ignore_self: bool = ...,
    verbose: bool = ...,
) -> Callable[P, R | Awaitable[R]]: ...
@overload
def memoize(
    _func: None = ...,
    *,
    keys: Optional[list[str]] = ...,
    key: Optional[Callable[..., Any]] = ...,
    cache_dir: str = ...,
    cache_type: Literal["memory", "disk", "both"] = ...,
    size: int = ...,
    ignore_self: bool = ...,
    verbose: bool = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...
@overload
def memoize(
    _func: None = ...,
    *,
    keys: Optional[list[str]] = ...,
    key: Optional[Callable[..., Any]] = ...,
    cache_dir: str = ...,
    cache_type: Literal["memory", "disk", "both"] = ...,
    size: int = ...,
    ignore_self: bool = ...,
    verbose: bool = ...,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...


def memoize(
    _func: Optional[Callable[P, Any]] = None,
    *,
    keys: Optional[list[str]] = None,
    key: Optional[Callable[..., Any]] = None,
    cache_dir: str = SPEED_CACHE_DIR,
    cache_type: Literal["memory", "disk", "both"] = "both",
    size: int = 256,
    ignore_self: bool = True,
    verbose: bool = False,
):
    """
    Universal memoizer that supports sync and async functions, preserves annotations
    for Pylance via ParamSpec/TypeVar, and caches in memory + disk by default.

    - keys: list of argument names to include in key (back-compat).
    - key:  custom callable (*args, **kwargs) -> hashable/serializable object for keying.
            Prefer this for performance on big inputs (e.g., key=lambda x: x.id).
    - cache_dir: disk cache base directory (unlimited size).
    - cache_type: "memory" | "disk" | "both" (default "both").
    - size: memory LRU size per-function (default 256 items).
    - ignore_self: ignore 'self' when building the default key for bound methods.
    - verbose: enable warnings on cache load/write errors.
    """
    if "~/" in cache_dir:
        cache_dir = osp.expanduser(cache_dir)
    from speedy_utils import timef

    def decorator(func: Callable[P, Any]) -> Callable[P, Any]:
        is_async = inspect.iscoroutinefunction(func)

        # Apply timing decorator if verbose=True
        target_func = timef(func) if verbose else func

        if cache_type == "memory":
            if is_async:
                return _async_memory_memoize(target_func, size, keys, ignore_self, key)  # type: ignore[return-value]
            return _memory_memoize(target_func, size, keys, ignore_self, key)  # type: ignore[return-value]

        if cache_type == "disk":
            if is_async:
                return _async_disk_memoize(
                    target_func, keys, cache_dir, ignore_self, verbose, key
                )  # type: ignore[return-value]
            return _disk_memoize(
                target_func, keys, cache_dir, ignore_self, verbose, key
            )  # type: ignore[return-value]

        # cache_type == "both"
        if is_async:
            return _async_both_memoize(
                target_func, keys, cache_dir, ignore_self, size, key
            )  # type: ignore[return-value]
        return both_memoize(target_func, keys, cache_dir, ignore_self, size, key)  # type: ignore[return-value]

    # Support both @memoize and @memoize(...)
    if _func is None:
        return decorator
    else:
        return decorator(_func)


__all__ = ["memoize", "identify"]
