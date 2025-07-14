import asyncio
import functools
import inspect
import json
import os
import os.path as osp
import pickle
import uuid
from threading import Lock
from typing import Any, Awaitable, Callable, Literal, TypeVar

import cachetools
import pandas as pd
import xxhash
from loguru import logger
from pydantic import BaseModel

from speedy_utils.common.utils_io import dump_json_or_pickle, load_json_or_pickle
from speedy_utils.common.utils_misc import mkdir_or_exist

SPEED_CACHE_DIR = osp.join(osp.expanduser("~"), ".cache/speedy_cache")
LRU_MEM_CACHE = cachetools.LRUCache(maxsize=128_000)

thread_locker = Lock()

# Add two locks for thread-safe cache access
disk_lock = Lock()
mem_lock = Lock()

# Add async-specific types
T = TypeVar('T')
AsyncFunc = Callable[..., Awaitable[T]]


def compute_func_id(func, args, kwargs, ignore_self, keys):
    func_source = get_source(func)
    if keys:
        arg_spec = inspect.getfullargspec(func).args
        used_args = {arg_spec[i]: arg for i, arg in enumerate(args)}
        used_args.update(kwargs)
        values = [used_args[k] for k in keys if k in used_args]
        if not values:
            raise ValueError(f"Keys {keys} not found in function arguments")
        param_hash = identify(values)
        dir_path = f"{func.__name__}_{identify(func_source)}"
        key_id = f"{'_'.join(keys)}_{param_hash}.pkl"
        return func_source, dir_path, key_id

    if (
        inspect.getfullargspec(func).args
        and inspect.getfullargspec(func).args[0] == "self"
        and ignore_self
    ):
        fid = (func_source, args[1:], kwargs)
    else:
        fid = (func_source, args, kwargs)
    return func_source, "funcs", f"{identify(fid)}.pkl"


def fast_serialize(x: Any) -> bytes:
    try:
        return json.dumps(x, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError):
        return pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)


def identify(obj: Any, depth=0, max_depth=2) -> str:
    if isinstance(obj, (list, tuple)):
        x = [identify(x, depth + 1, max_depth) for x in obj]
        x = "\n".join(x)
        return identify(x, depth + 1, max_depth)
    # is pandas row or dict
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        x = str(obj.to_dict())
        return identify(x, depth + 1, max_depth)
    elif hasattr(obj, "__code__"):
        return identify(get_source(obj), depth + 1, max_depth)
    elif isinstance(obj, BaseModel):
        obj = obj.model_dump()
        return identify(obj, depth + 1, max_depth)
    elif isinstance(obj, dict):
        ks = sorted(obj.keys())
        vs = [identify(obj[k], depth + 1, max_depth) for k in ks]
        return identify([ks, vs], depth + 1, max_depth)
    elif obj is None:
        return identify("None", depth + 1, max_depth)
    else:
        # primitive_types = [int, float, str, bool]
        # if not type(obj) in primitive_types:
        #     logger.warning(f"Unknown type: {type(obj)}")
        return xxhash.xxh64_hexdigest(fast_serialize(obj), seed=0)


def identify_uuid(x: Any) -> str:
    data = fast_serialize(x)
    hash_obj = xxhash.xxh128(data, seed=0)
    return str(uuid.UUID(bytes=hash_obj.digest()))


def get_source(func):
    code = inspect.getsource(func)
    for r in [" ", "\n", "\t", "\r"]:
        code = code.replace(r, "")
    return code


def _disk_memoize(func, keys, cache_dir, ignore_self, verbose):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Compute cache path as before
            func_source, sub_dir, key_id = compute_func_id(
                func, args, kwargs, ignore_self, keys
            )
            if func_source is None:
                return func(*args, **kwargs)
            if sub_dir == "funcs":
                cache_path = osp.join(cache_dir, sub_dir, func.__name__, key_id)
            else:
                cache_path = osp.join(cache_dir, sub_dir, key_id)
            mkdir_or_exist(osp.dirname(cache_path))

            # First check with disk lock
            with disk_lock:
                if osp.exists(cache_path):
                    # logger.debug(f"Cache HIT for {func.__name__}, key={cache_path}")
                    try:
                        return load_json_or_pickle(cache_path)
                    except Exception as e:
                        if osp.exists(cache_path):
                            os.remove(cache_path)
                        logger.opt(depth=1).warning(
                            f"Error loading cache: {str(e)[:100]}, continue to recompute"
                        )

            result = func(*args, **kwargs)

            # Write result under disk lock to avoid race conditions
            with disk_lock:
                if not osp.exists(cache_path):
                    dump_json_or_pickle(result, cache_path)
            return result
        except Exception as e:
            logger.opt(depth=1).warning(
                f"Failed to cache {func.__name__}: {e}, continue to recompute without cache"
            )
            return func(*args, **kwargs)

    return wrapper


def _async_disk_memoize(func, keys, cache_dir, ignore_self, verbose):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            # Compute cache path as before
            func_source, sub_dir, key_id = compute_func_id(
                func, args, kwargs, ignore_self, keys
            )
            if func_source is None:
                return await func(*args, **kwargs)
            if sub_dir == "funcs":
                cache_path = osp.join(cache_dir, sub_dir, func.__name__, key_id)
            else:
                cache_path = osp.join(cache_dir, sub_dir, key_id)
            mkdir_or_exist(osp.dirname(cache_path))

            # First check with disk lock (run in thread to avoid blocking)
            def check_cache():
                with disk_lock:
                    if osp.exists(cache_path):
                        try:
                            return load_json_or_pickle(cache_path)
                        except Exception as e:
                            if osp.exists(cache_path):
                                os.remove(cache_path)
                            logger.opt(depth=1).warning(
                                f"Error loading cache: {str(e)[:100]}, continue to recompute"
                            )
                    return None
            
            # Run cache check in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            cached_result = await loop.run_in_executor(None, check_cache)
            if cached_result is not None:
                return cached_result

            result = await func(*args, **kwargs)

            # Write result under disk lock (run in thread to avoid blocking)
            def write_cache():
                with disk_lock:
                    if not osp.exists(cache_path):
                        dump_json_or_pickle(result, cache_path)
            
            await loop.run_in_executor(None, write_cache)
            return result
        except Exception as e:
            logger.opt(depth=1).warning(
                f"Failed to cache {func.__name__}: {e}, continue to recompute without cache"
            )
            return await func(*args, **kwargs)

    return wrapper


def _memory_memoize(func, size, keys, ignore_self):
    global LRU_MEM_CACHE
    if LRU_MEM_CACHE.maxsize != size:
        LRU_MEM_CACHE = cachetools.LRUCache(maxsize=size)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_source, sub_dir, key_id = compute_func_id(
            func, args, kwargs, ignore_self, keys
        )
        if func_source is None:
            return func(*args, **kwargs)
        name = identify((func_source, sub_dir, key_id))

        if not hasattr(func, "_mem_cache"):
            func._mem_cache = LRU_MEM_CACHE

        with mem_lock:
            if name in func._mem_cache:
                # logger.debug(f"Cache HIT (memory) for {func.__name__}, key={name}")
                return func._mem_cache[name]

        result = func(*args, **kwargs)

        with mem_lock:
            if name not in func._mem_cache:
                func._mem_cache[name] = result
        return result

    return wrapper


def _async_memory_memoize(func, size, keys, ignore_self):
    global LRU_MEM_CACHE
    if LRU_MEM_CACHE.maxsize != size:
        LRU_MEM_CACHE = cachetools.LRUCache(maxsize=size)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        func_source, sub_dir, key_id = compute_func_id(
            func, args, kwargs, ignore_self, keys
        )
        if func_source is None:
            return await func(*args, **kwargs)
        name = identify((func_source, sub_dir, key_id))

        if not hasattr(func, "_mem_cache"):
            func._mem_cache = LRU_MEM_CACHE

        with mem_lock:
            if name in func._mem_cache:
                # logger.debug(f"Cache HIT (memory) for {func.__name__}, key={name}")
                return func._mem_cache[name]

        result = await func(*args, **kwargs)

        with mem_lock:
            if name not in func._mem_cache:
                func._mem_cache[name] = result
        return result

    return wrapper


def both_memoize(func, keys, cache_dir, ignore_self):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_source, sub_dir, key_id = compute_func_id(
            func, args, kwargs, ignore_self, keys
        )
        if func_source is None:
            return func(*args, **kwargs)

        mem_key = identify((func_source, sub_dir, key_id))
        if not hasattr(func, "_mem_cache"):
            func._mem_cache = LRU_MEM_CACHE

        with mem_lock:
            if mem_key in func._mem_cache:
                # logger.debug(f"Cache HIT (memory) for {func.__name__}, key={mem_key}")
                return func._mem_cache[mem_key]

        if sub_dir == "funcs":
            cache_path = osp.join(cache_dir, sub_dir, func.__name__, key_id)
        else:
            cache_path = osp.join(cache_dir, sub_dir, key_id)
        mkdir_or_exist(osp.dirname(cache_path))

        with disk_lock:
            if osp.exists(cache_path):
                # logger.debug(f"Cache HIT (disk) for {func.__name__}, key={cache_path}")
                result = load_json_or_pickle(cache_path)
                with mem_lock:
                    func._mem_cache[mem_key] = result
                return result
        # logger.debug(f"Cache MISS for {func.__name__}, key={cache_path}")
        result = func(*args, **kwargs)

        with disk_lock:
            if not osp.exists(cache_path):
                dump_json_or_pickle(result, cache_path)
        with mem_lock:
            func._mem_cache[mem_key] = result
        return result

    return wrapper


def _async_both_memoize(func, keys, cache_dir, ignore_self):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        func_source, sub_dir, key_id = compute_func_id(
            func, args, kwargs, ignore_self, keys
        )
        if func_source is None:
            return await func(*args, **kwargs)

        mem_key = identify((func_source, sub_dir, key_id))
        if not hasattr(func, "_mem_cache"):
            func._mem_cache = LRU_MEM_CACHE

        with mem_lock:
            if mem_key in func._mem_cache:
                # logger.debug(f"Cache HIT (memory) for {func.__name__}, key={mem_key}")
                return func._mem_cache[mem_key]

        if sub_dir == "funcs":
            cache_path = osp.join(cache_dir, sub_dir, func.__name__, key_id)
        else:
            cache_path = osp.join(cache_dir, sub_dir, key_id)
        mkdir_or_exist(osp.dirname(cache_path))

        # Check disk cache in thread pool to avoid blocking
        def check_disk_cache():
            with disk_lock:
                if osp.exists(cache_path):
                    return load_json_or_pickle(cache_path)
            return None

        loop = asyncio.get_event_loop()
        disk_result = await loop.run_in_executor(None, check_disk_cache)
        
        if disk_result is not None:
            with mem_lock:
                func._mem_cache[mem_key] = disk_result
            return disk_result

        # logger.debug(f"Cache MISS for {func.__name__}, key={cache_path}")
        result = await func(*args, **kwargs)

        # Write to disk in thread pool to avoid blocking
        def write_disk_cache():
            with disk_lock:
                if not osp.exists(cache_path):
                    dump_json_or_pickle(result, cache_path)

        await loop.run_in_executor(None, write_disk_cache)
        
        with mem_lock:
            func._mem_cache[mem_key] = result
        return result

    return wrapper


def memoize(
    _func=None,
    *,
    keys=None,
    cache_dir=SPEED_CACHE_DIR,
    cache_type: Literal["memory", "disk", "both"] = "disk",
    size=10240,
    ignore_self=True,
    verbose=False,
):
    if "~/" in cache_dir:
        cache_dir = osp.expanduser(cache_dir)

    def decorator(func):
        # Check if function is async
        is_async = inspect.iscoroutinefunction(func)
        
        if cache_type == "memory":
            if is_async:
                return _async_memory_memoize(
                    func,
                    size,
                    keys,
                    ignore_self,
                )
            return _memory_memoize(
                func,
                size,
                keys,
                ignore_self,
            )
        elif cache_type == "disk":
            if is_async:
                return _async_disk_memoize(
                    func,
                    keys,
                    cache_dir,
                    ignore_self,
                    verbose,
                )
            return _disk_memoize(
                func,
                keys,
                cache_dir,
                ignore_self,
                verbose,
            )
        
        # cache_type == "both"
        if is_async:
            return _async_both_memoize(
                func,
                keys,
                cache_dir,
                ignore_self,
            )
        return both_memoize(
            func,
            keys,
            cache_dir,
            verbose,
        )

    # Handle both @memoize and @memoize() usage patterns
    if _func is None:
        return decorator
    else:
        return decorator(_func)


def amemoize(
    _func=None,
    *,
    keys: list[str] | None = None,
    cache_dir: str = SPEED_CACHE_DIR,
    cache_type: Literal["memory", "disk", "both"] = "disk",
    size: int = 10240,
    ignore_self: bool = True,
    verbose: bool = False,
):
    """
    Async-specific memoization decorator for coroutine functions.
    
    Args:
        _func: The async function to memoize (when used without parentheses)
        keys: Specific argument keys to use for cache key generation
        cache_dir: Directory for disk cache storage
        cache_type: Type of caching - "memory", "disk", or "both"  
        size: Size of memory cache (for memory/both types)
        ignore_self: Whether to ignore 'self' parameter in cache key
        verbose: Enable verbose logging
        
    Returns:
        Decorated async function with memoization
        
    Example:
        @amemoize(cache_type="both")
        async def my_async_func(x: int) -> str:
            return str(x)
    """
    if "~/" in cache_dir:
        cache_dir = osp.expanduser(cache_dir)

    def decorator(func):
        # Ensure the function is actually async
        if not inspect.iscoroutinefunction(func):
            raise ValueError(f"amemoize can only be used with async functions. {func.__name__} is not async.")
        
        if cache_type == "memory":
            return _async_memory_memoize(func, size, keys, ignore_self)
        elif cache_type == "disk":
            return _async_disk_memoize(func, keys, cache_dir, ignore_self, verbose)
        else:  # cache_type == "both"
            return _async_both_memoize(func, keys, cache_dir, ignore_self)

    # Handle both @amemoize and @amemoize() usage patterns
    if _func is None:
        return decorator
    else:
        return decorator(_func)


__all__ = ["memoize", "identify", "identify_uuid", "amemoize"]
