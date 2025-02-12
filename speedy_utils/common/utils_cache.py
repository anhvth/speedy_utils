import functools
import inspect
import json
import os
import os.path as osp
import pickle
import uuid
from typing import Any, List, Literal
from loguru import logger
import cachetools
import xxhash

from .utils_io import dump_json_or_pickle, load_json_or_pickle
from .utils_misc import mkdir_or_exist

SPEED_CACHE_DIR = osp.join(osp.expanduser("~"), ".cache/av")
LRU_MEM_CACHE = cachetools.LRUCache(maxsize=128_000)


def fast_serialize(x: Any) -> bytes:
    try:
        return json.dumps(x, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError):
        return pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)


def identify(x: Any) -> str:
    return xxhash.xxh64_hexdigest(fast_serialize(x), seed=0)


def identify_uuid(x: Any) -> str:
    data = fast_serialize(x)
    hash_obj = xxhash.xxh128(data, seed=0)
    return str(uuid.UUID(bytes=hash_obj.digest()))


def _get_source(func):
    code = inspect.getsource(func)
    for r in [" ", "\n", "\t", "\r"]:
        code = code.replace(r, "")
    return code


def _compute_func_id(func, args, kwargs, ignore_self, cache_key, keys):
    func_source = _get_source(func)
    if keys:
        arg_spec = inspect.getfullargspec(func).args
        used_args = {arg_spec[i]: arg for i, arg in enumerate(args)}
        used_args.update(kwargs)
        values = [used_args[k] for k in keys if k in used_args]
        if not values:
            return None, None, None
        dir_path = f"{func.__name__}_{identify(func_source)}"
        key_id = f"{'_'.join(keys)}_{identify(values)}.pkl"
        return func_source, dir_path, key_id

    if cache_key and cache_key in kwargs:
        fid = [func_source, kwargs[cache_key]]
    elif (
        inspect.getfullargspec(func).args
        and inspect.getfullargspec(func).args[0] == "self"
        and ignore_self
    ):
        fid = (func_source, args[1:], kwargs)
    else:
        fid = (func_source, args, kwargs)
    return func_source, "funcs", f"{identify(fid)}.pkl"


def _disk_memoize(func, keys, cache_dir, ignore_self, verbose, cache_key):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_source, sub_dir, key_id = _compute_func_id(
            func, args, kwargs, ignore_self, cache_key, keys
        )
        if func_source is None:
            return func(*args, **kwargs)
        if sub_dir == "funcs":
            cache_path = osp.join(cache_dir, sub_dir, func.__name__, key_id)
            mkdir_or_exist(osp.dirname(cache_path))
        else:
            cache_path = osp.join(cache_dir, sub_dir, key_id)
            mkdir_or_exist(osp.dirname(cache_path))

        if osp.exists(cache_path):
            logger.debug(f"Cache HIT for {func.__name__}, key={cache_path}")
            return load_json_or_pickle(cache_path)

        result = func(*args, **kwargs)
        logger.debug(f"Cache MISS for {func.__name__}, key={cache_path}")
        dump_json_or_pickle(result, cache_path)
        return result

    return wrapper


def _memory_memoize(func, size, keys, ignore_self, cache_key):
    global LRU_MEM_CACHE
    if LRU_MEM_CACHE.maxsize != size:
        LRU_MEM_CACHE = cachetools.LRUCache(maxsize=size)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_source, sub_dir, key_id = _compute_func_id(
            func, args, kwargs, ignore_self, cache_key, keys
        )
        if func_source is None:
            return func(*args, **kwargs)
        name = identify((func_source, sub_dir, key_id))

        if not hasattr(func, "_mem_cache"):
            func._mem_cache = LRU_MEM_CACHE
        if name in func._mem_cache:
            logger.debug(f"Cache HIT (memory) for {func.__name__}, key={name}")
            return func._mem_cache[name]

        logger.debug(f"Cache MISS for {func.__name__}, key={name}")
        result = func(*args, **kwargs)
        func._mem_cache[name] = result
        return result

    return wrapper


def _both_memoize(func, keys, cache_dir, ignore_self, verbose, cache_key):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_source, sub_dir, key_id = _compute_func_id(
            func, args, kwargs, ignore_self, cache_key, keys
        )
        if func_source is None:
            return func(*args, **kwargs)

        mem_key = identify((func_source, sub_dir, key_id))
        if not hasattr(func, "_mem_cache"):
            func._mem_cache = LRU_MEM_CACHE

        if mem_key in func._mem_cache:
            logger.debug(f"Cache HIT (memory) for {func.__name__}, key={mem_key}")
            return func._mem_cache[mem_key]

        if sub_dir == "funcs":
            cache_path = osp.join(cache_dir, sub_dir, func.__name__, key_id)
            mkdir_or_exist(osp.dirname(cache_path))
        else:
            cache_path = osp.join(cache_dir, sub_dir, key_id)
            mkdir_or_exist(osp.dirname(cache_path))

        if osp.exists(cache_path):
            logger.debug(f"Cache HIT (disk) for {func.__name__}, key={cache_path}")
            result = load_json_or_pickle(cache_path)
            func._mem_cache[mem_key] = result
            return result

        logger.debug(f"Cache MISS for {func.__name__}, key={cache_path}")
        result = func(*args, **kwargs)
        dump_json_or_pickle(result, cache_path)
        func._mem_cache[mem_key] = result
        return result

    return wrapper


def memoize(
    _func=None,
    *,
    keys=None,
    cache_dir=SPEED_CACHE_DIR,
    cache_type: Literal["memory", "disk", "both"] = "both",
    size=128_000,
    ignore_self=True,
    verbose=False,
    cache_key=None,
):
    logger.info(f"cache_dir: {cache_dir}, cache_type: {cache_type}")

    def decorator(func):
        if cache_type == "memory":
            return _memory_memoize(func, size, keys, ignore_self, cache_key)
        elif cache_type == "disk":
            return _disk_memoize(func, keys, cache_dir, ignore_self, verbose, cache_key)
        return _both_memoize(func, keys, cache_dir, ignore_self, verbose, cache_key)

    if _func is None:
        return decorator
    return decorator(_func)


__all__ = ["memoize", "identify", "identify_uuid"]
