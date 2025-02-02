# utils/utils_cache.py

import functools
import inspect
import os
import os.path as osp
import pickle
import traceback
from typing import Any, Callable, Dict, List, Optional

import xxhash
from loguru import logger
import uuid

from .utils_io import dump_json_or_pickle, load_json_or_pickle
from .utils_misc import mkdir_or_exist

SPEED_CACHE_DIR = osp.join(osp.expanduser("~"), ".cache/av")
ICACHE: Dict[str, Any] = {}


def identify(x: Any) -> str:
    """Return an hex digest of the input."""
    return xxhash.xxh64(pickle.dumps(x), seed=0).hexdigest()


def identify_uuid(x: Any) -> str:
    id = identify(x)
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, id))


def memoize(
    func: Callable,
    ignore_self: bool = True,
    cache_dir: str = SPEED_CACHE_DIR,
    cache_type: str = ".pkl",
    verbose: bool = False,
    cache_key: Optional[str] = None,
) -> Callable:
    """
    also cache in memory by using an LRU or custom memoization approach.

    NOTE: This version preserves the original function's signature.
    """
    # Ensure we are dealing with supported cache types
    assert cache_type in [".pkl", ".json"]

    # If environment disables caching, just return the function as-is.
    if os.environ.get("AV_MEMOIZE_DISABLE", "0") == "1":
        print("Memoize is disabled by AV_MEMOIZE_DISABLE")
        return func

    @functools.wraps(func)  # Wrap the inner function to keep metadata
    def disk_cache_wrapper(*args, **kwargs):
        """
        Wraps the original function to add disk-based caching.
        """
        try:
            arg_names = inspect.getfullargspec(func).args
            # For hashing, also include the function's source code to catch changes
            func_source = inspect.getsource(func).replace(" ", "")

            # Allow custom cache key usage (e.g. from a kwarg), else default
            if cache_key is not None and cache_key in kwargs:
                fid = [func_source, kwargs[cache_key]]
            elif len(arg_names) > 0 and arg_names[0] == "self" and ignore_self:
                fid = (func_source, args[1:], kwargs)
            else:
                fid = (func_source, args, kwargs)

            func_id = identify(fid)

            cache_path = osp.join(
                cache_dir, "funcs", func.__name__, f"{func_id}{cache_type}"
            )
            mkdir_or_exist(os.path.dirname(cache_path))

            if osp.exists(cache_path):
                if verbose:
                    print(f"[memoize] Loading from cache: {cache_path}")
                result = load_json_or_pickle(cache_path)
            else:
                result = func(*args, **kwargs)
                dump_json_or_pickle(result, cache_path)

            return result

        except Exception as e:
            traceback.print_exc()
            print(f"[memoize] Exception: {e}, falling back to original function call.")
            return func(*args, **kwargs)

    # if on_memory:
    #     # If also caching in memory, wrap the disk_cache_wrapper with an in-memory layer
    #     # then re-wrap metadata from the *original* func:
    #     in_memory_cached = imemoize(disk_cache_wrapper)
    #     @functools.wraps(func)
    #     def final_wrapper(*args, **kwargs):
    #         return in_memory_cached(*args, **kwargs)
    #     return final_wrapper
    # else:
    #     # Otherwise, return disk_cache_wrapper directly (already @wraps(func))
    return disk_cache_wrapper


def imemoize(func: Callable) -> Callable:
    """Memoize a function into memory."""

    @functools.wraps(func)
    def _f(*args, **kwargs):
        ident_name = identify((inspect.getsource(func), args, kwargs))
        try:
            return ICACHE[ident_name]
        except KeyError:
            result = func(*args, **kwargs)
            ICACHE[ident_name] = result
            return result

    return _f


def imemoize_v2(keys: List[str]) -> Callable:
    """Memoize a function into memory based on specified keys."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            arg_names = inspect.getfullargspec(func).args
            args_dict = dict(zip(arg_names, args))
            all_args = {**args_dict, **kwargs}
            key_values = {key: all_args[key] for key in keys if key in all_args}
            if not key_values:
                return func(*args, **kwargs)

            ident_name = identify((func.__name__, tuple(sorted(key_values.items()))))
            try:
                return ICACHE[ident_name]
            except KeyError:
                result = func(*args, **kwargs)
                ICACHE[ident_name] = result
                return result

        return wrapper

    return decorator


def memoize_v2(keys: List[str], cache_dir: str = SPEED_CACHE_DIR) -> Callable:
    """Decorator to memoize function results based on specific keys."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            args_key_values = {}
            for i, arg in enumerate(args):
                arg_name = inspect.getfullargspec(func).args[i]
                args_key_values[arg_name] = arg
            args_key_values.update(kwargs)

            values = [args_key_values[key] for key in keys if key in args_key_values]
            if not values:
                return func(*args, **kwargs)

            key_id = identify(values)
            func_source = inspect.getsource(func).replace(" ", "")
            func_id = identify(func_source)
            key_names = "_".join(keys)
            cache_path = osp.join(
                cache_dir, f"{func.__name__}_{func_id}", f"{key_names}_{key_id}.pkl"
            )
            if osp.exists(cache_path):
                return load_json_or_pickle(cache_path)
            result = func(*args, **kwargs)
            dump_json_or_pickle(result, cache_path)
            return result

        return wrapper

    return decorator


def memoize_method(method):
    """
    Decorator function to memoize (cache) results of a class method.

    This decorator caches the output of the wrapped method based on its input arguments
    (both positional and keyword). If the method is called again with the same arguments,
    it returns the cached result instead of executing the method again.

    Args:
        method (Callable): The decorated method whose result will be memoized.
    """
    cache = {}

    def cached_method(cls, *args, **kwargs):
        cache_key = identify([args, kwargs])
        logger.debug("HIT" if cache_key in cache else "MISS")
        if cache_key not in cache:
            cache[cache_key] = method(cls, *args, **kwargs)
        return cache[cache_key]

    return cached_method
