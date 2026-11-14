from ..__imports import *
from .utils_io import dump_json_or_pickle, load_json_or_pickle
from .utils_misc import mkdir_or_exist


# --------------------------------------------------------------------------------------
# Defaults / Globals
# --------------------------------------------------------------------------------------

SPEED_CACHE_DIR = osp.expanduser('~/.cache/speedy_cache')

# Thread locks for safety
disk_lock = Lock()
mem_lock = Lock()

# Per-function memory caches (so different functions can have different LRU sizes)
_MEM_CACHES: 'weakref.WeakKeyDictionary[Callable[..., Any], cachetools.LRUCache]' = (
    weakref.WeakKeyDictionary()
)

# Global memory cache for imemoize (persists across IPython reloads)
_GLOBAL_MEMORY_CACHE: dict[str, Any] = {}

# Typing helpers
P = ParamSpec('P')
R = TypeVar('R')
AsyncFunc = Callable[P, Awaitable[R]]

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


def fast_serialize(x: Any) -> bytes:
    """Serialize x quickly; JSON if possible (stable), else pickle."""
    try:
        return json.dumps(x, sort_keys=True, default=str).encode('utf-8')
    except (TypeError, ValueError):
        return pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)


def get_source(func: Callable[..., Any]) -> str:
    """Minified function source; falls back to module + qualname for builtins/lambdas."""
    func = inspect.unwrap(func)
    try:
        code = inspect.getsource(func)
    except OSError:
        # source not available (e.g., builtins, some C extensions)
        mod = getattr(func, '__module__', 'unknown')
        qn = getattr(func, '__qualname__', getattr(func, '__name__', 'unknown'))
        code = f'{mod}.{qn}'
    # normalize whitespace to make it stable
    for r in (' ', '\n', '\t', '\r'):
        code = code.replace(r, '')
    return code


def identify(obj: Any) -> str:
    """Stable, content-based identifier for arbitrary Python objects."""
    from pydantic import BaseModel

    if isinstance(obj, (list, tuple)):
        x = '\n'.join(identify(x) for x in obj)
        return identify(x)
    # pandas is an optional extra: match by module name, never touch the proxy
    if type(obj).__module__.split('.')[0] == 'pandas':
        return identify(str(obj.to_dict()))
    if hasattr(obj, '__code__'):
        return identify(get_source(obj))
    if isinstance(obj, BaseModel):
        # Use hasattr for type checker compatibility
        model_data = (
            obj.model_dump()  # type: ignore
            if hasattr(obj, 'model_dump')
            else obj.dict()
        )
        return identify(model_data)
    if isinstance(obj, dict):
        ks = sorted(obj.keys(), key=lambda k: (isinstance(k, str), str(k)))
        return identify([ks, [identify(obj[k]) for k in ks]])
    if obj is None:
        return identify('None')
    # primitives / everything else
    return hashlib.blake2b(fast_serialize(obj), digest_size=8).hexdigest()


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
        raise ValueError(f'Keys {keys} not found in function arguments')
    return values


def _compute_cache_components(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    ignore_self: bool,
    keys: list[str] | None,
    key_fn: Callable[..., Any] | None,
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
            raise ValueError(f'key function for {func.__name__} raised: {e}') from e
        return func_source, 'custom', f'{identify(custom_val)}.pkl'

    # Named keys (back-compat)
    if keys:
        values = _build_named_keys(func, args, kwargs, keys)
        sub_dir = f'{func.__name__}_{identify(func_source)}'
        return func_source, sub_dir, f'{"_".join(keys)}_{identify(values)}.pkl'

    # Default: full argument identity (optionally ignoring 'self')
    if (
        inspect.getfullargspec(func).args
        and inspect.getfullargspec(func).args[0] == 'self'
        and ignore_self
    ):
        fid = (func_source, args[1:], kwargs)
    else:
        fid = (func_source, args, kwargs)

    # 'funcs' keys nest under the function name so two functions never share a file
    return func_source, osp.join('funcs', func.__name__), f'{identify(fid)}.pkl'


def _mem_cache_for(func: Callable[..., Any], size: int) -> cachetools.LRUCache:
    """Get or create a per-function LRU cache with the given size."""
    # Keep a per-function cache to avoid cross-talk of maxsize across functions
    with mem_lock:
        cache = _MEM_CACHES.get(func)
        if cache is None or cache.maxsize != size:
            cache = cachetools.LRUCache(maxsize=size)
            _MEM_CACHES[func] = cache
    return cache


# --------------------------------------------------------------------------------------
# Memory-only memoize (sync / async)
# --------------------------------------------------------------------------------------


def _memory_memoize(
    func: Callable[P, R],
    size: int,
    keys: list[str] | None,
    ignore_self: bool,
    key_fn: Callable[..., Any] | None,
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
                return mem_cache[name]

        result = func(*args, **kwargs)

        with mem_lock:
            if name not in mem_cache:
                mem_cache[name] = result
        return result

    return wrapper


def _async_memory_memoize(
    func: AsyncFunc[P, R],
    size: int,
    keys: list[str] | None,
    ignore_self: bool,
    key_fn: Callable[..., Any] | None,
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
                return mem_cache[name]
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
            mem_cache[name] = result
        return result

    return wrapper


# --------------------------------------------------------------------------------------
# Disk-only memoize (sync / async)
# --------------------------------------------------------------------------------------


def _disk_memoize(
    func: Callable[P, R],
    keys: list[str] | None,
    cache_dir: str,
    ignore_self: bool,
    verbose: bool,
    key_fn: Callable[..., Any] | None,
) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            func_source, sub_dir, key_id = _compute_cache_components(
                func, args, kwargs, ignore_self, keys, key_fn
            )
            cache_path = osp.join(cache_dir, sub_dir, key_id)
            mkdir_or_exist(osp.dirname(cache_path))
        except Exception as e:
            if verbose:
                logger.opt(depth=1).warning(
                    f'Failed to cache {func.__name__}: {e}, executing without cache'
                )
            return func(*args, **kwargs)

        with disk_lock:
            if osp.exists(cache_path):
                try:
                    return load_json_or_pickle(cache_path)
                except Exception as e:
                    if osp.exists(cache_path):
                        os.remove(cache_path)
                    if verbose:
                        logger.opt(depth=1).warning(
                            f'Error loading cache: {str(e)[:100]}, recomputing'
                        )

        result = func(*args, **kwargs)

        with disk_lock:
            if not osp.exists(cache_path):
                dump_json_or_pickle(result, cache_path)
        return result

    return wrapper


def _async_disk_memoize(
    func: AsyncFunc[P, R],
    keys: list[str] | None,
    cache_dir: str,
    ignore_self: bool,
    verbose: bool,
    key_fn: Callable[..., Any] | None,
) -> AsyncFunc[P, R]:
    @functools.wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            func_source, sub_dir, key_id = _compute_cache_components(
                func, args, kwargs, ignore_self, keys, key_fn
            )
            cache_path = osp.join(cache_dir, sub_dir, key_id)
            mkdir_or_exist(osp.dirname(cache_path))
        except Exception as e:
            if verbose:
                logger.opt(depth=1).warning(
                    f'Failed to cache {func.__name__}: {e}, executing without cache'
                )
            return await func(*args, **kwargs)

        def check_cache() -> R | None:
            with disk_lock:
                if osp.exists(cache_path):
                    try:
                        return load_json_or_pickle(cache_path)
                    except Exception as e:
                        if osp.exists(cache_path):
                            os.remove(cache_path)
                        if verbose:
                            logger.opt(depth=1).warning(
                                f'Error loading cache: {str(e)[:100]}, recomputing'
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

    return wrapper


# --------------------------------------------------------------------------------------
# Memory+Disk (sync / async)
# --------------------------------------------------------------------------------------


def both_memoize(
    func: Callable[P, R],
    keys: list[str] | None,
    cache_dir: str,
    ignore_self: bool,
    size: int,
    key_fn: Callable[..., Any] | None,
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
                return mem_cache[mem_key]

        # Disk next
        cache_path = osp.join(cache_dir, sub_dir, key_id)
        mkdir_or_exist(osp.dirname(cache_path))

        disk_result: R | None = None
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
                mem_cache[mem_key] = disk_result
            return disk_result

        # Miss: compute, then write both
        result = func(*args, **kwargs)
        with disk_lock:
            if not osp.exists(cache_path):
                dump_json_or_pickle(result, cache_path)
        with mem_lock:
            mem_cache[mem_key] = result
        return result

    return wrapper


def _async_both_memoize(
    func: AsyncFunc[P, R],
    keys: list[str] | None,
    cache_dir: str,
    ignore_self: bool,
    size: int,
    key_fn: Callable[..., Any] | None,
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
                return mem_cache[mem_key]

        # Disk
        cache_path = osp.join(cache_dir, sub_dir, key_id)
        mkdir_or_exist(osp.dirname(cache_path))

        def check_disk_cache() -> R | None:
            with disk_lock:
                if osp.exists(cache_path):
                    return load_json_or_pickle(cache_path)
            return None

        loop = asyncio.get_event_loop()
        disk_result = await loop.run_in_executor(None, check_disk_cache)

        if disk_result is not None:
            with mem_lock:
                mem_cache[mem_key] = disk_result
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
            mem_cache[mem_key] = result
        return result

    return wrapper


# --------------------------------------------------------------------------------------
# Public decorator (only export memoize)
# --------------------------------------------------------------------------------------


# Define overloads to preserve exact type information
@overload
def memoize(
    _func: Callable[P, Awaitable[R]],
    *,
    keys: list[str] | None = ...,
    key: Callable[..., Any] | None = ...,
    cache_dir: str = ...,
    cache_type: Literal['memory', 'disk', 'both'] = ...,
    size: int = ...,
    ignore_self: bool = ...,
    verbose: bool = ...,
) -> Callable[P, Awaitable[R]]: ...


@overload
def memoize(
    _func: Callable[P, R],
    *,
    keys: list[str] | None = ...,
    key: Callable[..., Any] | None = ...,
    cache_dir: str = ...,
    cache_type: Literal['memory', 'disk', 'both'] = ...,
    size: int = ...,
    ignore_self: bool = ...,
    verbose: bool = ...,
) -> Callable[P, R]: ...


@overload
def memoize(
    _func: None = ...,
    *,
    keys: list[str] | None = ...,
    key: Callable[..., Any] | None = ...,
    cache_dir: str = ...,
    cache_type: Literal['memory', 'disk', 'both'] = ...,
    size: int = ...,
    ignore_self: bool = ...,
    verbose: bool = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def memoize(  # type: ignore
    _func: None = ...,
    *,
    keys: list[str] | None = ...,
    key: Callable[..., Any] | None = ...,
    cache_dir: str = ...,
    cache_type: Literal['memory', 'disk', 'both'] = ...,
    size: int = ...,
    ignore_self: bool = ...,
    verbose: bool = ...,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...


def memoize(
    _func: Callable[P, Any] | None = None,
    *,
    keys: list[str] | None = None,
    key: Callable[..., Any] | None = None,
    cache_dir: str = SPEED_CACHE_DIR,
    cache_type: Literal['memory', 'disk', 'both'] = 'both',
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
    if '~/' in cache_dir:
        cache_dir = osp.expanduser(cache_dir)

    def decorator(func: Callable[P, Any]) -> Callable[P, Any]:
        is_async = inspect.iscoroutinefunction(func)

        # Apply timing decorator if verbose=True
        if verbose:
            from speedy_utils import timef

            target_func = timef(func)
        else:
            target_func = func

        if cache_type == 'memory':
            if is_async:
                return _async_memory_memoize(target_func, size, keys, ignore_self, key)
            return _memory_memoize(target_func, size, keys, ignore_self, key)

        if cache_type == 'disk':
            if is_async:
                return _async_disk_memoize(
                    target_func, keys, cache_dir, ignore_self, verbose, key
                )
            return _disk_memoize(
                target_func, keys, cache_dir, ignore_self, verbose, key
            )

        # cache_type == "both"
        if is_async:
            return _async_both_memoize(
                target_func, keys, cache_dir, ignore_self, size, key
            )
        return both_memoize(target_func, keys, cache_dir, ignore_self, size, key)

    # Support both @memoize and @memoize(...)
    if _func is None:
        return decorator
    return decorator(_func)


# --------------------------------------------------------------------------------------
# In-memory memoize with global persistent cache
# --------------------------------------------------------------------------------------


@overload
def imemoize(
    _func: Callable[P, Awaitable[R]],
    *,
    keys: list[str] | None = ...,
    key: Callable[..., Any] | None = ...,
    ignore_self: bool = ...,
) -> Callable[P, Awaitable[R]]: ...


@overload
def imemoize(
    _func: Callable[P, R],
    *,
    keys: list[str] | None = ...,
    key: Callable[..., Any] | None = ...,
    ignore_self: bool = ...,
) -> Callable[P, R]: ...


@overload
def imemoize(
    _func: None = ...,
    *,
    keys: list[str] | None = ...,
    key: Callable[..., Any] | None = ...,
    ignore_self: bool = ...,
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


@overload
def imemoize(  # type: ignore
    _func: None = ...,
    *,
    keys: list[str] | None = ...,
    key: Callable[..., Any] | None = ...,
    ignore_self: bool = ...,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...


def imemoize(
    _func: Callable[P, Any] | None = None,
    *,
    keys: list[str] | None = None,
    key: Callable[..., Any] | None = None,
    ignore_self: bool = True,
):
    """
    In-memory memoization decorator with global persistent cache.

    Unlike regular memoize, this uses a global memory cache that persists
    across IPython %load executions. The cache key is based on the function's
    source code combined with runtime arguments, making it suitable for
    notebook environments where functions may be reloaded.

    Args:
        keys: list of argument names to include in key (optional).
        key: custom callable (*args, **kwargs) -> hashable for keying (optional).
        ignore_self: ignore 'self' when building cache key for bound methods.

    Example:
        @imemoize
        def expensive_computation(x):
            import time
            time.sleep(2)
            return x * x

        # First call computes and caches
        result1 = expensive_computation(5)

        # Second call retrieves from memory cache
        result2 = expensive_computation(5)

        # Even after %load file.py in IPython, the cache persists
    """

    def decorator(func: Callable[P, Any]) -> Callable[P, Any]:
        is_async = inspect.iscoroutinefunction(func)

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                # Compute cache key based on function source + args
                func_source, sub_dir, key_id = _compute_cache_components(
                    func, args, kwargs, ignore_self, keys, key
                )
                cache_key = identify((func_source, sub_dir, key_id))

                # Check global memory cache
                with mem_lock:
                    if cache_key in _GLOBAL_MEMORY_CACHE:
                        return _GLOBAL_MEMORY_CACHE[cache_key]

                # Compute result and store in cache
                result = await func(*args, **kwargs)

                with mem_lock:
                    _GLOBAL_MEMORY_CACHE[cache_key] = result

                return result

            return async_wrapper

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
            # Compute cache key based on function source + args
            func_source, sub_dir, key_id = _compute_cache_components(
                func, args, kwargs, ignore_self, keys, key
            )
            cache_key = identify((func_source, sub_dir, key_id))

            # Check global memory cache
            with mem_lock:
                if cache_key in _GLOBAL_MEMORY_CACHE:
                    return _GLOBAL_MEMORY_CACHE[cache_key]

            # Compute result and store in cache
            result = func(*args, **kwargs)

            with mem_lock:
                _GLOBAL_MEMORY_CACHE[cache_key] = result

            return result

        return sync_wrapper

    # Support both @imemoize and @imemoize(...)
    if _func is None:
        return decorator
    return decorator(_func)


__all__ = ['memoize', 'imemoize', 'identify']
