"""
Multi-process implementation with sequential and threadpool backends.

Provides the public `multi_process` dispatcher and implementations for:
- 'seq': Sequential execution
- 'mp': ThreadPoolExecutor-based parallelism
- 'safe': Same as 'mp' but without process tracking (for tests)
"""
from __future__ import annotations

import concurrent.futures
import inspect
import os
import sys
import warnings
from typing import Any, Callable, Iterable, Literal

from tqdm import tqdm

from .common import (
    ErrorHandlerType,
    ErrorStats,
    _build_cache_dir,
    _call_with_log_control,
    _cleanup_log_gate,
    _exit_on_worker_error,
    _prune_dead_processes,
    _track_multiprocessing_processes,
    create_log_gate_path,
    wrap_dump,
)

# Import thread tracking functions if available
try:
    from .thread import _prune_dead_threads, _track_executor_threads
except ImportError:
    _prune_dead_threads = None  # type: ignore[assignment]
    _track_executor_threads = None  # type: ignore[assignment]


def multi_process(
    func: Callable[[Any], Any],
    items: Iterable[Any] | None = None,
    *,
    inputs: Iterable[Any] | None = None,
    workers: int | None = None,
    lazy_output: bool = False,
    progress: bool = True,
    backend: Literal['seq', 'ray', 'mp', 'safe'] = 'mp',
    desc: str | None = None,
    shared_kwargs: list[str] | None = None,
    dump_in_thread: bool = True,
    ray_metrics_port: int | None = None,
    log_worker: Literal['zero', 'first', 'all'] = 'first',
    total_items: int | None = None,
    poll_interval: float = 0.3,
    error_handler: ErrorHandlerType = 'log',
    max_error_files: int = 100,
    process_update_interval: int | None = None,
    batch: int | None = None,
    ordered: bool = True,
    stop_on_error: bool = True,
    **func_kwargs: Any,
) -> list[Any]:
    """
    Multi-process map with selectable backend.

    backend:
        - "seq": run sequentially
        - "ray": run in parallel with Ray
        - "mp": run in parallel with thread pool (uses ThreadPoolExecutor)
        - "safe": run in parallel with thread pool (explicitly safe for tests)

    shared_kwargs:
        - Optional list of kwarg names that should be shared via Ray's
          zero-copy object store
        - Only works with Ray backend
        - Useful for large objects (e.g., models, datasets)
        - Example: shared_kwargs=['model', 'tokenizer']

    dump_in_thread:
        - Whether to dump results to disk in a separate thread (default: True)
        - If False, dumping is done synchronously

    ray_metrics_port:
        - Optional port for Ray metrics export (Ray backend only)

    log_worker:
        - Control worker stdout/stderr noise
        - 'first': only first worker emits logs (default)
        - 'all': allow worker prints
        - 'zero': silence all worker output

    total_items:
        - Optional item-level total for progress tracking (Ray backend only)

    poll_interval:
        - Poll interval in seconds for progress actor updates (Ray only)

    error_handler:
        - 'raise': raise exception on first error
        - 'ignore': continue processing, return None for failed items
        - 'log': same as ignore, but logs errors to files (default)
        - Note: for 'mp' and 'ray' backends, 'raise' prints a formatted
          traceback and exits the process.

    max_error_files:
        - Maximum number of error log files to write (default: 100)
        - Error logs are written to .cache/speedy_utils/error_logs/{idx}.log
        - First error is always printed to screen with the log file path

    process_update_interval:
        - Legacy parameter, accepted for backward compatibility but not implemented

    batch:
        - Legacy parameter, accepted for backward compatibility but not implemented

    ordered:
        - Whether to maintain order of results (default: True)
        - Legacy parameter, accepted for backward compatibility but not implemented

    stop_on_error:
        - Whether to stop on first error (default: True)
        - Legacy parameter, accepted for backward compatibility
        - Use error_handler parameter instead for error handling control

    If lazy_output=True, every result is saved to .pkl and
    the returned list contains file paths.
    """

    # default backend selection
    if backend is None:
        try:
            import ray as _ray_module
            backend = 'ray'
        except ImportError:
            backend = 'mp'

    # Validate shared_kwargs
    if shared_kwargs:
        sig = inspect.signature(func)
        valid_params = set(sig.parameters.keys())

        for kw in shared_kwargs:
            if kw not in func_kwargs:
                raise ValueError(
                    f"shared_kwargs key '{kw}' not found in "
                    f"provided func_kwargs"
                )
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if kw not in valid_params and not has_var_keyword:
                raise ValueError(
                    f"shared_kwargs key '{kw}' is not a valid parameter "
                    f"for function '{func.__name__}'. "
                    f"Valid parameters: {valid_params}"
                )

    # Prefer Ray backend when shared kwargs are requested
    if shared_kwargs and backend != 'ray':
        warnings.warn(
            "shared_kwargs only supported with 'ray' backend, "
            "switching backend to 'ray'",
            UserWarning,
        )
        backend = 'ray'

    # unify items and coerce to concrete list
    if items is None and inputs is not None:
        items = list(inputs)
    if items is not None and not isinstance(items, list):
        items = list(items)
    if items is None:
        raise ValueError("'items' or 'inputs' must be provided")

    if workers is None and backend != 'ray':
        workers = os.cpu_count() or 1

    # build cache dir + wrap func
    cache_dir = _build_cache_dir(func, items) if lazy_output else None
    f_wrapped = wrap_dump(func, cache_dir, dump_in_thread)

    log_gate_path = create_log_gate_path(log_worker)

    total = len(items)
    if desc:
        desc = desc.strip() + f'[{backend}]'
    else:
        desc = f'Multi-process [{backend}]'

    # Initialize error stats for error handling
    func_name = getattr(func, '__name__', repr(func))
    error_stats = ErrorStats(
        func_name=func_name,
        max_error_files=max_error_files,
        write_logs=error_handler == 'log'
    )

    def _update_pbar_postfix(pbar: tqdm) -> None:
        """Update pbar with success/error counts."""
        postfix = error_stats.get_postfix_dict()
        pbar.set_postfix(postfix)

    # ---- sequential backend ----
    if backend == 'seq':
        return _run_seq_backend(
            f_wrapped=f_wrapped,
            items=items,
            total=total,
            desc=desc,
            progress=progress,
            func_kwargs=func_kwargs,
            log_worker=log_worker,
            log_gate_path=log_gate_path,
            error_handler=error_handler,
            error_stats=error_stats,
            func_name=func_name,
            update_pbar_postfix=_update_pbar_postfix,
        )

    # ---- ray backend ----
    if backend == 'ray':
        from ._multi_process_ray import run_ray_backend
        return run_ray_backend(
            f_wrapped=f_wrapped,
            items=items,
            total=total,
            workers=workers,
            progress=progress,
            desc=desc,
            func_kwargs=func_kwargs,
            shared_kwargs=shared_kwargs,
            log_worker=log_worker,
            log_gate_path=log_gate_path,
            total_items=total_items,
            poll_interval=poll_interval,
            ray_metrics_port=ray_metrics_port,
            error_handler=error_handler,
            error_stats=error_stats,
            func_name=func_name,
        )

    # ---- threadpool backends (mp / safe) ----
    if backend == 'mp':
        return _run_threadpool_backend(
            backend_label='mp',
            track_processes=True,
            f_wrapped=f_wrapped,
            items=items,
            total=total,
            workers=workers,
            desc=desc,
            progress=progress,
            func_kwargs=func_kwargs,
            log_worker=log_worker,
            log_gate_path=log_gate_path,
            error_handler=error_handler,
            error_stats=error_stats,
            func_name=func_name,
            update_pbar_postfix=_update_pbar_postfix,
        )

    if backend == 'safe':
        return _run_threadpool_backend(
            backend_label='safe',
            track_processes=False,
            f_wrapped=f_wrapped,
            items=items,
            total=total,
            workers=workers,
            desc=desc,
            progress=progress,
            func_kwargs=func_kwargs,
            log_worker=log_worker,
            log_gate_path=log_gate_path,
            error_handler=error_handler,
            error_stats=error_stats,
            func_name=func_name,
            update_pbar_postfix=_update_pbar_postfix,
        )

    raise ValueError(f'Unsupported backend: {backend!r}')


def _run_seq_backend(
    *,
    f_wrapped,
    items: list,
    total: int,
    desc: str,
    progress: bool,
    func_kwargs: dict,
    log_worker,
    log_gate_path,
    error_handler,
    error_stats: ErrorStats,
    func_name: str,
    update_pbar_postfix,
) -> list[Any]:
    """Run sequential (single-threaded) backend."""
    results: list[Any] = []
    with tqdm(
        total=total,
        desc=desc,
        disable=not progress,
        file=sys.stdout,
    ) as pbar:
        for idx, x in enumerate(items):
            try:
                result = _call_with_log_control(
                    f_wrapped,
                    x,
                    func_kwargs,
                    log_worker,
                    log_gate_path,
                )
                error_stats.record_success()
                results.append(result)
            except Exception as e:
                if error_handler == 'raise':
                    raise
                error_stats.record_error(idx, e, x, func_name)
                results.append(None)
            pbar.update(1)
            update_pbar_postfix(pbar)
    _cleanup_log_gate(log_gate_path)
    return results


def _run_threadpool_backend(
    *,
    backend_label: str,
    track_processes: bool,
    f_wrapped,
    items: list,
    total: int,
    workers: int | None,
    desc: str,
    progress: bool,
    func_kwargs: dict,
    log_worker,
    log_gate_path,
    error_handler,
    error_stats: ErrorStats,
    func_name: str,
    update_pbar_postfix,
) -> list[Any]:
    """Run ThreadPoolExecutor backend for 'mp' and 'safe' modes."""
    # Capture caller frame for better error reporting
    caller_frame = inspect.currentframe()
    caller_info = None
    if caller_frame and caller_frame.f_back and caller_frame.f_back.f_back:
        # Go back two frames: _run_threadpool_backend -> multi_process -> user
        outer = caller_frame.f_back.f_back
        caller_info = {
            'filename': outer.f_code.co_filename,
            'lineno': outer.f_lineno,
            'function': outer.f_code.co_name,
        }

    def worker_func(x):
        return _call_with_log_control(
            f_wrapped,
            x,
            func_kwargs,
            log_worker,
            log_gate_path,
        )

    results: list[Any] = [None] * total
    with tqdm(
        total=total,
        desc=desc,
        disable=not progress,
        file=sys.stdout,
    ) as pbar:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers
        ) as executor:
            if _track_executor_threads is not None:
                _track_executor_threads(executor)

            # Submit all tasks
            future_to_idx = {
                executor.submit(worker_func, x): idx
                for idx, x in enumerate(items)
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    error_stats.record_success()
                    results[idx] = result
                except Exception as e:
                    if error_handler == 'raise':
                        # Cancel remaining futures
                        for f in future_to_idx:
                            f.cancel()
                        _exit_on_worker_error(
                            e,
                            pbar,
                            caller_info,
                            backend=backend_label,
                        )
                    error_stats.record_error(idx, e, items[idx], func_name)
                    results[idx] = None
                pbar.update(1)
                update_pbar_postfix(pbar)

        if _prune_dead_threads is not None:
            _prune_dead_threads()

    if track_processes:
        _track_multiprocessing_processes()
        _prune_dead_processes()

    _cleanup_log_gate(log_gate_path)
    return results


__all__ = ['multi_process']
