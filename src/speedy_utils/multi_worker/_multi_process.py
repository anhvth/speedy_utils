"""
Multi-process implementation with sequential, multiprocessing, and test backends.

Provides the public `multi_process` dispatcher and implementations for:
- 'seq': Sequential execution
- 'mp': Multiprocessing fan-out with optional threads inside each process
- 'safe': In-process thread pool backend for tests and local debugging
"""
from __future__ import annotations

import concurrent.futures
import inspect
import multiprocessing
import os
import queue
import sys
import threading
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import psutil
from tqdm import tqdm

from .common import (
    ErrorHandlerType,
    ErrorStats,
    _build_cache_dir,
    _call_with_log_control,
    _cleanup_log_gate,
    _display_formatted_error_and_exit,
    _exit_on_worker_error,
    _prune_dead_processes,
    _ThreadLocalStream,
    _track_processes,
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
    num_procs: int | None = None,
    num_threads: int = 1,
    lazy_output: bool = False,
    progress: bool = True,
    backend: Literal["seq", "ray", "mp", "safe"] = "mp",
    desc: str | None = None,
    shared_kwargs: list[str] | None = None,
    dump_in_thread: bool = True,
    ray_metrics_port: int | None = None,
    log_worker: Literal["zero", "first", "all"] = "first",
    total_items: int | None = None,
    poll_interval: float = 0.3,
    error_handler: ErrorHandlerType = "log",
    max_error_files: int = 100,
    process_update_interval: int | None = None,
    batch: int | None = None,
    ordered: bool = True,
    stop_on_error: bool = True,
    **func_kwargs: Any,
) -> list[Any]:
    """
    Multi-process map with selectable backend and optional inner thread pools.

    This function provides nested parallelism: it spawns multiple processes
    (num_procs), and each process has its own thread pool (num_threads) for
    concurrent execution within the process.

    Parameters
    ----------
    func : Callable
        Function to apply to each item.
    items : Iterable
        Input items to process.
    inputs : Iterable, optional
        Alias for items (for backward compatibility).
    workers : int, optional
        Deprecated. Use num_procs instead.
    num_procs : int, optional
        Number of worker processes. Default: os.cpu_count() for 'mp' backend.
        Ignored for 'safe' and 'seq' backends.
    num_threads : int, default=1
        Number of threads per process. Each process gets its own ThreadPoolExecutor
        with this many workers. Use >1 for I/O-bound work within each process.

    Backend Options
    ---------------
    backend : str
        - "seq": Sequential execution (for debugging)
        - "mp": Multiprocessing with optional threads per process (default)
        - "safe": In-process thread pool (for testing, ignores num_procs)
        - "ray": Distributed Ray backend (requires ray extra)

    Parallelism Strategy
    --------------------
    - CPU-bound tasks: Use num_procs > 1, num_threads = 1 (processes bypass GIL)
    - I/O-bound tasks: Use num_procs = 1, num_threads > 1 (threads are lighter)
    - Mixed workloads: Use both (e.g., num_procs=4, num_threads=4)

    Examples
    --------
    >>> # CPU-bound: 4 processes, 1 thread each
    >>> results = multi_process(lambda x: x**2, range(100), num_procs=4)

    >>> # I/O-bound: 1 process, 8 threads
    >>> results = multi_process(fetch_url, urls, num_threads=8, backend='safe')

    >>> # Mixed: 4 processes with 4 threads each = 16 concurrent workers
    >>> results = multi_process(process_data, data, num_procs=4, num_threads=4)

    Other Parameters
    ----------------
    shared_kwargs : list[str], optional
        Kwarg names to share via Ray's zero-copy object store (Ray only).
    dump_in_thread : bool, default=True
        Whether to dump results asynchronously when lazy_output=True.
    ray_metrics_port : int, optional
        Port for Ray metrics export (Ray backend only).
    log_worker : str, default='first'
        Control worker stdout/stderr: 'first', 'all', or 'zero'.
    total_items : int, optional
        Item count for progress tracking (Ray backend only).
    poll_interval : float, default=0.3
        Poll interval for progress updates (Ray only).
    error_handler : str, default='log'
        - 'raise': Stop on first error with traceback
        - 'ignore': Continue, return None for failed items
        - 'log': Continue, log errors to files
    max_error_files : int, default=100
        Max error log files to write.
    process_update_interval : int, optional
        Legacy parameter (not implemented).
    batch : int, optional
        Legacy parameter (not implemented).
    ordered : bool, default=True
        Whether to maintain input order in results.
    stop_on_error : bool, default=True
        Deprecated. Use error_handler instead.
    lazy_output : bool, default=False
        If True, save results to .pkl files and return file paths.
    progress : bool, default=True
        Show progress bar.
    desc : str, optional
        Progress bar description.

    Returns
    -------
    list[Any]
        Results in same order as inputs (when ordered=True).
        Failed items return None when error_handler != 'raise'.
    """
    if num_threads <= 0:
        raise ValueError("num_threads must be a positive integer")

    if workers is not None:
        warnings.warn(
            "'workers' is deprecated for multi_process; use 'num_procs' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if num_procs is not None and num_procs != workers:
            raise ValueError(
                "'workers' and 'num_procs' must match when both are provided"
            )
        if num_procs is None:
            num_procs = workers

    # default backend selection
    if backend is None:
        try:
            import ray as _ray_module

            backend = "ray"
        except ImportError:
            backend = "mp"

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
    if shared_kwargs and backend != "ray":
        warnings.warn(
            "shared_kwargs only supported with 'ray' backend, "
            "switching backend to 'ray'",
            UserWarning,
            stacklevel=2,
        )
        backend = "ray"

    # unify items and coerce to concrete list
    if items is None and inputs is not None:
        items = list(inputs)
    if items is not None and not isinstance(items, list):
        items = list(items)
    if items is None:
        raise ValueError("'items' or 'inputs' must be provided")

    if num_procs is None and backend == "mp":
        num_procs = os.cpu_count() or 1
    if workers is None and backend == "ray":
        workers = os.cpu_count() or 1

    # build cache dir + wrap func
    cache_dir = _build_cache_dir(func, items) if lazy_output else None
    f_wrapped = wrap_dump(func, cache_dir, dump_in_thread)

    log_gate_path = create_log_gate_path(log_worker)

    total = len(items)
    if desc:
        desc = desc.strip() + f"[{backend}]"
    else:
        desc = f"Multi-process [{backend}]"

    # Initialize error stats for error handling
    func_name = getattr(func, "__name__", repr(func))
    error_stats = ErrorStats(
        func_name=func_name,
        max_error_files=max_error_files,
        write_logs=error_handler == "log",
    )

    def _update_pbar_postfix(pbar: tqdm) -> None:
        """Update pbar with success/error counts."""
        postfix = error_stats.get_postfix_dict()
        pbar.set_postfix(postfix)

    # ---- sequential backend ----
    if backend == "seq":
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
    if backend == "ray":
        try:
            import ray  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Ray backend requires optional dependency `ray`. Install with "
                "`pip install 'speedy-utils[ray]'` (or `uv sync --extra ray`)."
            ) from e

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

    if backend == "mp":
        return _run_multiprocess_backend(
            func=func,
            cache_dir=cache_dir,
            dump_in_thread=dump_in_thread,
            items=items,
            total=total,
            num_procs=num_procs or 1,
            num_threads=num_threads,
            desc=desc,
            progress=progress,
            func_kwargs=func_kwargs,
            log_worker=log_worker,
            log_gate_path=log_gate_path,
            error_handler=error_handler,
            error_stats=error_stats,
            func_name=func_name,
            update_pbar_postfix=_update_pbar_postfix,
            max_error_files=max_error_files,
        )

    if backend == "safe":
        safe_workers = num_threads
        if safe_workers == 1 and workers is not None:
            safe_workers = workers
        return _run_threadpool_backend(
            backend_label="safe",
            f_wrapped=f_wrapped,
            items=items,
            total=total,
            workers=safe_workers,
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

    raise ValueError(f"Unsupported backend: {backend!r}")


def _chunk_indexed_items(
    indexed_items: list[tuple[int, Any]],
    num_procs: int,
) -> list[list[tuple[int, Any]]]:
    """Split indexed items into contiguous chunks for worker processes."""
    if not indexed_items:
        return []
    proc_count = max(1, min(num_procs, len(indexed_items)))
    chunk_size = max((len(indexed_items) + proc_count - 1) // proc_count, 1)
    return [
        indexed_items[start : start + chunk_size]
        for start in range(0, len(indexed_items), chunk_size)
    ]


def _serialize_exception_frames(exc: Exception) -> list[tuple[str, int, str, dict]]:
    """Extract a picklable traceback payload for cross-process reporting."""
    tb = sys.exc_info()[2] or exc.__traceback__
    if tb is None:
        return []
    return [
        (frame.filename, frame.lineno, frame.name, {})
        for frame in traceback.extract_tb(tb)
    ]


def _bump_error_count(error_stats: ErrorStats, err_inc: int) -> None:
    if err_inc <= 0:
        return
    with error_stats._lock:  # type: ignore[attr-defined]
        error_stats._error_count += err_inc  # type: ignore[attr-defined]


def _run_mp_chunk(
    *,
    chunk: list[tuple[int, Any]],
    func: Callable[[Any], Any],
    cache_dir: Path | None,
    dump_in_thread: bool,
    num_threads: int,
    func_kwargs: dict[str, Any],
    log_worker: Literal["zero", "first", "all"],
    log_gate_path: Path | None,
    error_handler: ErrorHandlerType,
    max_error_files: int,
    func_name: str,
    event_queue: multiprocessing.queues.Queue,
) -> None:
    """Execute one process chunk, optionally with a per-process thread pool."""
    f_wrapped = wrap_dump(func, cache_dir, dump_in_thread)
    child_error_stats = None
    child_error_lock = threading.Lock()
    child_error_count = 0
    if error_handler == "log":
        child_error_stats = ErrorStats(
            func_name=func_name,
            max_error_files=max_error_files,
            write_logs=True,
        )

    fatal_lock = threading.Lock()
    fatal_sent = False

    def report_fatal(exc: Exception) -> None:
        nonlocal fatal_sent
        if error_handler != "raise":
            return
        with fatal_lock:
            if fatal_sent:
                return
            fatal_sent = True
        event_queue.put(
            (
                "fatal",
                type(exc).__name__,
                str(exc),
                _serialize_exception_frames(exc),
            )
        )

    def process_one(payload: tuple[int, Any]) -> Any:
        nonlocal child_error_count
        idx, item = payload
        try:
            result = _call_with_log_control(
                f_wrapped,
                item,
                func_kwargs,
                log_worker,
                log_gate_path,
            )
        except Exception as exc:
            if child_error_stats is not None:
                with child_error_lock:
                    child_error_count += 1
                    should_write = child_error_count <= max_error_files
                if should_write:
                    child_error_stats._write_error_log(  # type: ignore[attr-defined]
                        idx,
                        exc,
                        item,
                        func_name,
                    )
            event_queue.put(("progress", 0, 1))
            report_fatal(exc)
            raise
        event_queue.put(("result", idx, result))
        event_queue.put(("progress", 1, 0))
        return result

    try:
        if num_threads <= 1:
            for payload in chunk:
                try:
                    process_one(payload)
                except Exception:
                    if error_handler == "raise":
                        return
            return

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads
        ) as executor:
            futures = [executor.submit(process_one, payload) for payload in chunk]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception:
                    if error_handler == "raise":
                        for pending in futures:
                            pending.cancel()
                        return
    finally:
        event_queue.put(("process_done", os.getpid()))


def _multiprocess_entrypoint(
    chunk: list[tuple[int, Any]],
    func: Callable[[Any], Any],
    cache_dir: Path | None,
    dump_in_thread: bool,
    num_threads: int,
    func_kwargs: dict[str, Any],
    log_worker: Literal["zero", "first", "all"],
    log_gate_path: Path | None,
    error_handler: ErrorHandlerType,
    max_error_files: int,
    func_name: str,
    event_queue: multiprocessing.queues.Queue,
) -> None:
    """Multiprocessing target wrapper with crash-safe error forwarding."""
    try:
        _run_mp_chunk(
            chunk=chunk,
            func=func,
            cache_dir=cache_dir,
            dump_in_thread=dump_in_thread,
            num_threads=num_threads,
            func_kwargs=func_kwargs,
            log_worker=log_worker,
            log_gate_path=log_gate_path,
            error_handler=error_handler,
            max_error_files=max_error_files,
            func_name=func_name,
            event_queue=event_queue,
        )
    except BaseException as exc:  # pragma: no cover - last-resort forwarding
        event_queue.put(
            (
                "fatal",
                type(exc).__name__,
                str(exc),
                _serialize_exception_frames(exc),
            )
        )
        raise


def _terminate_processes(processes: list[multiprocessing.Process]) -> None:
    """Best-effort shutdown for spawned worker processes."""
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(timeout=1)
    _prune_dead_processes()


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
                if error_handler == "raise":
                    raise
                error_stats.record_error(idx, e, x, func_name)
                results.append(None)
            pbar.update(1)
            update_pbar_postfix(pbar)
    _cleanup_log_gate(log_gate_path)
    return results


def _run_multiprocess_backend(
    *,
    func: Callable[[Any], Any],
    cache_dir: Path | None,
    dump_in_thread: bool,
    items: list,
    total: int,
    num_procs: int,
    num_threads: int,
    desc: str,
    progress: bool,
    func_kwargs: dict,
    log_worker,
    log_gate_path,
    error_handler,
    error_stats: ErrorStats,
    func_name: str,
    update_pbar_postfix,
    max_error_files: int,
) -> list[Any]:
    """Run the multiprocessing backend with a single parent-owned tqdm."""
    caller_frame = inspect.currentframe()
    caller_info = None
    if caller_frame and caller_frame.f_back and caller_frame.f_back.f_back:
        outer = caller_frame.f_back.f_back
        caller_info = {
            "filename": outer.f_code.co_filename,
            "lineno": outer.f_lineno,
            "function": outer.f_code.co_name,
        }

    indexed_items = list(enumerate(items))
    chunks = _chunk_indexed_items(indexed_items, num_procs)
    if not chunks:
        _cleanup_log_gate(log_gate_path)
        return []

    ctx = multiprocessing.get_context("spawn")
    event_queue = ctx.Queue()
    processes: list[multiprocessing.Process] = []

    try:
        for chunk in chunks:
            proc = ctx.Process(
                target=_multiprocess_entrypoint,
                args=(
                    chunk,
                    func,
                    cache_dir,
                    dump_in_thread,
                    num_threads,
                    func_kwargs,
                    log_worker,
                    log_gate_path,
                    error_handler,
                    max_error_files,
                    func_name,
                    event_queue,
                ),
            )
            proc.start()
            processes.append(proc)

        tracked = []
        for proc in processes:
            if proc.pid is None:
                continue
            try:
                tracked.append(psutil.Process(proc.pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _track_processes(tracked)

        results: list[Any] = [None] * total
        done_processes = 0

        with tqdm(
            total=total,
            desc=desc,
            disable=not progress,
            file=sys.stdout,
        ) as pbar:
            while done_processes < len(processes):
                try:
                    msg = event_queue.get(timeout=0.1)
                except queue.Empty:
                    if all(proc.exitcode is not None for proc in processes):
                        done_processes = len(processes)
                    continue

                tag = msg[0]
                if tag == "result":
                    _, idx, result = msg
                    results[idx] = result
                    continue

                if tag == "progress":
                    _, ok_inc, err_inc = msg
                    for _ in range(ok_inc):
                        error_stats.record_success()
                    _bump_error_count(error_stats, err_inc)
                    pbar.update(ok_inc + err_inc)
                    update_pbar_postfix(pbar)
                    continue

                if tag == "process_done":
                    done_processes += 1
                    continue

                if tag == "fatal":
                    _, exc_type_name, exc_msg, frames = msg
                    _terminate_processes(processes)
                    _display_formatted_error_and_exit(
                        exc_type_name=exc_type_name,
                        exc_msg=exc_msg,
                        frames=frames,
                        caller_info=caller_info,
                        backend="mp",
                        pbar=pbar,
                    )

            for proc in processes:
                proc.join()

        _cleanup_log_gate(log_gate_path)
        return results
    finally:
        _cleanup_log_gate(log_gate_path)


def _run_threadpool_backend(
    *,
    backend_label: str,
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
    """Run ThreadPoolExecutor backend for the in-process safe mode."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _ThreadLocalStream(original_stdout)
    sys.stderr = _ThreadLocalStream(original_stderr)

    caller_frame = inspect.currentframe()
    caller_info = None
    if caller_frame and caller_frame.f_back and caller_frame.f_back.f_back:
        outer = caller_frame.f_back.f_back
        caller_info = {
            "filename": outer.f_code.co_filename,
            "lineno": outer.f_lineno,
            "function": outer.f_code.co_name,
        }

    def worker_func(x):
        return _call_with_log_control(
            f_wrapped,
            x,
            func_kwargs,
            log_worker,
            log_gate_path,
        )

    try:
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

                future_to_idx = {
                    executor.submit(worker_func, x): idx
                    for idx, x in enumerate(items)
                }

                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        error_stats.record_success()
                        results[idx] = result
                    except Exception as e:
                        if error_handler == "raise":
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

        _cleanup_log_gate(log_gate_path)
        return results
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


__all__ = ["multi_process"]
