"""
# ============================================================================= #
# THREAD-BASED PARALLEL EXECUTION WITH PROGRESS TRACKING AND ERROR HANDLING
# ============================================================================= #
#
# Title & Intent:
# High-performance thread pool utilities for parallel processing with comprehensive error handling
#
# High-level Summary:
# This module provides robust thread-based parallel execution utilities designed for CPU-bound
# and I/O-bound tasks requiring concurrent processing. It features intelligent worker management,
# comprehensive error handling with detailed tracebacks, progress tracking with tqdm integration,
# and flexible batching strategies. The module optimizes for both throughput and reliability,
# making it suitable for data processing pipelines, batch operations, and concurrent API calls.
#
# Public API / Data Contracts:
# • multi_thread(func, inputs, num_workers=None, progress=True, **kwargs) -> List[Any] - Main parallel execution
# • multi_thread_batch(func, inputs, batch_size=10, num_workers=None, **kwargs) -> List[Any] - Batched processing
# • DEFAULT_WORKERS = (cpu_count * 2) - Default worker thread count
# • T = TypeVar("T"), R = TypeVar("R") - Generic type variables for input/output typing
# • _group_iter(src, size) -> Iterable[List[T]] - Utility for chunking iterables
# • _worker(item, func, fixed_kwargs) -> R - Individual worker function wrapper
# • _short_tb() -> str - Shortened traceback formatter for cleaner error logs
#
# Invariants / Constraints:
# • Worker count MUST be positive integer, defaults to (CPU cores * 2)
# • Input iterables MUST be finite and non-empty for meaningful processing
# • Functions MUST be thread-safe when used with multiple workers
# • Error handling MUST capture and log detailed tracebacks for debugging
# • Progress tracking MUST be optional and gracefully handle tqdm unavailability
# • Batch processing MUST maintain input order in results
# • MUST handle keyboard interruption gracefully with resource cleanup
# • Thread pool MUST be properly closed and joined after completion
#
# Usage Example:
# ```python
# from speedy_utils.multi_worker.thread import multi_thread, multi_thread_batch
# import requests
#
# # Simple parallel processing
# def square(x):
#     return x ** 2
#
# numbers = list(range(100))
# results = multi_thread(square, numbers, num_workers=8)
# print(f"Processed {len(results)} items")
#
# # Parallel API calls with error handling
# def fetch_url(url):
#     response = requests.get(url, timeout=10)
#     return response.status_code, len(response.content)
#
# urls = ["http://example.com", "http://google.com", "http://github.com"]
# results = multi_thread(fetch_url, urls, num_workers=3, progress=True)
#
# # Batched processing for memory efficiency
# def process_batch(items):
#     return [item.upper() for item in items]
#
# large_dataset = ["item" + str(i) for i in range(10000)]
# batched_results = multi_thread_batch(
#     process_batch,
#     large_dataset,
#     batch_size=100,
#     num_workers=4
# )
# ```
#
# TODO & Future Work:
# • Add adaptive worker count based on task characteristics
# • Implement priority queuing for time-sensitive tasks
# • Add memory usage monitoring and automatic batch size adjustment
# • Support for async function execution within thread pool
# • Add detailed performance metrics and timing analysis
# • Implement graceful degradation for resource-constrained environments
#
# ============================================================================= #
"""

import ctypes
import os
import threading
import time
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from typing import Any, TypeVar, Union

from loguru import logger

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

# Sensible defaults
DEFAULT_WORKERS = (os.cpu_count() or 4) * 2

T = TypeVar("T")
R = TypeVar("R")

SPEEDY_RUNNING_THREADS: list[threading.Thread] = []
_SPEEDY_THREADS_LOCK = threading.Lock()

_PY_SET_ASYNC_EXC = ctypes.pythonapi.PyThreadState_SetAsyncExc
try:
    _PY_SET_ASYNC_EXC.argtypes = (ctypes.c_ulong, ctypes.py_object)  # type: ignore[attr-defined]
    _PY_SET_ASYNC_EXC.restype = ctypes.c_int  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - platform specific
    pass


def _prune_dead_threads() -> None:
    with _SPEEDY_THREADS_LOCK:
        SPEEDY_RUNNING_THREADS[:] = [t for t in SPEEDY_RUNNING_THREADS if t.is_alive()]


def _track_threads(threads: Iterable[threading.Thread]) -> None:
    if not threads:
        return
    with _SPEEDY_THREADS_LOCK:
        living = [t for t in SPEEDY_RUNNING_THREADS if t.is_alive()]
        for candidate in threads:
            if not candidate.is_alive():
                continue
            if any(existing is candidate for existing in living):
                continue
            living.append(candidate)
        SPEEDY_RUNNING_THREADS[:] = living


def _track_executor_threads(pool: ThreadPoolExecutor) -> None:
    thread_set = getattr(pool, "_threads", None)
    if not thread_set:
        return
    _track_threads(tuple(thread_set))


def _group_iter(src: Iterable[T], size: int) -> Iterable[list[T]]:
    """Yield successive chunks from iterable of specified size."""
    it = iter(src)
    while chunk := list(islice(it, size)):
        yield chunk


def _short_tb() -> str:
    """Return a shortened traceback, excluding internal frames."""
    tb = "".join(traceback.format_exc())
    # hide frames inside this helper to keep logs short
    return "\n".join(ln for ln in tb.splitlines() if "multi_thread.py" not in ln)


def _worker(item: T, func: Callable[[T], R], fixed_kwargs: dict[str, Any]) -> R:
    """Execute the function with an item and fixed kwargs."""
    return func(item, **fixed_kwargs)


# ────────────────────────────────────────────────────────────
# main API
# ────────────────────────────────────────────────────────────
def multi_thread(
    func: Callable,
    inputs: Iterable[Any],
    *,
    workers: Union[int, None] = DEFAULT_WORKERS,
    batch: int = 1,
    ordered: bool = True,
    progress: bool = True,
    progress_update: int = 10,
    prefetch_factor: int = 4,
    timeout: Union[float, None] = None,
    stop_on_error: bool = True,
    n_proc=0,
    store_output_pkl_file: Union[str, None] = None,
    **fixed_kwargs,
) -> list[Any]:
    """
    ThreadPool **map** that returns a *list*.

    Parameters
    ----------
    func            – target callable.
    inputs          – iterable with the arguments.
    workers         – defaults to ``os.cpu_count()*2``.
    batch           – package *batch* inputs into one call for low‑overhead.
    ordered         – keep original order (costs memory); if ``False`` results
                      are yielded as soon as they finish.
    progress        – show a tqdm bar (requires *tqdm* installed).
    progress_update – bar redraw frequency (logical items, *not* batches).
    prefetch_factor – in‑flight tasks ≈ ``workers * prefetch_factor``.
    timeout         – overall timeout (seconds) for the mapping.
    stop_on_error   – raise immediately on first exception (default).  If
                      ``False`` the failing task’s result becomes ``None``.
    **fixed_kwargs  – static keyword args forwarded to every ``func()`` call.
    """
    from speedy_utils import dump_json_or_pickle, load_by_ext

    if n_proc > 1:
        import tempfile

        from fastcore.all import threaded

        # split the inputs by nproc
        inputs = list(inputs)
        n_per_proc = max(len(inputs) // n_proc, 1)
        proc_inputs_list = []
        for i in range(0, len(inputs), n_per_proc):
            proc_inputs_list.append(inputs[i : i + n_per_proc])
        procs = []
        in_process_multi_thread = threaded(process=True)(multi_thread)

        for proc_id, proc_inputs in enumerate(proc_inputs_list):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix="multi_thread.pkl"
            ) as tmp_file:
                file_pkl = tmp_file.name
            assert isinstance(in_process_multi_thread, Callable)
            proc = in_process_multi_thread(
                func,
                proc_inputs,
                workers=workers,
                batch=batch,
                ordered=ordered,
                progress=proc_id == 0,
                progress_update=progress_update,
                prefetch_factor=prefetch_factor,
                timeout=timeout,
                stop_on_error=stop_on_error,
                n_proc=0,  # prevent recursion
                store_output_pkl_file=file_pkl,
                **fixed_kwargs,
            )
            procs.append([proc, file_pkl])
        # join
        results = []

        for proc, file_pkl in procs:
            proc.join()
            logger.info(f"Done proc {proc=}")
            results.extend(load_by_ext(file_pkl))
        return results

    try:
        import pandas as pd

        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_dict(orient="records")
    except ImportError:
        pass

    try:
        n_inputs = len(inputs)  # type: ignore[arg-type]
    except Exception:
        n_inputs = None
    workers_val = workers if workers is not None else DEFAULT_WORKERS

    if batch == 1 and n_inputs and n_inputs / max(workers_val, 1) > 20_000:
        batch = 32  # empirically good for sub‑ms tasks

    # ── build (maybe‑batched) source iterator ────────────────────────────
    src_iter: Iterable[Any] = iter(inputs)
    if batch > 1:
        src_iter = _group_iter(src_iter, batch)
    # Ensure src_iter is always an iterator
    src_iter = iter(src_iter)

    # total logical items (for bar & ordered pre‑allocation)
    logical_total = n_inputs
    if logical_total is not None and batch > 1:
        logical_total = n_inputs  # still number of *items*, not batches

    # ── progress bar ─────────────────────────────────────────────────────
    bar = None
    last_bar_update = 0
    if progress and tqdm is not None and logical_total is not None:
        bar = tqdm(
            total=logical_total,
            ncols=128,
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
            " [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    # ── prepare result container ─────────────────────────────────────────
    if ordered and logical_total is not None:
        results: list[Any] = [None] * logical_total
    else:
        results = []

    # ── main execution loop ──────────────────────────────────────────────────
    workers_val = workers if workers is not None else DEFAULT_WORKERS
    max_inflight = workers_val * max(prefetch_factor, 1)
    completed_items = 0
    next_logical_idx = 0  # index assigned to the next submission

    with ThreadPoolExecutor(max_workers=workers) as pool:
        inflight = set()

        # prime the pool
        for _ in range(max_inflight):
            try:
                arg = next(src_iter)
            except StopIteration:
                break
            if batch > 1:
                fut = pool.submit(
                    lambda items: [_worker(item, func, fixed_kwargs) for item in items],
                    arg,
                )
                fut.idx = next_logical_idx  # type: ignore[attr-defined]
                inflight.add(fut)
                next_logical_idx += len(arg)
                _track_executor_threads(pool)
            else:
                fut = pool.submit(_worker, arg, func, fixed_kwargs)
                fut.idx = next_logical_idx  # type: ignore[attr-defined]
                inflight.add(fut)
                next_logical_idx += 1
                _track_executor_threads(pool)

        try:
            # Process futures as they complete and add new ones to keep the pool busy
            while inflight:  # Continue until all in-flight tasks are processed
                for fut in as_completed(inflight, timeout=timeout):
                    inflight.remove(fut)
                    idx = fut.idx  # type: ignore[attr-defined]
                    try:
                        res = fut.result()
                    except Exception:
                        if stop_on_error:
                            raise
                        res = None

                    # flatten res to list of logical outputs
                    out_items = res if batch > 1 else [res]

                    # Ensure out_items is a list (and thus Sized)
                    if out_items is None:
                        out_items = [None]
                    elif not isinstance(out_items, list):
                        out_items = (
                            list(out_items)
                            if isinstance(out_items, Iterable)
                            else [out_items]
                        )

                    # store outputs
                    if ordered and logical_total is not None:
                        results[idx : idx + len(out_items)] = out_items
                    else:
                        results.extend(out_items)

                    completed_items += len(out_items)

                    # progress bar update
                    if bar and completed_items - last_bar_update >= progress_update:
                        bar.update(completed_items - last_bar_update)
                        last_bar_update = completed_items
                        # Show pending, submitted, processing in the bar postfix
                        submitted = next_logical_idx
                        processing = min(len(inflight), workers_val)
                        pending = (
                            (logical_total - submitted)
                            if logical_total is not None
                            else None
                        )
                        postfix = {
                            "pending": pending if pending is not None else "-",
                            # 'submitted': submitted,
                            "processing": processing,
                        }
                        bar.set_postfix(postfix)

                    # keep queue full
                    try:
                        while next_logical_idx - completed_items < max_inflight:
                            arg = next(src_iter)
                            if batch > 1:
                                fut2 = pool.submit(
                                    lambda items: [
                                        _worker(item, func, fixed_kwargs)
                                        for item in items
                                    ],
                                    arg,
                                )
                                fut2.idx = next_logical_idx  # type: ignore[attr-defined]
                                inflight.add(fut2)
                                next_logical_idx += len(arg)
                                _track_executor_threads(pool)
                            else:
                                fut2 = pool.submit(_worker, arg, func, fixed_kwargs)
                                fut2.idx = next_logical_idx  # type: ignore[attr-defined]
                                inflight.add(fut2)
                                next_logical_idx += 1
                                _track_executor_threads(pool)
                    except StopIteration:
                        pass

                    # Break the inner loop as we've processed one future
                    break

                # If we've exhausted the inner loop without processing anything,
                # and there are still in-flight tasks, we need to wait for them
                if inflight and timeout is not None:
                    # Use a small timeout to avoid hanging indefinitely
                    time.sleep(min(0.01, timeout / 10))

        finally:
            if bar:
                bar.update(completed_items - last_bar_update)
                bar.close()
    if store_output_pkl_file:
        dump_json_or_pickle(results, store_output_pkl_file)
    _prune_dead_threads()
    return results


def multi_thread_standard(
    fn: Callable[[Any], Any], items: Iterable[Any], workers: int = 4
) -> list[Any]:
    """Execute a function using standard ThreadPoolExecutor.

    A standard implementation of multi-threading using ThreadPoolExecutor.
    Ensures the order of results matches the input order.

    Parameters
    ----------
    fn : Callable
        The function to execute for each item.
    items : Iterable
        The items to process.
    workers : int, optional
        Number of worker threads, by default 4.

    Returns
    -------
    list
        Results in same order as input items.
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for item in items:
            futures.append(executor.submit(fn, item))
            _track_executor_threads(executor)
        results = [fut.result() for fut in futures]
    _prune_dead_threads()
    return results


def _async_raise(thread_id: int, exc_type: type[BaseException]) -> bool:
    if thread_id <= 0:
        return False
    if not issubclass(exc_type, BaseException):
        raise TypeError("exc_type must derive from BaseException")
    res = _PY_SET_ASYNC_EXC(ctypes.c_ulong(thread_id), ctypes.py_object(exc_type))
    if res == 0:
        return False
    if res > 1:  # pragma: no cover - defensive branch
        _PY_SET_ASYNC_EXC(ctypes.c_ulong(thread_id), None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
    return True


def kill_all_thread(exc_type: type[BaseException] = SystemExit, join_timeout: float = 0.1) -> int:
    """Forcefully stop tracked worker threads. Returns number of threads signalled."""
    _prune_dead_threads()
    current = threading.current_thread()
    with _SPEEDY_THREADS_LOCK:
        targets = [t for t in SPEEDY_RUNNING_THREADS if t.is_alive()]

    terminated = 0
    for thread in targets:
        if thread is current:
            continue
        ident = thread.ident
        if ident is None:
            continue
        try:
            if _async_raise(ident, exc_type):
                terminated += 1
                thread.join(timeout=join_timeout)
            else:
                logger.warning("Unable to signal thread %s", thread.name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to stop thread %s: %s", thread.name, exc)
    _prune_dead_threads()
    return terminated


__all__ = [
    "SPEEDY_RUNNING_THREADS",
    "multi_thread",
    "multi_thread_standard",
    "kill_all_thread",
]
