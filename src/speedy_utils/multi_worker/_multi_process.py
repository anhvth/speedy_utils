"""Multi-process dispatcher without Ray support.

This module keeps the surviving `seq`, `mp`, and thread-backed execution paths
while removing the Ray-specific backend and its plumbing.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import multiprocessing as mp
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


try:
    from .thread import _prune_dead_threads, _track_executor_threads
except ImportError:  # pragma: no cover - optional compatibility
    _prune_dead_threads = None  # type: ignore[assignment]
    _track_executor_threads = None  # type: ignore[assignment]


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
    tb = exc.__traceback__
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


def _terminate_processes(processes: list[mp.Process]) -> None:
    """Best-effort shutdown for spawned worker processes."""
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(timeout=1)
    _prune_dead_processes()


def _run_seq_backend(
    *,
    f_wrapped: Callable[[Any], Any],
    items: list[Any],
    total: int,
    desc: str,
    progress: bool,
    func_kwargs: dict[str, Any],
    log_worker: Literal["zero", "first", "all"],
    log_gate_path: Path | None,
    error_handler: ErrorHandlerType,
    error_stats: ErrorStats,
    func_name: str,
    update_pbar_postfix: Callable[[tqdm], None],
) -> list[Any]:
    """Run sequential execution in the current process."""
    results: list[Any] = []
    with tqdm(total=total, desc=desc, disable=not progress, file=sys.stdout) as pbar:
        for idx, item in enumerate(items):
            try:
                result = _call_with_log_control(
                    f_wrapped,
                    item,
                    func_kwargs,
                    log_worker,
                    log_gate_path,
                )
                error_stats.record_success()
                results.append(result)
            except Exception as exc:
                if error_handler == "raise":
                    raise
                error_stats.record_error(idx, exc, item, func_name)
                results.append(None)
            pbar.update(1)
            update_pbar_postfix(pbar)
    _cleanup_log_gate(log_gate_path)
    return results


def _run_threadpool_backend(
    *,
    backend_label: str,
    f_wrapped: Callable[[Any], Any],
    items: list[Any],
    total: int,
    workers: int | None,
    desc: str,
    progress: bool,
    func_kwargs: dict[str, Any],
    log_worker: Literal["zero", "first", "all"],
    log_gate_path: Path | None,
    error_handler: ErrorHandlerType,
    error_stats: ErrorStats,
    func_name: str,
    update_pbar_postfix: Callable[[tqdm], None],
) -> list[Any]:
    """Run the in-process thread pool backend."""
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

    def worker_func(x: Any) -> Any:
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
            total=total, desc=desc, disable=not progress, file=sys.stdout
        ) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                if _track_executor_threads is not None:
                    _track_executor_threads(executor)

                future_to_idx = {
                    executor.submit(worker_func, item): idx
                    for idx, item in enumerate(items)
                }

                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        error_stats.record_success()
                        results[idx] = result
                    except Exception as exc:
                        if error_handler == "raise":
                            for pending in future_to_idx:
                                pending.cancel()
                            _exit_on_worker_error(
                                exc,
                                pbar,
                                caller_info,
                                backend=backend_label,
                            )
                        error_stats.record_error(idx, exc, items[idx], func_name)
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
    event_queue: mp.queues.Queue,
) -> None:
    """Execute one process chunk, optionally with a per-process thread pool."""
    f_wrapped = wrap_dump(func, cache_dir, dump_in_thread)
    child_error_stats = (
        ErrorStats(
            func_name=func_name, max_error_files=max_error_files, write_logs=True
        )
        if error_handler == "log"
        else None
    )
    child_error_count = 0

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
                child_error_count += 1
                if child_error_count <= max_error_files:
                    child_error_stats._write_error_log(  # type: ignore[attr-defined]
                        idx,
                        exc,
                        item,
                        func_name,
                    )
            event_queue.put(("progress", 0, 1))
            if error_handler == "raise":
                event_queue.put(
                    (
                        "fatal",
                        type(exc).__name__,
                        str(exc),
                        _serialize_exception_frames(exc),
                    )
                )
                raise
            return None

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

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
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
    event_queue: mp.queues.Queue,
) -> None:
    """Multiprocessing target wrapper."""
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


def _run_multiprocess_backend(
    *,
    func: Callable[[Any], Any],
    cache_dir: Path | None,
    dump_in_thread: bool,
    items: list[Any],
    total: int,
    num_procs: int,
    num_threads: int,
    desc: str,
    progress: bool,
    func_kwargs: dict[str, Any],
    log_worker: Literal["zero", "first", "all"],
    log_gate_path: Path | None,
    error_handler: ErrorHandlerType,
    error_stats: ErrorStats,
    func_name: str,
    update_pbar_postfix: Callable[[tqdm], None],
    max_error_files: int,
) -> list[Any]:
    """Run the multiprocessing backend with one parent-owned progress bar."""
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

    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    processes: list[mp.Process] = []

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

        tracked: list[psutil.Process] = []
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
            total=total, desc=desc, disable=not progress, file=sys.stdout
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
        _terminate_processes(processes)


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
    backend: str | None = "mp",
    desc: str | None = None,
    dump_in_thread: bool = True,
    log_worker: Literal["zero", "first", "all"] = "first",
    error_handler: ErrorHandlerType = "log",
    max_error_files: int = 100,
    process_update_interval: int | None = None,
    batch: int | None = None,
    ordered: bool = True,
    stop_on_error: bool | None = None,
    **func_kwargs: Any,
) -> list[Any]:
    """Map `func` over `items` using the surviving non-Ray backends."""
    del process_update_interval, batch, ordered

    if stop_on_error is not None:
        warnings.warn(
            "stop_on_error is deprecated, use error_handler instead",
            DeprecationWarning,
            stacklevel=2,
        )
        error_handler = "raise" if stop_on_error else "log"

    if num_threads <= 0:
        raise ValueError("num_threads must be a positive integer")

    if workers is not None:
        warnings.warn(
            "'workers' is deprecated for multi_process; use 'num_procs' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        if num_procs is None:
            num_procs = workers

    if items is None and inputs is not None:
        items = inputs
    if items is None:
        raise ValueError("'items' or 'inputs' must be provided")

    items = list(items)
    if not items:
        return []

    backend = "mp" if backend is None else backend
    if backend == "safe":
        warnings.warn(
            "'safe' backend is deprecated; use 'thread' instead",
            DeprecationWarning,
            stacklevel=2,
        )
        backend = "thread"

    if backend not in {"seq", "mp", "thread"}:
        raise ValueError(f"Unsupported backend: {backend!r}")

    if num_procs is None and backend == "mp":
        num_procs = os.cpu_count() or 1

    cache_dir = _build_cache_dir(func, items) if lazy_output else None
    f_wrapped = wrap_dump(func, cache_dir, dump_in_thread)
    log_gate_path = create_log_gate_path(log_worker)
    total = len(items)
    if desc:
        desc = desc.strip() + f"[{backend}]"
    else:
        desc = f"Multi-process [{backend}]"

    func_name = getattr(func, "__name__", repr(func))
    error_stats = ErrorStats(
        func_name=func_name,
        max_error_files=max_error_files,
        write_logs=error_handler == "log",
    )

    def _update_pbar_postfix(pbar: tqdm) -> None:
        pbar.set_postfix(error_stats.get_postfix_dict())

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

    if backend == "thread":
        safe_workers = num_threads
        if safe_workers == 1 and workers is not None:
            safe_workers = workers
        return _run_threadpool_backend(
            backend_label="thread",
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


__all__ = ["multi_process"]
