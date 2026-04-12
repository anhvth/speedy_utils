"""Backend runners and worker entrypoints for multi_process."""

from __future__ import annotations

import concurrent.futures
import inspect
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from ._mp_spawn import _deserialize_spawn_callable, _serialize_exception_frames
from .common import (
    ErrorHandlerType,
    ErrorStats,
    _call_with_log_control,
    _cleanup_log_gate,
    _exit_on_worker_error,
    _prune_dead_processes,
    _ThreadLocalStream,
    wrap_dump,
)


try:
    from .thread import _prune_dead_threads, _track_executor_threads
except ImportError:  # pragma: no cover - optional compatibility
    _prune_dead_threads = None  # type: ignore[assignment]
    _track_executor_threads = None  # type: ignore[assignment]


if TYPE_CHECKING:
    from tqdm import tqdm


@dataclass(frozen=True)
class BackendRunContext:
    f_wrapped: Callable[[Any], Any]
    items: list[Any]
    total: int
    desc: str
    progress: bool
    func_kwargs: dict[str, Any]
    log_worker: Literal["zero", "first", "all"]
    log_gate_path: Path | None
    error_handler: ErrorHandlerType
    error_stats: ErrorStats
    func_name: str
    tqdm_cls: type["tqdm"]


@dataclass(frozen=True)
class MpWorkerContext:
    func: Callable[[Any], Any] | None
    serialized_func: bytes | None
    cache_dir: Path | None
    dump_in_thread: bool
    num_threads: int
    func_kwargs: dict[str, Any]
    log_worker: Literal["zero", "first", "all"]
    log_gate_path: Path | None
    error_handler: ErrorHandlerType
    max_error_files: int
    func_name: str


def build_backend_context(
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
    tqdm_cls: type["tqdm"],
) -> BackendRunContext:
    return BackendRunContext(
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
        tqdm_cls=tqdm_cls,
    )


def _caller_info_from_stack(depth: int = 2) -> dict[str, Any] | None:
    frame = inspect.currentframe()
    try:
        for _ in range(depth):
            if frame is None:
                return None
            frame = frame.f_back
        if frame is None:
            return None
        return {
            "filename": frame.f_code.co_filename,
            "lineno": frame.f_lineno,
            "function": frame.f_code.co_name,
        }
    finally:
        del frame


def chunk_indexed_items(
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


def bump_error_count(error_stats: ErrorStats, err_inc: int) -> None:
    if err_inc <= 0:
        return
    with error_stats._lock:  # type: ignore[attr-defined]
        error_stats._error_count += err_inc  # type: ignore[attr-defined]


def terminate_processes(processes: list[mp.Process]) -> None:
    """Best-effort shutdown for spawned worker processes."""
    for proc in processes:
        if proc.is_alive():
            proc.terminate()
    for proc in processes:
        proc.join(timeout=1)
    _prune_dead_processes()


def run_seq_backend(
    ctx: BackendRunContext,
    *,
    update_pbar_postfix: Callable[["tqdm"], None],
) -> list[Any]:
    """Run sequential execution in the current process."""
    results: list[Any] = []
    with ctx.tqdm_cls(
        total=ctx.total,
        desc=ctx.desc,
        disable=not ctx.progress,
        file=sys.stdout,
        dynamic_ncols=True,
    ) as pbar:
        for idx, item in enumerate(ctx.items):
            try:
                result = _call_with_log_control(
                    ctx.f_wrapped,
                    item,
                    ctx.func_kwargs,
                    ctx.log_worker,
                    ctx.log_gate_path,
                )
                ctx.error_stats.record_success()
                results.append(result)
            except Exception as exc:
                if ctx.error_handler == "raise":
                    raise
                ctx.error_stats.record_error(idx, exc, item, ctx.func_name)
                results.append(None)
            pbar.update(1)
            update_pbar_postfix(pbar)
    _cleanup_log_gate(ctx.log_gate_path)
    return results


def run_threadpool_backend(
    ctx: BackendRunContext,
    *,
    backend_label: str,
    workers: int | None,
    update_pbar_postfix: Callable[["tqdm"], None],
) -> list[Any]:
    """Run the in-process thread pool backend."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _ThreadLocalStream(original_stdout)
    sys.stderr = _ThreadLocalStream(original_stderr)
    caller_info = _caller_info_from_stack(depth=2)

    def worker_func(x: Any) -> Any:
        return _call_with_log_control(
            ctx.f_wrapped,
            x,
            ctx.func_kwargs,
            ctx.log_worker,
            ctx.log_gate_path,
        )

    try:
        results: list[Any] = [None] * ctx.total
        with ctx.tqdm_cls(
            total=ctx.total,
            desc=ctx.desc,
            disable=not ctx.progress,
            file=sys.stdout,
            dynamic_ncols=True,
        ) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                if _track_executor_threads is not None:
                    _track_executor_threads(executor)

                future_to_idx = {
                    executor.submit(worker_func, item): idx
                    for idx, item in enumerate(ctx.items)
                }

                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        ctx.error_stats.record_success()
                        results[idx] = result
                    except Exception as exc:
                        if ctx.error_handler == "raise":
                            for pending in future_to_idx:
                                pending.cancel()
                            _exit_on_worker_error(
                                exc,
                                pbar,
                                caller_info,
                                backend=backend_label,
                            )
                        ctx.error_stats.record_error(
                            idx,
                            exc,
                            ctx.items[idx],
                            ctx.func_name,
                        )
                        results[idx] = None
                    pbar.update(1)
                    update_pbar_postfix(pbar)

            if _prune_dead_threads is not None:
                _prune_dead_threads()

        _cleanup_log_gate(ctx.log_gate_path)
        return results
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def _run_mp_chunk(
    *,
    chunk: list[tuple[int, Any]],
    worker_ctx: MpWorkerContext,
    event_queue: mp.queues.Queue,
) -> None:
    """Execute one process chunk, optionally with a per-process thread pool."""
    try:
        func = worker_ctx.func
        if worker_ctx.serialized_func is not None:
            func = _deserialize_spawn_callable(worker_ctx.serialized_func)
        if func is None:
            raise ValueError("func or serialized_func must be provided")
    except Exception:
        event_queue.put(("process_done", os.getpid()))
        raise

    f_wrapped = wrap_dump(func, worker_ctx.cache_dir, worker_ctx.dump_in_thread)
    child_error_stats = (
        ErrorStats(
            func_name=worker_ctx.func_name,
            max_error_files=worker_ctx.max_error_files,
            write_logs=True,
        )
        if worker_ctx.error_handler == "log"
        else None
    )
    child_error_count = 0

    def process_one(payload: tuple[int, Any]) -> Any:
        nonlocal child_error_count
        idx, item = payload
        event_queue.put(("started", 1))
        try:
            result = _call_with_log_control(
                f_wrapped,
                item,
                worker_ctx.func_kwargs,
                worker_ctx.log_worker,
                worker_ctx.log_gate_path,
            )
        except Exception as exc:
            if child_error_stats is not None:
                child_error_count += 1
                if child_error_count <= worker_ctx.max_error_files:
                    log_path = child_error_stats._write_error_log(  # type: ignore[attr-defined]
                        idx,
                        exc,
                        item,
                        worker_ctx.func_name,
                    )
                    if log_path is not None:
                        event_queue.put(("error_log", log_path))
            event_queue.put(("progress", 0, 1))
            if worker_ctx.error_handler == "raise":
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
        if worker_ctx.num_threads <= 1:
            for payload in chunk:
                try:
                    process_one(payload)
                except Exception:
                    if worker_ctx.error_handler == "raise":
                        return
            return

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_ctx.num_threads
        ) as executor:
            futures = [executor.submit(process_one, payload) for payload in chunk]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception:
                    if worker_ctx.error_handler == "raise":
                        for pending in futures:
                            pending.cancel()
                        return
    finally:
        event_queue.put(("process_done", os.getpid()))


def multiprocess_entrypoint(
    chunk: list[tuple[int, Any]],
    worker_ctx: MpWorkerContext,
    event_queue: mp.queues.Queue,
) -> None:
    """Multiprocessing target wrapper."""
    _run_mp_chunk(chunk=chunk, worker_ctx=worker_ctx, event_queue=event_queue)
