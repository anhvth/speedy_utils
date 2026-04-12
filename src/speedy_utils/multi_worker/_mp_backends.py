"""Backend runners and worker entrypoints for multi_process."""

from __future__ import annotations

import concurrent.futures
import inspect
import multiprocessing as mp
import os
import queue
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import psutil
from loguru import logger

from ._mp_progress import (
    MpProgressState,
    build_multiprocess_postfix,
    refresh_progress_bar,
    set_progress_postfix,
    start_mp_progress_monitor,
)
from ._mp_spawn import (
    _deserialize_spawn_callable,
    _serialize_exception_frames,
    patched_spawn_environment,
    prepare_spawn_callable,
    validate_spawn_kwargs,
)
from .common import (
    ErrorHandlerType,
    ErrorStats,
    _call_with_log_control,
    _cleanup_log_gate,
    _display_formatted_error_and_exit,
    _exit_on_worker_error,
    _prune_dead_processes,
    _ThreadLocalStream,
    _track_processes,
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


@dataclass(frozen=True)
class MultiprocessBackendContext:
    backend: BackendRunContext
    func: Callable[[Any], Any]
    cache_dir: Path | None
    dump_in_thread: bool
    num_procs: int
    num_threads: int
    max_error_files: int
    caller_info: dict[str, Any] | None


@dataclass
class _MultiprocessRuntime:
    results: list[Any]
    done_processes: int = 0
    announced_error_log: bool = False
    progress: MpProgressState = field(default_factory=MpProgressState)


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


def build_multiprocess_context(
    *,
    backend: BackendRunContext,
    func: Callable[[Any], Any],
    cache_dir: Path | None,
    dump_in_thread: bool,
    num_procs: int,
    num_threads: int,
    max_error_files: int,
    caller_info: dict[str, Any] | None,
) -> MultiprocessBackendContext:
    return MultiprocessBackendContext(
        backend=backend,
        func=func,
        cache_dir=cache_dir,
        dump_in_thread=dump_in_thread,
        num_procs=num_procs,
        num_threads=num_threads,
        max_error_files=max_error_files,
        caller_info=caller_info,
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


def _snapshot_processes(
    processes: list[mp.Process],
    processes_lock: threading.Lock,
) -> list[mp.Process]:
    with processes_lock:
        return list(processes)


def _sync_multiprocess_pbar(
    *,
    pbar: "tqdm",
    processes: list[mp.Process],
    error_stats: ErrorStats,
    runtime: _MultiprocessRuntime,
    total_processes: int,
) -> None:
    set_progress_postfix(
        pbar,
        build_multiprocess_postfix(
            error_stats=error_stats,
            processes=processes,
            state=runtime.progress,
            total_processes=total_processes,
        ),
    )
    refresh_progress_bar(pbar)


def _track_spawned_processes(processes: list[mp.Process]) -> None:
    tracked: list[psutil.Process] = []
    for proc in processes:
        if proc.pid is None:
            continue
        try:
            tracked.append(psutil.Process(proc.pid))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _track_processes(tracked)


def _spawn_worker_processes(
    *,
    mp_context: Any,
    chunks: list[list[tuple[int, Any]]],
    worker_ctx: MpWorkerContext,
    event_queue: mp.queues.Queue,
    processes: list[mp.Process],
    processes_lock: threading.Lock,
    pbar_lock: threading.Lock,
    sync_pbar: Callable[[], None],
    spawn_done: threading.Event,
) -> None:
    try:
        for chunk in chunks:
            proc = mp_context.Process(
                target=multiprocess_entrypoint,
                args=(chunk, worker_ctx, event_queue),
            )
            proc.start()
            with processes_lock:
                processes.append(proc)  # type: ignore[arg-type]
            with pbar_lock:
                sync_pbar()

        _track_spawned_processes(_snapshot_processes(processes, processes_lock))
    except Exception as exc:
        event_queue.put(
            (
                "spawn_error",
                type(exc).__name__,
                str(exc),
                _serialize_exception_frames(exc),
            )
        )
    finally:
        spawn_done.set()


def _handle_multiprocess_event(
    msg: tuple[Any, ...],
    *,
    runtime: _MultiprocessRuntime,
    error_stats: ErrorStats,
    pbar: "tqdm",
    pbar_lock: threading.Lock,
    sync_pbar: Callable[[], None],
    processes: list[mp.Process],
    processes_lock: threading.Lock,
    spawn_done: threading.Event,
    spawn_thread: threading.Thread,
    caller_info: dict[str, Any] | None,
) -> None:
    tag = msg[0]

    if tag == "result":
        _, idx, result = msg
        runtime.results[idx] = result
        return

    if tag == "started":
        _, started_inc = msg
        with pbar_lock:
            runtime.progress.started += started_inc
            runtime.progress.active += started_inc
            sync_pbar()
        return

    if tag == "progress":
        _, ok_inc, err_inc = msg
        with pbar_lock:
            for _ in range(ok_inc):
                error_stats.record_success()
            bump_error_count(error_stats, err_inc)
            runtime.progress.active = max(
                0,
                runtime.progress.active - (ok_inc + err_inc),
            )
            pbar.update(ok_inc + err_inc)
            sync_pbar()
        return

    if tag == "process_done":
        runtime.done_processes += 1
        with pbar_lock:
            sync_pbar()
        return

    if tag == "error_log":
        _, log_path = msg
        if not runtime.announced_error_log:
            logger.opt(depth=1).warning("Error log: {}", log_path)
            runtime.announced_error_log = True
        return

    if tag in {"fatal", "spawn_error"}:
        _, exc_type_name, exc_msg, frames = msg
        spawn_done.set()
        spawn_thread.join(timeout=1)
        terminate_processes(_snapshot_processes(processes, processes_lock))
        _display_formatted_error_and_exit(
            exc_type_name=exc_type_name,
            exc_msg=exc_msg,
            frames=frames,
            caller_info=caller_info,
            backend="spawn",
            pbar=pbar,
        )


def _consume_multiprocess_events(
    *,
    event_queue: mp.queues.Queue,
    runtime: _MultiprocessRuntime,
    total_processes: int,
    spawn_done: threading.Event,
    spawn_thread: threading.Thread,
    processes: list[mp.Process],
    processes_lock: threading.Lock,
    pbar: "tqdm",
    pbar_lock: threading.Lock,
    sync_pbar: Callable[[], None],
    error_stats: ErrorStats,
    caller_info: dict[str, Any] | None,
) -> None:
    while runtime.done_processes < total_processes:
        try:
            msg = event_queue.get(timeout=0.1)
        except queue.Empty:
            if spawn_done.is_set() and all(
                proc.exitcode is not None
                for proc in _snapshot_processes(processes, processes_lock)
            ):
                runtime.done_processes = total_processes
                with pbar_lock:
                    sync_pbar()
            continue

        _handle_multiprocess_event(
            msg,
            runtime=runtime,
            error_stats=error_stats,
            pbar=pbar,
            pbar_lock=pbar_lock,
            sync_pbar=sync_pbar,
            processes=processes,
            processes_lock=processes_lock,
            spawn_done=spawn_done,
            spawn_thread=spawn_thread,
            caller_info=caller_info,
        )


def run_seq_backend(
    ctx: BackendRunContext,
    *,
    update_pbar_postfix: Callable[["tqdm"], None],
) -> list[Any]:
    """Run sequential execution in the current process."""
    results: list[Any] = []
    try:
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
        return results
    finally:
        _cleanup_log_gate(ctx.log_gate_path)


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
        with (
            ctx.tqdm_cls(
                total=ctx.total,
                desc=ctx.desc,
                disable=not ctx.progress,
                file=sys.stdout,
                dynamic_ncols=True,
            ) as pbar,
            concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor,
        ):
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

        return results
    finally:
        if _prune_dead_threads is not None:
            _prune_dead_threads()
        _cleanup_log_gate(ctx.log_gate_path)
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def run_multiprocess_backend(ctx: MultiprocessBackendContext) -> list[Any]:
    """Run the spawn multiprocessing backend with a parent-owned progress loop."""
    chunks = chunk_indexed_items(list(enumerate(ctx.backend.items)), ctx.num_procs)
    if not chunks:
        _cleanup_log_gate(ctx.backend.log_gate_path)
        return []

    process_func, serialized_func = prepare_spawn_callable(ctx.func)
    validate_spawn_kwargs(ctx.backend.func_kwargs)

    mp_context = mp.get_context("spawn")
    event_queue = mp_context.Queue()
    processes: list[mp.Process] = []
    total_processes = len(chunks)
    worker_ctx = MpWorkerContext(
        func=process_func,
        serialized_func=serialized_func,
        cache_dir=ctx.cache_dir,
        dump_in_thread=ctx.dump_in_thread,
        num_threads=ctx.num_threads,
        func_kwargs=ctx.backend.func_kwargs,
        log_worker=ctx.backend.log_worker,
        log_gate_path=ctx.backend.log_gate_path,
        error_handler=ctx.backend.error_handler,
        max_error_files=ctx.max_error_files,
        func_name=ctx.backend.func_name,
    )

    try:
        with patched_spawn_environment():
            runtime = _MultiprocessRuntime(results=[None] * ctx.backend.total)
            with ctx.backend.tqdm_cls(
                total=ctx.backend.total,
                desc=ctx.backend.desc,
                disable=not ctx.backend.progress,
                file=sys.stdout,
                dynamic_ncols=True,
            ) as pbar:
                pbar_lock = threading.Lock()
                processes_lock = threading.Lock()
                stop_event = threading.Event()
                spawn_done = threading.Event()

                def sync_pbar() -> None:
                    _sync_multiprocess_pbar(
                        pbar=pbar,
                        processes=_snapshot_processes(processes, processes_lock),
                        error_stats=ctx.backend.error_stats,
                        runtime=runtime,
                        total_processes=total_processes,
                    )

                monitor_thread = start_mp_progress_monitor(
                    pbar=pbar,
                    pbar_lock=pbar_lock,
                    stop_event=stop_event,
                    error_stats=ctx.backend.error_stats,
                    processes=processes,
                    processes_lock=processes_lock,
                    total_processes=total_processes,
                    state=runtime.progress,
                )
                spawn_thread = threading.Thread(
                    target=_spawn_worker_processes,
                    kwargs={
                        "mp_context": mp_context,
                        "chunks": chunks,
                        "worker_ctx": worker_ctx,
                        "event_queue": event_queue,
                        "processes": processes,
                        "processes_lock": processes_lock,
                        "pbar_lock": pbar_lock,
                        "sync_pbar": sync_pbar,
                        "spawn_done": spawn_done,
                    },
                    name="speedy-mp-spawn",
                    daemon=True,
                )

                with pbar_lock:
                    sync_pbar()

                try:
                    spawn_thread.start()
                    _consume_multiprocess_events(
                        event_queue=event_queue,
                        runtime=runtime,
                        total_processes=total_processes,
                        spawn_done=spawn_done,
                        spawn_thread=spawn_thread,
                        processes=processes,
                        processes_lock=processes_lock,
                        pbar=pbar,
                        pbar_lock=pbar_lock,
                        sync_pbar=sync_pbar,
                        error_stats=ctx.backend.error_stats,
                        caller_info=ctx.caller_info,
                    )
                    spawn_thread.join(timeout=1)
                    for proc in _snapshot_processes(processes, processes_lock):
                        proc.join()
                finally:
                    stop_event.set()
                    monitor_thread.join(timeout=1)

            return runtime.results
    finally:
        _cleanup_log_gate(ctx.backend.log_gate_path)
        terminate_processes(processes)


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
