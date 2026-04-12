"""Multi-process dispatcher without Ray support.

This module now keeps the public orchestration readable by delegating the real
subsystems to private siblings:

- spawn/importability support
- progress/reporting helpers
- backend execution loops
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import sys
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, cast

import psutil
from loguru import logger
from tqdm import tqdm

from . import _mp_spawn as _spawn
from ._mp_backends import (
    MpWorkerContext,
    _caller_info_from_stack,
    build_backend_context,
    bump_error_count,
    chunk_indexed_items,
    multiprocess_entrypoint,
    run_seq_backend,
    run_threadpool_backend,
    terminate_processes,
)
from ._mp_progress import (
    MpProgressState,
    build_multiprocess_postfix,
    build_progress_desc,
    refresh_progress_bar,
    set_progress_postfix,
    start_mp_progress_monitor,
)
from ._mp_spawn import (
    _serialize_exception_frames,
    _SpawnFallbackRequired,
    patched_spawn_environment,
    prepare_spawn_callable,
    validate_spawn_kwargs,
)
from .common import (
    ErrorHandlerType,
    ErrorStats,
    _build_cache_dir,
    _cleanup_log_gate,
    _display_formatted_error_and_exit,
    _track_processes,
    create_log_gate_path,
    wrap_dump,
)


BackendName = Literal["seq", "mp", "thread"]

_deserialize_spawn_callable = _spawn._deserialize_spawn_callable
_infer_importable_module = _spawn._infer_importable_module
_serialize_spawn_callable = _spawn._serialize_spawn_callable


@dataclass(frozen=True)
class NormalizedRequest:
    items: list[Any]
    backend: BackendName
    num_procs: int | None
    num_threads: int
    workers: int | None


@dataclass(frozen=True)
class PreparedRun:
    func: Callable[[Any], Any]
    cache_dir: Path | None
    dump_in_thread: bool
    max_error_files: int
    backend_ctx: Any


def _normalize_request(
    *,
    items: Iterable[Any] | None,
    inputs: Iterable[Any] | None,
    workers: int | None,
    num_procs: int | None,
    num_threads: int,
    backend: str | None,
    stop_on_error: bool | None,
) -> tuple[NormalizedRequest, ErrorHandlerType | None]:
    """Normalize API aliases and compatibility shims near the boundary."""
    compat_error_handler: ErrorHandlerType | None = None

    if stop_on_error is not None:
        warnings.warn(
            "stop_on_error is deprecated, use error_handler instead",
            DeprecationWarning,
            stacklevel=3,
        )
        compat_error_handler = "raise" if stop_on_error else "log"

    if num_threads <= 0:
        raise ValueError("num_threads must be a positive integer")

    if workers is not None:
        warnings.warn(
            "'workers' is deprecated for multi_process; use 'num_procs' instead",
            DeprecationWarning,
            stacklevel=3,
        )
        if num_procs is None:
            num_procs = workers

    if items is None and inputs is not None:
        items = inputs
    if items is None:
        raise ValueError("'items' or 'inputs' must be provided")

    normalized_items = list(items)
    normalized_backend = "mp" if backend is None else backend
    if normalized_backend == "safe":
        warnings.warn(
            "'safe' backend is deprecated; use 'thread' instead",
            DeprecationWarning,
            stacklevel=3,
        )
        normalized_backend = "thread"

    if normalized_backend not in {"seq", "mp", "thread"}:
        raise ValueError(f"Unsupported backend: {normalized_backend!r}")

    if num_procs is None and normalized_backend == "mp":
        num_procs = os.cpu_count() or 1

    typed_backend = cast(BackendName, normalized_backend)

    return (
        NormalizedRequest(
            items=normalized_items,
            backend=typed_backend,
            num_procs=num_procs,
            num_threads=num_threads,
            workers=workers,
        ),
        compat_error_handler,
    )


def _prepare_run(
    *,
    func: Callable[[Any], Any],
    request: NormalizedRequest,
    lazy_output: bool,
    progress: bool,
    desc: str | None,
    dump_in_thread: bool,
    log_worker: Literal["zero", "first", "all"],
    error_handler: ErrorHandlerType,
    max_error_files: int,
    func_kwargs: dict[str, Any],
) -> PreparedRun:
    """Create the shared backend context once after normalization."""
    cache_dir = _build_cache_dir(func, request.items) if lazy_output else None
    f_wrapped = wrap_dump(func, cache_dir, dump_in_thread)
    log_gate_path = create_log_gate_path(log_worker)
    func_name = getattr(func, "__name__", repr(func))
    error_stats = ErrorStats(
        func_name=func_name,
        max_error_files=max_error_files,
        write_logs=error_handler == "log",
    )
    backend_ctx = build_backend_context(
        f_wrapped=f_wrapped,
        items=request.items,
        total=len(request.items),
        desc=build_progress_desc(
            desc=desc,
            backend=request.backend,
            num_procs=request.num_procs,
            num_threads=request.num_threads,
            workers=request.workers,
        ),
        progress=progress,
        func_kwargs=func_kwargs,
        log_worker=log_worker,
        log_gate_path=log_gate_path,
        error_handler=error_handler,
        error_stats=error_stats,
        func_name=func_name,
        tqdm_cls=tqdm,
    )
    return PreparedRun(
        func=func,
        cache_dir=cache_dir,
        dump_in_thread=dump_in_thread,
        max_error_files=max_error_files,
        backend_ctx=backend_ctx,
    )


def _thread_workers_for_request(request: NormalizedRequest) -> int:
    workers = request.num_threads
    if workers == 1 and request.workers is not None:
        return request.workers
    return workers


def _update_error_postfix(pbar: tqdm, error_stats: ErrorStats) -> None:
    set_progress_postfix(pbar, error_stats.get_postfix_dict())


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
    max_error_files: int,
) -> list[Any]:
    """Run the multiprocessing backend with one parent-owned progress bar."""
    caller_info = _caller_info_from_stack(depth=2)
    chunks = chunk_indexed_items(list(enumerate(items)), num_procs)
    if not chunks:
        _cleanup_log_gate(log_gate_path)
        return []

    process_func, serialized_func = prepare_spawn_callable(func)
    validate_spawn_kwargs(func_kwargs)

    ctx = mp.get_context("spawn")
    event_queue = ctx.Queue()
    processes: list[mp.Process] = []
    total_processes = len(chunks)
    worker_ctx = MpWorkerContext(
        func=process_func,
        serialized_func=serialized_func,
        cache_dir=cache_dir,
        dump_in_thread=dump_in_thread,
        num_threads=num_threads,
        func_kwargs=func_kwargs,
        log_worker=log_worker,
        log_gate_path=log_gate_path,
        error_handler=error_handler,
        max_error_files=max_error_files,
        func_name=func_name,
    )

    try:
        with patched_spawn_environment():
            results: list[Any] = [None] * total
            done_processes = 0
            announced_error_log = False
            state = MpProgressState()

            with tqdm(
                total=total,
                desc=desc,
                disable=not progress,
                file=sys.stdout,
                dynamic_ncols=True,
            ) as pbar:
                pbar_lock = threading.Lock()
                processes_lock = threading.Lock()
                stop_event = threading.Event()
                spawn_done = threading.Event()

                def _process_snapshot() -> list[mp.Process]:
                    with processes_lock:
                        return list(processes)

                def _sync_pbar() -> None:
                    set_progress_postfix(
                        pbar,
                        build_multiprocess_postfix(
                            error_stats=error_stats,
                            processes=_process_snapshot(),
                            state=state,
                            total_processes=total_processes,
                        ),
                    )
                    refresh_progress_bar(pbar)

                monitor_thread = start_mp_progress_monitor(
                    pbar=pbar,
                    pbar_lock=pbar_lock,
                    stop_event=stop_event,
                    error_stats=error_stats,
                    processes=processes,
                    processes_lock=processes_lock,
                    total_processes=total_processes,
                    state=state,
                )

                def _spawn_processes() -> None:
                    try:
                        for chunk in chunks:
                            proc = ctx.Process(
                                target=multiprocess_entrypoint,
                                args=(chunk, worker_ctx, event_queue),
                            )
                            proc.start()
                            with processes_lock:
                                processes.append(proc)  # type: ignore[arg-type]
                            with pbar_lock:
                                _sync_pbar()

                        tracked: list[psutil.Process] = []
                        for proc in _process_snapshot():
                            if proc.pid is None:
                                continue
                            try:
                                tracked.append(psutil.Process(proc.pid))
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                continue
                        _track_processes(tracked)
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

                spawn_thread = threading.Thread(
                    target=_spawn_processes,
                    name="speedy-mp-spawn",
                    daemon=True,
                )

                with pbar_lock:
                    _sync_pbar()

                try:
                    spawn_thread.start()

                    while done_processes < total_processes:
                        try:
                            msg = event_queue.get(timeout=0.1)
                        except queue.Empty:
                            if spawn_done.is_set() and all(
                                proc.exitcode is not None for proc in _process_snapshot()
                            ):
                                done_processes = total_processes
                            continue

                        tag = msg[0]
                        if tag == "result":
                            _, idx, result = msg
                            results[idx] = result
                            continue

                        if tag == "started":
                            _, started_inc = msg
                            with pbar_lock:
                                state.started += started_inc
                                state.active += started_inc
                                _sync_pbar()
                            continue

                        if tag == "progress":
                            _, ok_inc, err_inc = msg
                            with pbar_lock:
                                for _ in range(ok_inc):
                                    error_stats.record_success()
                                bump_error_count(error_stats, err_inc)
                                state.active = max(
                                    0,
                                    state.active - (ok_inc + err_inc),
                                )
                                pbar.update(ok_inc + err_inc)
                                _sync_pbar()
                            continue

                        if tag == "process_done":
                            done_processes += 1
                            with pbar_lock:
                                _sync_pbar()
                            continue

                        if tag in {"fatal", "spawn_error"}:
                            _, exc_type_name, exc_msg, frames = msg
                            spawn_done.set()
                            spawn_thread.join(timeout=1)
                            terminate_processes(processes)
                            _display_formatted_error_and_exit(
                                exc_type_name=exc_type_name,
                                exc_msg=exc_msg,
                                frames=frames,
                                caller_info=caller_info,
                                backend="mp",
                                pbar=pbar,
                            )

                        if tag == "error_log":
                            _, log_path = msg
                            if not announced_error_log:
                                logger.opt(depth=1).warning("Error log: {}", log_path)
                                announced_error_log = True
                            continue

                    spawn_thread.join(timeout=1)
                    for proc in _process_snapshot():
                        proc.join()
                finally:
                    stop_event.set()
                    monitor_thread.join(timeout=1)

            return results
    finally:
        _cleanup_log_gate(log_gate_path)
        terminate_processes(processes)


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
    """Map ``func`` over ``items`` using the surviving non-Ray backends."""
    if os.environ.get("_SPEEDY_MP_CHILD") == "1":
        return []
    del process_update_interval, batch, ordered

    request, compat_error_handler = _normalize_request(
        items=items,
        inputs=inputs,
        workers=workers,
        num_procs=num_procs,
        num_threads=num_threads,
        backend=backend,
        stop_on_error=stop_on_error,
    )
    if compat_error_handler is not None:
        error_handler = compat_error_handler

    if not request.items:
        return []

    prepared = _prepare_run(
        func=func,
        request=request,
        lazy_output=lazy_output,
        progress=progress,
        desc=desc,
        dump_in_thread=dump_in_thread,
        log_worker=log_worker,
        error_handler=error_handler,
        max_error_files=max_error_files,
        func_kwargs=func_kwargs,
    )

    def update_pbar_postfix(pbar: tqdm) -> None:
        _update_error_postfix(pbar, prepared.backend_ctx.error_stats)

    if request.backend == "seq":
        return run_seq_backend(
            prepared.backend_ctx,
            update_pbar_postfix=update_pbar_postfix,
        )

    if request.backend == "thread":
        return run_threadpool_backend(
            prepared.backend_ctx,
            backend_label="thread",
            workers=_thread_workers_for_request(request),
            update_pbar_postfix=update_pbar_postfix,
        )

    if (request.num_procs or 1) <= 1:
        return run_threadpool_backend(
            prepared.backend_ctx,
            backend_label="thread",
            workers=request.num_threads,
            update_pbar_postfix=update_pbar_postfix,
        )

    try:
        return _run_multiprocess_backend(
            func=prepared.func,
            cache_dir=prepared.cache_dir,
            dump_in_thread=prepared.dump_in_thread,
            items=prepared.backend_ctx.items,
            total=prepared.backend_ctx.total,
            num_procs=request.num_procs or 1,
            num_threads=request.num_threads,
            desc=prepared.backend_ctx.desc,
            progress=prepared.backend_ctx.progress,
            func_kwargs=prepared.backend_ctx.func_kwargs,
            log_worker=prepared.backend_ctx.log_worker,
            log_gate_path=prepared.backend_ctx.log_gate_path,
            error_handler=prepared.backend_ctx.error_handler,
            error_stats=prepared.backend_ctx.error_stats,
            func_name=prepared.backend_ctx.func_name,
            max_error_files=prepared.max_error_files,
        )
    except _SpawnFallbackRequired as exc:
        warnings.warn(
            (
                "Falling back to thread backend because multiprocessing spawn "
                f"payload is not serializable ({exc})."
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        return run_threadpool_backend(
            prepared.backend_ctx,
            backend_label="thread",
            workers=request.num_threads,
            update_pbar_postfix=update_pbar_postfix,
        )


__all__ = ["multi_process"]
