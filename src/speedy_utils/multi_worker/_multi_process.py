"""Multi-process dispatcher without Ray support.

This module now keeps the public orchestration readable by delegating the real
subsystems to private siblings:

- spawn/importability support
- progress/reporting helpers
- backend execution loops
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, cast

from tqdm import tqdm

from . import _mp_spawn as _spawn
from ._mp_backends import (
    BackendRunContext,
    MpWorkerContext,
    _caller_info_from_stack,
    build_backend_context,
    build_multiprocess_context,
    bump_error_count,
    chunk_indexed_items,
    multiprocess_entrypoint,
    run_multiprocess_backend,
    run_seq_backend,
    run_threadpool_backend,
    terminate_processes,
)
from ._mp_progress import (
    build_progress_desc,
    set_progress_postfix,
)
from ._mp_spawn import (
    _SpawnFallbackRequired,
)
from .common import (
    ErrorHandlerType,
    ErrorStats,
    _build_cache_dir,
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


@dataclass(frozen=True)
class PreparedRun:
    func: Callable[[Any], Any]
    cache_dir: Path | None
    dump_in_thread: bool
    max_error_files: int
    backend_ctx: BackendRunContext


def _normalize_request(
    *,
    items: Iterable[Any] | None,
    num_procs: int | None,
    num_threads: int,
    backend: str,
) -> NormalizedRequest:
    """Validate the public request near the boundary."""

    if num_threads <= 0:
        raise ValueError("num_threads must be a positive integer")

    if items is None:
        raise ValueError("'items' must be provided")

    normalized_items = list(items)
    if backend not in {"seq", "mp", "thread"}:
        raise ValueError(f"Unsupported backend: {backend!r}")
    normalized_backend = backend

    if num_procs is None and normalized_backend == "mp":
        num_procs = os.cpu_count() or 1

    return NormalizedRequest(
        items=normalized_items,
        backend=cast(BackendName, normalized_backend),
        num_procs=num_procs,
        num_threads=num_threads,
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


def _update_error_postfix(pbar: tqdm, error_stats: ErrorStats) -> None:
    set_progress_postfix(pbar, error_stats.get_postfix_dict())


def _resolve_backend(request: NormalizedRequest) -> BackendName:
    if request.backend == "mp" and (request.num_procs or 1) <= 1:
        return "thread"
    return request.backend


def _run_local_backend(
    *,
    backend: BackendName,
    request: NormalizedRequest,
    prepared: PreparedRun,
    update_pbar_postfix: Callable[[tqdm], None],
) -> list[Any]:
    if backend == "seq":
        return run_seq_backend(
            prepared.backend_ctx,
            update_pbar_postfix=update_pbar_postfix,
        )

    return run_threadpool_backend(
        prepared.backend_ctx,
        backend_label="thread",
        workers=request.num_threads,
        update_pbar_postfix=update_pbar_postfix,
    )


def _run_multiprocess_with_fallback(
    *,
    request: NormalizedRequest,
    prepared: PreparedRun,
    update_pbar_postfix: Callable[[tqdm], None],
) -> list[Any]:
    mp_ctx = build_multiprocess_context(
        backend=prepared.backend_ctx,
        func=prepared.func,
        cache_dir=prepared.cache_dir,
        dump_in_thread=prepared.dump_in_thread,
        num_procs=request.num_procs or 1,
        num_threads=request.num_threads,
        max_error_files=prepared.max_error_files,
        caller_info=_caller_info_from_stack(depth=2),
    )

    try:
        return run_multiprocess_backend(mp_ctx)
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


def multi_process(
    func: Callable[[Any], Any],
    items: Iterable[Any] | None = None,
    *,
    num_procs: int | None = None,
    num_threads: int = 1,
    lazy_output: bool = False,
    progress: bool = True,
    backend: str = "mp",
    desc: str | None = None,
    dump_in_thread: bool = True,
    log_worker: Literal["zero", "first", "all"] = "first",
    error_handler: ErrorHandlerType = "log",
    max_error_files: int = 100,
    **func_kwargs: Any,
) -> list[Any]:
    """Map ``func`` over ``items`` using the surviving non-Ray backends."""
    if os.environ.get("_SPEEDY_MP_CHILD") == "1":
        return []

    request = _normalize_request(
        items=items,
        num_procs=num_procs,
        num_threads=num_threads,
        backend=backend,
    )

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

    backend = _resolve_backend(request)
    if backend != "mp":
        return _run_local_backend(
            backend=backend,
            request=request,
            prepared=prepared,
            update_pbar_postfix=update_pbar_postfix,
        )

    return _run_multiprocess_with_fallback(
        request=request,
        prepared=prepared,
        update_pbar_postfix=update_pbar_postfix,
    )


__all__ = ["multi_process"]
