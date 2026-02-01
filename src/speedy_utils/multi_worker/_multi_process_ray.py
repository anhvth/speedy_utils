"""
Ray-specific multi_process backend implementation.

Contains:
- ensure_ray(): Ray initialization/lifecycle management
- run_ray_backend(): Ray-based parallel execution
"""
from __future__ import annotations

import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

from tqdm import tqdm

from .common import (
    ErrorStats,
    _call_with_log_control,
    _cleanup_log_gate,
    _exit_on_ray_error,
    _track_ray_processes,
)
from .progress import ProgressPoller, create_progress_tracker

if TYPE_CHECKING:
    from .common import ErrorHandlerType

# ─── Ray management ─────────────────────────────────────────────

RAY_WORKER = None


def ensure_ray(
    workers: int | None,
    pbar: tqdm | None = None,
    ray_metrics_port: int | None = None,
) -> None:
    """
    Initialize or reinitialize Ray safely for both local and cluster envs.

    1. Tries to connect to an existing cluster (address='auto') first.
    2. If no cluster is found, starts a local Ray instance with 'workers' CPUs.
    """
    import ray as _ray_module

    global RAY_WORKER
    requested_workers = workers
    if workers is None:
        workers = os.cpu_count() or 1

    if ray_metrics_port is not None:
        os.environ['RAY_metrics_export_port'] = str(ray_metrics_port)

    allow_restart = os.environ.get('RESTART_RAY', '0').lower() in ('1', 'true')
    is_cluster_env = (
        'RAY_ADDRESS' in os.environ
        or os.environ.get('RAY_CLUSTER') == '1'
    )

    # 1. Handle existing session
    if _ray_module.is_initialized():
        if not allow_restart:
            if pbar:
                pbar.set_postfix_str('Using existing Ray session')
            return

        # Avoid shutting down shared cluster sessions.
        if is_cluster_env:
            if pbar:
                pbar.set_postfix_str(
                    'Cluster active: skipping restart to protect connection'
                )
            return

        # Local restart: only if worker count changed
        if workers != RAY_WORKER:
            if pbar:
                pbar.set_postfix_str(
                    f'Restarting local Ray with {workers} workers'
                )
            _ray_module.shutdown()
        else:
            return

    # 2. Initialization logic
    t0 = time.time()
    
    # Try to connect to existing cluster FIRST (address="auto")
    try:
        if pbar:
            pbar.set_postfix_str('Searching for Ray cluster...')
        
        # MUST NOT pass num_cpus/num_gpus here to avoid ValueError
        _ray_module.init(
            address='auto', 
            ignore_reinit_error=True, 
            logging_level=logging.ERROR,
            log_to_driver=False
        )
        
        if pbar:
            resources = _ray_module.cluster_resources()
            cpus = resources.get('CPU', 0)
            pbar.set_postfix_str(f'Connected to Ray Cluster ({int(cpus)} CPUs)')
            
    except Exception:
        # 3. Fallback: Start a local Ray session
        if pbar:
            pbar.set_postfix_str(
                f'No cluster found. Starting local Ray ({workers} CPUs)...'
            )
            
        _ray_module.init(
            num_cpus=workers,
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            log_to_driver=False,
        )
        
        if pbar:
            took = time.time() - t0
            pbar.set_postfix_str(f'ray.init local {workers} took {took:.2f}s')

    _track_ray_processes()

    if requested_workers is None:
        try:
            resources = _ray_module.cluster_resources()
            total_cpus = int(resources.get('CPU', 0))
            if total_cpus > 0:
                workers = total_cpus
        except Exception:
            pass

    RAY_WORKER = workers


def run_ray_backend(
    *,
    f_wrapped: Callable,
    items: list[Any],
    total: int,
    workers: int | None,
    progress: bool,
    desc: str,
    func_kwargs: dict[str, Any],
    shared_kwargs: list[str] | None,
    log_worker: Literal['zero', 'first', 'all'],
    log_gate_path: Path | None,
    total_items: int | None,
    poll_interval: float,
    ray_metrics_port: int | None,
    error_handler: ErrorHandlerType,
    error_stats: ErrorStats,
    func_name: str,
) -> list[Any]:
    """
    Run the Ray backend for multi_process.

    Returns a list of results in the same order as items.
    """
    import ray as _ray_module

    # Capture caller frame for better error reporting
    # Go back to multi_process -> user code
    caller_frame = inspect.currentframe()
    caller_info = None
    if (
        caller_frame
        and caller_frame.f_back
        and caller_frame.f_back.f_back
    ):
        outer = caller_frame.f_back.f_back
        caller_info = {
            'filename': outer.f_code.co_filename,
            'lineno': outer.f_lineno,
            'function': outer.f_code.co_name,
        }

    results = []
    gate_path_str = str(log_gate_path) if log_gate_path else None

    # Determine if we're doing item-level or task-level tracking
    use_item_tracking = total_items is not None
    pbar_total = total_items if use_item_tracking else total
    pbar_desc = desc if use_item_tracking else desc

    with tqdm(
        total=pbar_total,
        desc=pbar_desc,
        disable=not progress,
        file=sys.stdout,
        unit='items' if use_item_tracking else 'tasks',
    ) as pbar:
        ensure_ray(workers, pbar, ray_metrics_port)
        
        shared_refs: dict[str, Any] = {}
        regular_kwargs: dict[str, Any] = {}
        
        # Create progress actor for item-level tracking if total_items specified
        progress_actor = None
        progress_poller = None
        if use_item_tracking:
            progress_actor = create_progress_tracker(total_items, pbar_desc or 'Items')
            shared_refs['progress_actor'] = progress_actor

        if shared_kwargs:
            for kw in shared_kwargs:
                # Put large objects in Ray's object store (zero-copy)
                shared_refs[kw] = _ray_module.put(func_kwargs[kw])
                pbar.set_postfix_str(f'ray: shared `{kw}` via object store')

            # Remaining kwargs are regular
            regular_kwargs = {
                k: v for k, v in func_kwargs.items()
                if k not in shared_kwargs
            }
        else:
            regular_kwargs = func_kwargs

        @_ray_module.remote
        def _task(x, shared_refs_dict, regular_kwargs_dict):
            # Dereference shared objects (zero-copy for numpy arrays)
            import ray as _ray_in_task
            from .progress import set_progress_context
            
            gate = Path(gate_path_str) if gate_path_str else None
            dereferenced = {}
            progress_actor_ref = None
            
            for k, v in shared_refs_dict.items():
                if k == 'progress_actor':
                    progress_actor_ref = v
                    # Don't add progress_actor to kwargs - it's for context only
                else:
                    dereferenced[k] = _ray_in_task.get(v)
            
            # Set progress context for this worker thread
            # This allows user code to call report_progress() directly
            if progress_actor_ref is not None:
                set_progress_context(progress_actor_ref)
            
            all_kwargs = {**dereferenced, **regular_kwargs_dict}
            return _call_with_log_control(
                f_wrapped,
                x,
                all_kwargs,
                log_worker,
                gate,
            )

        refs = [
            _task.remote(x, shared_refs, regular_kwargs) for x in items
        ]

        t_start = time.time()
        
        # Start progress poller if item-level tracking enabled
        if use_item_tracking and progress_actor is not None:
            progress_poller = ProgressPoller(progress_actor, pbar, poll_interval)
            progress_poller.start()
        
        for idx, r in enumerate(refs):
            try:
                result = _ray_module.get(r)
                error_stats.record_success()
                results.append(result)
            except _ray_module.exceptions.RayTaskError as e:
                if error_handler == 'raise':
                    if progress_poller is not None:
                        progress_poller.stop()
                    _exit_on_ray_error(e, pbar, caller_info)
                # Extract original error from RayTaskError
                cause = e.cause if hasattr(e, 'cause') else e.__cause__
                original_error = cause if cause else e
                # Pass full RayTaskError for fallback frame extraction
                error_stats.record_error(
                    idx, original_error, items[idx], func_name,
                    ray_task_error=e
                )
                results.append(None)
            
            # Only update progress bar for task-level tracking
            # Item-level tracking is handled by progress_poller
            if not use_item_tracking:
                pbar.update(1)
            # Update pbar with success/error counts
            postfix = error_stats.get_postfix_dict()
            pbar.set_postfix(postfix)
        
        if progress_poller is not None:
            progress_poller.stop()
            
        t_end = time.time()
        item_desc = (
            f'{total_items:,} items' if total_items else f'{total} tasks'
        )
        print(f'Ray processing took {t_end - t_start:.2f}s for {item_desc}')

    _cleanup_log_gate(log_gate_path)
    return results


__all__ = ['ensure_ray', 'run_ray_backend', 'RAY_WORKER']
