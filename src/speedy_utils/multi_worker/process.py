"""
Multi-process map with selectable backends.

This module re-exports from the refactored implementation files:
- common.py: shared utilities (error formatting, log gating, cache, tracking)
- _multi_process.py: sequential + threadpool backends + dispatcher
- _multi_process_ray.py: Ray-specific backend

For backward compatibility, all public symbols are re-exported here.
"""
import warnings
import os

# Suppress Ray FutureWarnings before any imports
warnings.filterwarnings('ignore', category=FutureWarning, module='ray.*')
warnings.filterwarnings(
    'ignore',
    message='.*pynvml.*deprecated.*',
    category=FutureWarning,
)

# Set environment variables before Ray is imported anywhere
os.environ['RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO'] = '0'
os.environ['RAY_DEDUP_LOGS'] = '1'
os.environ['RAY_LOG_TO_STDERR'] = '0'
os.environ['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'

# Re-export public API from common
from .common import (
    ErrorHandlerType,
    ErrorStats,
    SPEEDY_RUNNING_PROCESSES,
    cleanup_phantom_workers,
)

# Re-export main dispatcher
from ._multi_process import multi_process

# Re-export Ray utilities (lazy import to avoid requiring Ray)
try:
    from ._multi_process_ray import ensure_ray, RAY_WORKER
except ImportError:
    ensure_ray = None  # type: ignore[assignment,misc]
    RAY_WORKER = None  # type: ignore[assignment,misc]

# Re-export progress utilities
from .progress import create_progress_tracker, get_ray_progress_actor

# Re-export tqdm for backward compatibility
from tqdm import tqdm


__all__ = [
    'SPEEDY_RUNNING_PROCESSES',
    'ErrorStats',
    'ErrorHandlerType',
    'multi_process',
    'cleanup_phantom_workers',
    'create_progress_tracker',
    'get_ray_progress_actor',
    'ensure_ray',
    'RAY_WORKER',
    'tqdm',
]


def cleanup_phantom_workers():
    """
    Kill all tracked processes and threads (phantom workers) without killing the Jupyter kernel itself.
    Also lists non-daemon threads that remain.
    """
    # Clean up tracked processes first
    _prune_dead_processes()
    killed_processes = 0
    with _SPEEDY_PROCESSES_LOCK:
        for process in SPEEDY_RUNNING_PROCESSES[
            :
        ]:  # Copy to avoid modification during iteration
            try:
                print(f'🔪 Killing tracked process {process.pid} ({process.name()})')
                process.kill()
                killed_processes += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f'⚠️ Could not kill process {process.pid}: {e}')
        SPEEDY_RUNNING_PROCESSES.clear()

    # Also kill any remaining child processes (fallback)
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        try:
            print(f'🔪 Killing child process {child.pid} ({child.name()})')
            child.kill()
        except psutil.NoSuchProcess:
            pass

    # Try to clean up threads using thread module functions if available
    try:
        from .thread import SPEEDY_RUNNING_THREADS, _prune_dead_threads, kill_all_thread

        _prune_dead_threads()
        killed_threads = kill_all_thread()
        if killed_threads > 0:
            print(f'🔪 Killed {killed_threads} tracked threads')
    except ImportError:
        # Fallback: just report stray threads
        for t in threading.enumerate():
            if t is threading.current_thread():
                continue
            if not t.daemon:
                print(f'⚠️ Thread {t.name} is still running (cannot be force-killed).')

    print(
        f'✅ Cleaned up {killed_processes} tracked processes and child processes (kernel untouched).'
    )


# Usage: run this anytime after cancelling a cell


__all__ = [
    'SPEEDY_RUNNING_PROCESSES',
    'ErrorStats',
    'ErrorHandlerType',
    'multi_process',
    'cleanup_phantom_workers',
    'create_progress_tracker',
    'get_ray_progress_actor',
]
