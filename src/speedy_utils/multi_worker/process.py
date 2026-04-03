"""Compatibility wrapper for the surviving multi-process APIs."""

from __future__ import annotations

import os
import threading

import psutil
from tqdm import tqdm

from ._multi_process import multi_process
from .common import (
    _SPEEDY_PROCESSES_LOCK,
    SPEEDY_RUNNING_PROCESSES,
    ErrorHandlerType,
    ErrorStats,
    _prune_dead_processes,
)


__all__ = [
    "SPEEDY_RUNNING_PROCESSES",
    "ErrorStats",
    "ErrorHandlerType",
    "multi_process",
    "cleanup_phantom_workers",
    "tqdm",
]


def cleanup_phantom_workers() -> None:
    """
    Kill all tracked processes and threads without touching the Jupyter kernel.
    """
    _prune_dead_processes()
    killed_processes = 0
    with _SPEEDY_PROCESSES_LOCK:
        for process in SPEEDY_RUNNING_PROCESSES[:]:
            try:
                print(f"🔪 Killing tracked process {process.pid} ({process.name()})")
                process.kill()
                killed_processes += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f"⚠️ Could not kill process {process.pid}: {e}")
        SPEEDY_RUNNING_PROCESSES.clear()

    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        try:
            print(f"🔪 Killing child process {child.pid} ({child.name()})")
            child.kill()
        except psutil.NoSuchProcess:
            pass

    try:
        from .thread import _prune_dead_threads, kill_all_thread

        _prune_dead_threads()
        killed_threads = kill_all_thread()
        if killed_threads > 0:
            print(f"🔪 Killed {killed_threads} tracked threads")
    except ImportError:
        for t in threading.enumerate():
            if t is threading.current_thread():
                continue
            if not t.daemon:
                print(f"⚠️ Thread {t.name} is still running (cannot be force-killed).")

    print(
        f"✅ Cleaned up {killed_processes} tracked processes and child processes (kernel untouched)."
    )
