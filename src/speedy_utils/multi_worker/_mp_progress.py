"""Progress/reporting helpers for multi_process backends."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .common import ErrorStats


if TYPE_CHECKING:
    import multiprocessing as mp

    from tqdm import tqdm


@dataclass
class MpProgressState:
    started: int = 0
    active: int = 0


def build_progress_desc(
    *,
    desc: str | None,
    mode: str,
    num_procs: int | None = None,
    num_threads: int | None = None,
) -> str:
    """Build a compact progress label with backend topology."""
    base_desc = desc.strip() if desc and desc.strip() else "Multi-worker"

    if mode == "seq":
        return f"{base_desc} [seq]"

    if mode == "thread":
        thread_count = max(1, num_threads or 1)
        return f"{base_desc} [thread: {thread_count}t]"

    if mode == "spawn":
        proc_count = max(1, num_procs or 1)
        return f"{base_desc} [spawn: {proc_count}p]"

    if mode == "hybrid":
        proc_count = max(1, num_procs or 1)
        thread_count = max(1, num_threads or 1)
        return f"{base_desc} [hybrid: {proc_count}p x {thread_count}t]"

    return f"{base_desc} [{mode}]"


def set_progress_postfix(pbar: "tqdm", postfix: dict[str, Any]) -> None:
    """Update tqdm postfix without forcing an immediate redraw."""
    try:
        pbar.set_postfix(postfix, refresh=False)
    except TypeError:
        pbar.set_postfix(postfix)


def refresh_progress_bar(pbar: "tqdm") -> None:
    """Force a visible redraw when tqdm is available."""
    refresh = getattr(pbar, "refresh", None)
    if callable(refresh):
        refresh()


def build_multiprocess_postfix(
    *,
    error_stats: ErrorStats,
    processes: list["mp.Process"],
    state: MpProgressState,
    total_processes: int | None = None,
) -> dict[str, Any]:
    """Return compact parent-owned status for the multiprocessing bar."""
    postfix: dict[str, Any] = error_stats.get_postfix_dict()
    expected_processes = (
        total_processes if total_processes is not None else len(processes)
    )
    if expected_processes:
        live_processes = sum(1 for proc in processes if proc.exitcode is None)
        postfix["proc"] = f"{live_processes}/{expected_processes}"
    postfix["start"] = state.started
    postfix["active"] = state.active
    return postfix


def start_mp_progress_monitor(
    *,
    pbar: "tqdm",
    pbar_lock: threading.Lock,
    stop_event: threading.Event,
    error_stats: ErrorStats,
    processes: list["mp.Process"],
    processes_lock: threading.Lock,
    total_processes: int,
    state: MpProgressState,
    interval: float = 0.1,
) -> threading.Thread:
    """Keep the multiprocessing tqdm visible even while workers warm up."""

    def _monitor() -> None:
        while not stop_event.wait(interval):
            with pbar_lock:
                with processes_lock:
                    processes_snapshot = list(processes)
                set_progress_postfix(
                    pbar,
                    build_multiprocess_postfix(
                        error_stats=error_stats,
                        processes=processes_snapshot,
                        state=state,
                        total_processes=total_processes,
                    ),
                )
                refresh_progress_bar(pbar)

    thread = threading.Thread(
        target=_monitor,
        name="speedy-mp-progress",
        daemon=True,
    )
    thread.start()
    return thread
