"""Runnable demo for speedy_utils.multi_process routing.

This script shows the practical difference between:
- sequential execution
- thread-only execution
- spawn/fork multiprocessing
- hybrid spawn/fork + threads
- error handling

Use `cat $file` to inspect it, then run:

    uv run python notebooks/demo_multi_process.py
"""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any, Callable

from speedy_utils import multi_process


def sleep_and_count(task_id: int, steps: int = 5, delay: float = 0.15) -> str:
    """Pretend work that is easy to visually compare across modes."""
    for step in range(steps):
        time.sleep(delay)
        print(f"[task {task_id}] step {step + 1}/{steps}")
    return f"task-{task_id}"


def maybe_fail(task_id: int) -> str:
    """Small helper to demonstrate error handling."""
    if task_id in {3, 7}:
        raise ValueError(f"boom at task {task_id}")
    time.sleep(0.12)
    return f"ok-{task_id}"


def timed_run(
    title: str,
    func: Callable[[Any], Any],
    items: list[Any],
    **kwargs: Any,
) -> list[Any]:
    """Run a scenario, print elapsed time, and return the results."""
    print(f"\n=== {title} ===")
    print(f"kwargs: {kwargs}")
    start = time.perf_counter()
    results = multi_process(func, items, progress=True, **kwargs)
    elapsed = time.perf_counter() - start
    preview = results[: min(5, len(results))]
    print(f"elapsed: {elapsed:.3f}s")
    print(f"preview: {preview}")
    return results


def main() -> None:
    items = list(range(8))

    print("speedy_utils multi_process demo")
    print("This script is intentionally small and easy to run repeatedly.")

    timed_run(
        "Sequential mode",
        sleep_and_count,
        items[:4],
        num_procs=1,
        num_threads=1,
        backend="spawn",
    )

    timed_run(
        "Thread-only mode",
        sleep_and_count,
        items,
        num_procs=1,
        num_threads=4,
        backend="spawn",
    )

    timed_run(
        "Spawn-only mode",
        sleep_and_count,
        items,
        num_procs=4,
        num_threads=1,
        backend="spawn",
    )

    if "fork" in mp.get_all_start_methods():
        timed_run(
            "Fork-only mode",
            sleep_and_count,
            items,
            num_procs=4,
            num_threads=1,
            backend="fork",
        )

    timed_run(
        "Hybrid mode",
        sleep_and_count,
        items,
        num_procs=2,
        num_threads=2,
        backend="spawn",
    )

    timed_run(
        "Error handling with log mode",
        maybe_fail,
        items,
        num_procs=2,
        num_threads=2,
        backend="spawn",
        error_handler="log",
    )

    print("\nTip: tweak `num_procs` and `num_threads` to feel the routing rules:")
    print("  - 1/1 -> sequential")
    print("  - 1/>1 -> threads")
    print("  - >1/1 -> spawn/fork processes")
    print("  - >1/>1 -> hybrid")


if __name__ == "__main__":
    main()
