"""Demonstrate the speedy_utils.parallel API.

This example shows:
- basic scalar inputs
- tuple and dict unpacking
- deduplication of repeated inputs within a run
- cache reuse across repeated runs
- cache cleanup
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from speedy_utils import cleanup_parallel_cache, parallel


def square(x: int) -> int:
    return x * x


def add(a: int, b: int) -> int:
    return a + b


def format_score(name: str, points: int, bonus: int = 0) -> dict[str, Any]:
    total = points + bonus
    return {"name": name, "points": points, "bonus": bonus, "total": total}


def cached_square(x: int) -> int:
    marker_dir = Path(os.environ["PARALLEL_MARKER_DIR"])
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"{uuid.uuid4().hex}_{x}.txt"
    marker_path.write_text(str(x), encoding="utf-8")
    time.sleep(0.15)
    return x * x


def print_section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def demo_basic_usage() -> None:
    print_section("1. Basic Scalar Inputs")
    inputs = list(range(8))
    results = parallel(square, inputs, num_procs=2, num_threads=2, progress=False)
    print(f"inputs:  {inputs}")
    print(f"output:  {results}")


def demo_adaptive_unpacking() -> None:
    print_section("2. Tuple And Dict Unpacking")

    tuple_inputs = [(1, 2), (3, 4), (10, -2)]
    tuple_results = parallel(
        add,
        tuple_inputs,
        num_procs=2,
        num_threads=2,
        progress=False,
    )
    print("tuple inputs:")
    print(tuple_inputs)
    print("tuple results:")
    print(tuple_results)

    dict_inputs = [
        {"name": "Ada", "points": 10, "bonus": 2},
        {"name": "Grace", "points": 8},
        {"name": "Linus", "points": 9, "bonus": 1},
    ]
    dict_results = parallel(
        format_score,
        dict_inputs,
        num_procs=2,
        num_threads=2,
        progress=False,
    )
    print("dict inputs:")
    print(dict_inputs)
    print("dict results:")
    print(dict_results)


def demo_cache_behavior() -> None:
    print_section("3. Deduplication, Cache Reuse, And Cleanup")

    old_tmp_root = os.environ.get("PARALLEL_TMP_ROOT")
    old_marker_dir = os.environ.get("PARALLEL_MARKER_DIR")

    with TemporaryDirectory(prefix="speedy-parallel-demo-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        cache_root = tmp_path / "parallel-cache"
        marker_dir = tmp_path / "markers"

        os.environ["PARALLEL_TMP_ROOT"] = str(cache_root)
        os.environ["PARALLEL_MARKER_DIR"] = str(marker_dir)

        inputs = [1, 1, 2, 3, 3]

        start = time.perf_counter()
        first = parallel(
            cached_square,
            inputs,
            num_procs=2,
            num_threads=2,
            progress=False,
        )
        first_seconds = time.perf_counter() - start
        marker_count_after_first = len(list(marker_dir.iterdir()))

        start = time.perf_counter()
        second = parallel(
            cached_square,
            inputs,
            num_procs=2,
            num_threads=2,
            progress=False,
        )
        second_seconds = time.perf_counter() - start
        marker_count_after_second = len(list(marker_dir.iterdir()))

        print(f"inputs:                 {inputs}")
        print(f"first run output:       {first}")
        print(f"second run output:      {second}")
        print(f"unique inputs:          {len(set(inputs))}")
        print(f"files after first run:  {marker_count_after_first}")
        print(f"files after second run: {marker_count_after_second}")
        print(f"first run seconds:      {first_seconds:.3f}")
        print(f"second run seconds:     {second_seconds:.3f}")

        cleanup_parallel_cache()
        cache_entries_after_cleanup = list(cache_root.iterdir()) if cache_root.exists() else []
        print(f"cache entries after cleanup: {len(cache_entries_after_cleanup)}")

    if old_tmp_root is None:
        os.environ.pop("PARALLEL_TMP_ROOT", None)
    else:
        os.environ["PARALLEL_TMP_ROOT"] = old_tmp_root

    if old_marker_dir is None:
        os.environ.pop("PARALLEL_MARKER_DIR", None)
    else:
        os.environ["PARALLEL_MARKER_DIR"] = old_marker_dir


def main() -> None:
    print("speedy_utils.parallel demo")
    demo_basic_usage()
    demo_adaptive_unpacking()
    demo_cache_behavior()
    print()
    print("Demo completed.")


if __name__ == "__main__":
    main()
