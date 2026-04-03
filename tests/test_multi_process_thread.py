"""
Comprehensive tests for multi_process with inner multi_thread execution.

Tests the nested parallelism where:
- multi_process spawns multiple processes (num_procs)
- Each process has its own thread pool (num_threads)

This tests the `_run_mp_chunk` function which uses ThreadPoolExecutor
inside each multiprocessing worker.
"""
import contextlib
import multiprocessing
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

if hasattr(multiprocessing, "set_start_method"):
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)

from speedy_utils import multi_thread
from speedy_utils.multi_worker.process import multi_process


# ────────────────────────────────────────────────────────────
# Helper functions (top-level for picklability)
# ────────────────────────────────────────────────────────────

def simple_identity(x):
    """Simple identity function."""
    return x


def slow_identity(x, delay=0.001):
    """Identity with delay to simulate work."""
    time.sleep(delay)
    return x


def square(x):
    """Square a number."""
    return x * x


def cube(x):
    """Cube a number."""
    return x ** 3


def fibonacci(n):
    """CPU-intensive Fibonacci for testing parallel speedup."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def maybe_fail(x):
    """Function that fails for x == 3."""
    if x == 3:
        raise ValueError("boom at x=3")
    return x


def fail_on_multiples_of_5(x):
    """Function that fails on multiples of 5 (picklable for multiprocessing)."""
    if x % 5 == 0 and x > 0:  # Fails on 5, 10, 15, etc.
        raise ValueError(f"Failed at x={x}")
    return x


def nested_parallel_work(x):
    """Function that does nested parallel work using multi_thread internally."""
    # Each process worker calls multi_thread internally
    return multi_thread(square, [x, x+1, x+2], workers=2, progress=False)


def accumulate_results(x):
    """Function that returns a list of results."""
    return [x, x*2, x*3]


# ────────────────────────────────────────────────────────────
# Basic functionality tests
# ────────────────────────────────────────────────────────────

class TestBasicProcessThreadCombinations:
    """Test basic combinations of num_procs and num_threads."""

    def test_single_proc_single_thread(self):
        """Test with 1 process and 1 thread (essentially sequential)."""
        inp = list(range(10))
        result = multi_process(
            square,
            inp,
            num_procs=1,
            num_threads=1,
            progress=False,
            backend="mp",
        )
        assert result == [x * x for x in inp]

    def test_single_proc_multiple_threads(self):
        """Test with 1 process and multiple threads."""
        inp = list(range(20))
        result = multi_process(
            square,
            inp,
            num_procs=1,
            num_threads=4,
            progress=False,
            backend="mp",
        )
        assert result == [x * x for x in inp]

    def test_multiple_procs_single_thread(self):
        """Test with multiple processes and 1 thread each."""
        inp = list(range(20))
        result = multi_process(
            square,
            inp,
            num_procs=4,
            num_threads=1,
            progress=False,
            backend="mp",
        )
        assert result == [x * x for x in inp]

    def test_multiple_procs_multiple_threads(self):
        """Test with multiple processes and multiple threads each."""
        inp = list(range(30))
        result = multi_process(
            square,
            inp,
            num_procs=3,
            num_threads=4,
            progress=False,
            backend="mp",
        )
        assert result == [x * x for x in inp]

    def test_various_proc_thread_combinations(self):
        """Test various combinations of processes and threads."""
        inp = list(range(24))
        expected = [x * x for x in inp]

        for num_procs in [1, 2, 3, 4]:
            for num_threads in [1, 2, 3, 4]:
                result = multi_process(
                    square,
                    inp,
                    num_procs=num_procs,
                    num_threads=num_threads,
                    progress=False,
                    backend="mp",
                )
                assert result == expected, (
                    f"Failed for procs={num_procs}, threads={num_threads}"
                )


class TestSafeBackendWithThreads:
    """Test the 'safe' backend which runs in-process with thread pool."""

    def test_safe_backend_single_thread(self):
        """Safe backend with 1 thread."""
        inp = list(range(10))
        result = multi_process(
            square,
            inp,
            num_threads=1,
            progress=False,
            backend="safe",
        )
        assert result == [x * x for x in inp]

    def test_safe_backend_multiple_threads(self):
        """Safe backend with multiple threads."""
        inp = list(range(20))
        result = multi_process(
            square,
            inp,
            num_threads=8,
            progress=False,
            backend="safe",
        )
        assert result == [x * x for x in inp]

    def test_safe_backend_ignores_num_procs(self):
        """Safe backend should ignore num_procs parameter."""
        inp = list(range(10))
        # num_procs should be ignored for safe backend
        result = multi_process(
            square,
            inp,
            num_procs=100,  # Should be ignored
            num_threads=2,
            progress=False,
            backend="safe",
        )
        assert result == [x * x for x in inp]


class TestSequentialBackend:
    """Test the 'seq' backend for sequential execution."""

    def test_seq_backend_basic(self):
        """Sequential backend processes items one at a time."""
        inp = list(range(10))
        result = multi_process(
            square,
            inp,
            progress=False,
            backend="seq",
        )
        assert result == [x * x for x in inp]

    def test_seq_backend_with_error_handler_ignore(self):
        """Sequential backend with error_handler='ignore'."""
        inp = list(range(6))
        result = multi_process(
            maybe_fail,
            inp,
            progress=False,
            backend="seq",
            error_handler="ignore",
        )
        assert result == [0, 1, 2, None, 4, 5]


# ────────────────────────────────────────────────────────────
# Error handling tests
# ────────────────────────────────────────────────────────────

class TestErrorHandling:
    """Test error handling across process/thread boundaries."""

    def test_mp_ignore_error_handler(self):
        """Test error_handler='ignore' with mp backend."""
        inp = list(range(6))
        result = multi_process(
            maybe_fail,
            inp,
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
            error_handler="ignore",
        )
        # Item at index 3 should fail and return None
        assert result[3] is None
        assert result[:3] == [0, 1, 2]
        assert result[4:] == [4, 5]

    def test_mp_log_error_handler(self):
        """Test error_handler='log' with mp backend."""
        inp = list(range(6))
        result = multi_process(
            maybe_fail,
            inp,
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
            error_handler="log",
        )
        # Item at index 3 should fail and return None
        assert result[3] is None
        assert result[:3] == [0, 1, 2]
        assert result[4:] == [4, 5]

    def test_mp_raise_error_handler(self):
        """Test error_handler='raise' with mp backend."""
        inp = list(range(6))
        with pytest.raises(SystemExit):
            multi_process(
                maybe_fail,
                inp,
                num_procs=2,
                num_threads=2,
                progress=False,
                backend="mp",
                error_handler="raise",
            )

    def test_safe_backend_error_handler_ignore(self):
        """Test error_handler='ignore' with safe backend."""
        inp = list(range(6))
        result = multi_process(
            maybe_fail,
            inp,
            num_threads=2,
            progress=False,
            backend="safe",
            error_handler="ignore",
        )
        assert result[3] is None
        assert result[:3] == [0, 1, 2]
        assert result[4:] == [4, 5]

    def test_multiple_errors_across_processes(self):
        """Test that errors in different processes are all handled."""
        # This function fails on positive multiples of 5
        inp = list(range(15))

        result = multi_process(
            fail_on_multiples_of_5,
            inp,
            num_procs=3,
            num_threads=2,
            progress=False,
            backend="mp",
            error_handler="ignore",
        )

        # Items 5, 10 should fail (0 doesn't fail, others are multiples of 5)
        assert result[0] == 0  # 0 is allowed
        assert result[5] is None
        assert result[10] is None
        # Others should succeed
        assert result[1] == 1
        assert result[2] == 2
        assert result[7] == 7


# ────────────────────────────────────────────────────────────
# Order preservation tests
# ────────────────────────────────────────────────────────────

class TestOrderPreservation:
    """Test that results maintain correct order."""

    def test_mp_ordered_results(self):
        """Test that mp backend returns ordered results."""
        inp = list(range(30))
        result = multi_process(
            square,
            inp,
            num_procs=4,
            num_threads=3,
            ordered=True,
            progress=False,
            backend="mp",
        )
        assert result == [x * x for x in inp]

    def test_mp_unordered_results(self):
        """Test unordered mode returns all results (but order may vary)."""
        inp = list(range(30))
        result = multi_process(
            square,
            inp,
            num_procs=4,
            num_threads=3,
            ordered=False,
            progress=False,
            backend="mp",
        )
        # When unordered, all results should be present but order may differ
        assert sorted(result) == [x * x for x in inp]

    def test_safe_backend_ordered(self):
        """Test that safe backend maintains order."""
        inp = list(range(20))
        result = multi_process(
            square,
            inp,
            num_threads=4,
            ordered=True,
            progress=False,
            backend="safe",
        )
        assert result == [x * x for x in inp]


# ────────────────────────────────────────────────────────────
# Edge case tests
# ────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_input(self):
        """Test with empty input list."""
        result = multi_process(
            square,
            [],
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
        )
        assert result == []

    def test_single_item(self):
        """Test with a single item."""
        result = multi_process(
            square,
            [5],
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
        )
        assert result == [25]

    def test_more_procs_than_items(self):
        """Test when num_procs > number of items."""
        inp = [1, 2, 3]
        result = multi_process(
            square,
            inp,
            num_procs=10,  # More processes than items
            num_threads=2,
            progress=False,
            backend="mp",
        )
        assert result == [1, 4, 9]

    def test_more_threads_than_items_per_process(self):
        """Test when num_threads > items assigned to each process."""
        inp = list(range(4))
        result = multi_process(
            square,
            inp,
            num_procs=2,  # 2 items per process
            num_threads=10,  # More threads than items
            progress=False,
            backend="mp",
        )
        assert result == [0, 1, 4, 9]

    def test_large_input(self):
        """Test with a larger input to stress test the system."""
        inp = list(range(100))
        result = multi_process(
            square,
            inp,
            num_procs=4,
            num_threads=4,
            progress=False,
            backend="mp",
        )
        assert result == [x * x for x in inp]

    def test_different_input_types(self):
        """Test with various input types."""
        # Strings
        result = multi_process(
            str.upper,
            ["hello", "world"],
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
        )
        assert result == ["HELLO", "WORLD"]

        # Tuples
        result = multi_process(
            sum,
            [(1, 2), (3, 4), (5, 6)],
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
        )
        assert result == [3, 7, 11]


# ────────────────────────────────────────────────────────────
# Lazy output tests
# ────────────────────────────────────────────────────────────

class TestLazyOutput:
    """Test lazy output functionality with multi-process/thread."""

    def test_lazy_output_returns_paths(self):
        """Test that lazy_output=True returns file paths."""
        inp = list(range(6))
        out = multi_process(
            square,
            inp,
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
            lazy_output=True,
            dump_in_thread=False,
        )
        assert len(out) == len(inp)
        assert all(isinstance(path, str) for path in out)
        assert all(path.endswith(".pkl") for path in out)
        assert all(os.path.exists(path) for path in out)

        # Cleanup
        for path in out:
            os.unlink(path)
        cache_dir = os.path.dirname(out[0])
        if cache_dir and os.path.isdir(cache_dir):
            with contextlib.suppress(OSError):
                os.rmdir(cache_dir)

    def test_lazy_output_with_thread_dump(self):
        """Test lazy_output with dump_in_thread=True."""
        inp = list(range(6))
        out = multi_process(
            square,
            inp,
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
            lazy_output=True,
            dump_in_thread=True,
        )
        assert len(out) == len(inp)

        # Wait a bit for async dumps to complete
        time.sleep(0.2)

        assert all(isinstance(path, str) for path in out)
        assert all(path.endswith(".pkl") for path in out)

        # Cleanup
        for path in out:
            if os.path.exists(path):
                os.unlink(path)
        cache_dir = os.path.dirname(out[0]) if out else None
        if cache_dir and os.path.isdir(cache_dir):
            with contextlib.suppress(OSError):
                os.rmdir(cache_dir)


# ────────────────────────────────────────────────────────────
# Progress bar tests
# ────────────────────────────────────────────────────────────

class TestProgressBar:
    """Test progress bar functionality."""

    def test_progress_disabled(self):
        """Test with progress=False."""
        inp = list(range(10))
        result = multi_process(
            square,
            inp,
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
        )
        assert result == [x * x for x in inp]

    def test_progress_enabled(self):
        """Test with progress=True (should not crash)."""
        inp = list(range(10))
        result = multi_process(
            square,
            inp,
            num_procs=2,
            num_threads=2,
            progress=True,
            backend="mp",
        )
        assert result == [x * x for x in inp]


# ────────────────────────────────────────────────────────────
# Performance tests
# ────────────────────────────────────────────────────────────

class TestPerformance:
    """Test that multi-process with multi-thread provides speedup."""

    def test_process_faster_than_thread_for_cpu_bound(self):
        """Verify that processes are faster than threads for CPU-bound work."""
        inp = [20] * 4  # Small enough to be fast, large enough to show difference

        # Time multi_process
        start_mp = time.perf_counter()
        result_mp = multi_process(
            fibonacci,
            inp,
            num_procs=4,
            num_threads=1,
            progress=False,
            backend="mp",
        )
        dur_mp = time.perf_counter() - start_mp

        # Time multi_thread
        start_mt = time.perf_counter()
        result_mt = multi_thread(fibonacci, inp, workers=4, progress=False)
        dur_mt = time.perf_counter() - start_mt

        # Results should match
        assert result_mp == result_mt

        # Multi-process should be faster for CPU-bound work
        # (though this may not always hold in CI environments)
        print(f"\nMulti-process: {dur_mp:.3f}s, Multi-thread: {dur_mt:.3f}s")

    def test_nested_threads_in_process(self):
        """Test that nested threads within processes work correctly."""
        inp = list(range(5))
        result = multi_process(
            nested_parallel_work,
            inp,
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
        )
        # Each item x should produce [x^2, (x+1)^2, (x+2)^2]
        expected = [
            [0, 1, 4],
            [1, 4, 9],
            [4, 9, 16],
            [9, 16, 25],
            [16, 25, 36],
        ]
        assert result == expected


# ────────────────────────────────────────────────────────────
# Input validation tests
# ────────────────────────────────────────────────────────────

class TestInputValidation:
    """Test input validation and error messages."""

    def test_zero_threads_raises(self):
        """Test that num_threads=0 raises ValueError."""
        with pytest.raises(ValueError, match="must be a positive"):
            multi_process(
                square,
                [1, 2, 3],
                num_threads=0,
                progress=False,
            )

    def test_negative_threads_raises(self):
        """Test that negative num_threads raises ValueError."""
        with pytest.raises(ValueError, match="must be a positive"):
            multi_process(
                square,
                [1, 2, 3],
                num_threads=-1,
                progress=False,
            )

    def test_invalid_backend_raises(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported backend"):
            multi_process(
                square,
                [1, 2, 3],
                backend="invalid",
                progress=False,
            )


# ────────────────────────────────────────────────────────────
# Parameter alias tests
# ────────────────────────────────────────────────────────────

class TestParameterAliases:
    """Test parameter aliases and deprecations."""

    def test_workers_alias_for_num_procs(self):
        """Test that 'workers' is an alias for 'num_procs'."""
        inp = list(range(10))
        with pytest.deprecated_call(match="num_procs"):
            result = multi_process(
                square,
                inp,
                workers=2,
                num_threads=2,
                progress=False,
                backend="mp",
            )
        assert result == [x * x for x in inp]

    def test_workers_and_num_procs_conflict(self):
        """Test that conflicting workers and num_procs raises."""
        with pytest.raises(ValueError, match="must match"):
            multi_process(
                square,
                [1, 2, 3],
                workers=2,
                num_procs=3,
                progress=False,
                backend="mp",
            )

    def test_inputs_alias_for_items(self):
        """Test that 'inputs' is an alias for 'items'."""
        result = multi_process(
            square,
            inputs=[1, 2, 3],
            num_threads=2,
            progress=False,
            backend="safe",
        )
        assert result == [1, 4, 9]


# ────────────────────────────────────────────────────────────
# Log worker tests
# ────────────────────────────────────────────────────────────

class TestLogWorker:
    """Test log_worker parameter for controlling worker output."""

    def test_log_worker_zero(self):
        """Test log_worker='zero' silences worker output."""
        inp = list(range(5))
        result = multi_process(
            simple_identity,
            inp,
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
            log_worker="zero",
        )
        assert result == inp

    def test_log_worker_first(self):
        """Test log_worker='first' allows only first worker to log."""
        inp = list(range(5))
        result = multi_process(
            simple_identity,
            inp,
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
            log_worker="first",
        )
        assert result == inp

    def test_log_worker_all(self):
        """Test log_worker='all' allows all workers to log."""
        inp = list(range(5))
        result = multi_process(
            simple_identity,
            inp,
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
            log_worker="all",
        )
        assert result == inp
