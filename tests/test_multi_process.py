"""
Smoke tests for speedy_utils.multi_worker.process.multi_process.

This file keeps the current backend behavior covered without relying on
deprecated aliases or the deprecated safe backend.
"""

from __future__ import annotations

import contextlib
import os

import pytest

from speedy_utils import multi_thread
from speedy_utils.multi_worker.process import multi_process


def simple_identity(x):
    return x


def square(x):
    return x * x


def maybe_fail(x):
    if x == 3:
        raise ValueError("boom at x=3")
    return x


def fail_on_multiples_of_5(x):
    if x % 5 == 0:
        raise ValueError(f"bad item: {x}")
    return x


def nested_parallel_work(x):
    return multi_thread(lambda y: y * y, [x, x + 1, x + 2], workers=2, progress=False)


def _cleanup_paths(paths: list[str]) -> None:
    for path in paths:
        if os.path.exists(path):
            os.unlink(path)

    if paths:
        cache_dir = os.path.dirname(paths[0])
        if cache_dir and os.path.isdir(cache_dir):
            with contextlib.suppress(OSError):
                os.rmdir(cache_dir)


@pytest.mark.parametrize("num_threads", [1, 2])
def test_thread_backend_basic_square(num_threads):
    result = multi_process(
        square,
        [0, 1, 2, 3],
        num_threads=num_threads,
        progress=False,
        backend="thread",
    )
    assert result == [0, 1, 4, 9]


def test_seq_backend_error_handler_ignore():
    result = multi_process(
        maybe_fail,
        list(range(6)),
        progress=False,
        backend="seq",
        error_handler="ignore",
    )
    assert result == [0, 1, 2, None, 4, 5]


def test_thread_backend_error_handler_ignore():
    result = multi_process(
        maybe_fail,
        list(range(6)),
        num_threads=2,
        progress=False,
        backend="thread",
        error_handler="ignore",
    )
    assert result == [0, 1, 2, None, 4, 5]


def test_thread_ordered_results():
    result = multi_process(
        square,
        list(range(20)),
        num_threads=3,
        progress=False,
        backend="thread",
    )
    assert result == [x * x for x in range(20)]


def test_mp_lazy_output_returns_paths():
    out = multi_process(
        square,
        list(range(6)),
        num_threads=2,
        progress=False,
        backend="thread",
        lazy_output=True,
        dump_in_thread=False,
    )
    assert len(out) == 6
    assert all(isinstance(path, str) for path in out)
    assert all(path.endswith(".pkl") for path in out)
    assert all(os.path.exists(path) for path in out)
    _cleanup_paths(out)


def test_nested_threads():
    result = multi_process(
        nested_parallel_work,
        list(range(5)),
        num_threads=2,
        progress=False,
        backend="thread",
    )
    assert result == [
        [0, 1, 4],
        [1, 4, 9],
        [4, 9, 16],
        [9, 16, 25],
        [16, 25, 36],
    ]


def test_zero_threads_raises():
    with pytest.raises(ValueError, match="must be a positive"):
        multi_process(square, [1, 2, 3], num_threads=0, progress=False)


def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="Unsupported backend"):
        multi_process(square, [1, 2, 3], backend="invalid", progress=False)
