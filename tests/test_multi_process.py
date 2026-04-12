"""
Smoke tests for speedy_utils.multi_worker.process.multi_process.

This file keeps the current backend behavior covered without relying on
deprecated aliases or the deprecated safe backend.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import ssl
import warnings

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


def passthrough_with_client(item, client=None):
    return item


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
def test_spawn_backend_basic_square(num_threads):
    result = multi_process(
        square,
        [0, 1, 2, 3],
        num_procs=1,
        num_threads=num_threads,
        progress=False,
        backend="spawn",
    )
    assert result == [0, 1, 4, 9]


def test_sequential_backend_error_handler_ignore():
    result = multi_process(
        maybe_fail,
        list(range(6)),
        num_procs=1,
        progress=False,
        backend="spawn",
        error_handler="ignore",
    )
    assert result == [0, 1, 2, None, 4, 5]


def test_thread_backend_error_handler_ignore():
    result = multi_process(
        maybe_fail,
        list(range(6)),
        num_procs=1,
        num_threads=2,
        progress=False,
        backend="spawn",
        error_handler="ignore",
    )
    assert result == [0, 1, 2, None, 4, 5]


def test_thread_ordered_results():
    result = multi_process(
        square,
        list(range(20)),
        num_procs=1,
        num_threads=3,
        progress=False,
        backend="spawn",
    )
    assert result == [x * x for x in range(20)]


def test_thread_lazy_output_returns_paths():
    out = multi_process(
        square,
        list(range(6)),
        num_procs=1,
        num_threads=2,
        progress=False,
        backend="spawn",
        lazy_output=True,
        dump_in_thread=False,
    )
    assert len(out) == 6
    assert all(isinstance(path, str) for path in out)
    assert all(path.endswith(".pkl") for path in out)
    assert all(os.path.exists(path) for path in out)
    _cleanup_paths(out)


FORK_SUPPORTED = "fork" in mp.get_all_start_methods()


@pytest.mark.skipif(not FORK_SUPPORTED, reason="fork backend unavailable")
def test_fork_backend_basic_square():
    result = multi_process(
        square,
        [0, 1, 2, 3],
        num_procs=2,
        progress=False,
        backend="fork",
    )
    assert result == [0, 1, 4, 9]


@pytest.mark.skipif(not FORK_SUPPORTED, reason="fork backend unavailable")
def test_fork_backend_accepts_unpicklable_kwargs_without_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = multi_process(
            passthrough_with_client,
            [1, 2, 3],
            num_procs=2,
            progress=False,
            backend="fork",
            client=ssl.create_default_context(),
        )

    assert result == [1, 2, 3]
    assert caught == []


def test_nested_threads():
    result = multi_process(
        nested_parallel_work,
        list(range(5)),
        num_procs=1,
        num_threads=2,
        progress=False,
        backend="spawn",
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


@pytest.mark.parametrize(
    ("backend", "match"),
    [
        ("mp", "backend='mp' was removed"),
        ("thread", "backend='thread' was removed"),
        ("seq", "backend='seq' was removed"),
        ("invalid", "Unsupported backend"),
    ],
)
def test_invalid_backend_raises(backend, match):
    with pytest.raises(ValueError, match=match):
        multi_process(square, [1, 2, 3], backend=backend, progress=False)
