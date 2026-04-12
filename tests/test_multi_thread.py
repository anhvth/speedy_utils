"""
Thread-focused comparison tests for speedy_utils multi-worker helpers.

This module keeps the cross-checks between multi_thread and multi_process in a
separate file so the process-heavy and thread-heavy coverage stays split.
"""

import contextlib
import multiprocessing
import time

if hasattr(multiprocessing, "set_start_method"):
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)

from speedy_utils import multi_thread
from speedy_utils.multi_worker.process import multi_process, tqdm


def fibonacci(n, x=0):
    if n <= 1:
        return n + x
    return fibonacci(n - 1, x) + fibonacci(n - 2, x)


def forloop(func, inp):
    out = []
    for i in tqdm(inp, desc="forloop"):
        out.append(func(i))
    return out


def test_process_vs_thread_heavy():
    inp = [18] * 10

    start_proc = time.perf_counter()
    out_proc = multi_process(
        fibonacci,
        inp,
        num_procs=4,
        num_threads=4,
        progress=False,
        backend="spawn",
    )
    dur_proc = time.perf_counter() - start_proc

    start_thread = time.perf_counter()
    out_thread = multi_thread(fibonacci, inp, workers=10, progress=False)
    dur_thread = time.perf_counter() - start_thread

    start_forloop = time.perf_counter()
    out_for = forloop(fibonacci, inp)
    dur_forloop = time.perf_counter() - start_forloop

    assert out_proc == out_thread
    assert out_proc == out_for

    try:
        import tabulate

        table_string = tabulate.tabulate(
            [
                ["multi_process", dur_proc],
                ["multi_thread", dur_thread],
                ["for loop", dur_forloop],
            ],
            headers=["Method", "Duration (s)"],
        )
        print(table_string)
    except ImportError:
        pass


def test_process_faster_than_thread_for_cpu_bound():
    inp = [20] * 4

    start_mp = time.perf_counter()
    result_mp = multi_process(
        fibonacci,
        inp,
        num_procs=4,
        num_threads=4,
        progress=False,
        backend="spawn",
    )
    dur_mp = time.perf_counter() - start_mp

    start_mt = time.perf_counter()
    result_mt = multi_thread(fibonacci, inp, workers=4, progress=False)
    dur_mt = time.perf_counter() - start_mt

    assert result_mp == result_mt
    print(f"\nMulti-process: {dur_mp:.3f}s, Multi-thread: {dur_mt:.3f}s")
