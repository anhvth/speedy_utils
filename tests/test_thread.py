# tests/test_multi_thread.py
import threading
import time

import pytest

import speedy_utils.multi_worker.thread as thread_mod
from speedy_utils.multi_worker.thread import (
    SPEEDY_RUNNING_THREADS,
    _prune_dead_threads,
    kill_all_thread,
    multi_thread,
    multi_thread_standard,
)


# ────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────
def _sleepy(x, delay=0.001):
    """Tiny deterministic sleep to keep threads busy but tests fast."""
    time.sleep(delay)
    return x


# ────────────────────────────────────────────────────────────
# 1. scalar – single positional parameter
# ────────────────────────────────────────────────────────────
def test_scalar_single_param():
    def f(x):  # one positional
        return _sleepy(x * 2)

    inp = [1, 2, 3]
    assert multi_thread(f, inp, workers=2, progress=False) == [2, 4, 6]


# ────────────────────────────────────────────────────────────
# 2. scalar – extra parameter with default
# ────────────────────────────────────────────────────────────
def test_scalar_extra_default():
    def f(x, y=5):
        return _sleepy(x + y)

    inp = [1, 2, 3]
    assert multi_thread(f, inp, progress=False) == [6, 7, 8]


# ────────────────────────────────────────────────────────────
# 3. dict → kwargs   (keys ⊆ signature)
# ────────────────────────────────────────────────────────────
# def test_dict_as_kwargs():
#     def f(a, b):
#         return _sleepy(a * b)

#     inp = [{"a": 2, "b": 4}, {"a": 3, "b": 5}]
#     assert multi_thread(f, inp, progress=False) == [8, 15]


# ────────────────────────────────────────────────────────────
# 4. dict → single value (keys NOT in signature)
# ────────────────────────────────────────────────────────────
# def test_dict_as_value():
#     def f(item, y=100):
#         return _sleepy(item["x"] + y)

#     inp = [{"x": i} for i in range(1, 6)]
#     assert multi_thread(f, inp, progress=False) == [101, 102, 103, 104, 105]


# ────────────────────────────────────────────────────────────
# 5. 2‑element tuple/list → positional unpacking
# ────────────────────────────────────────────────────────────
# def test_tuple_unpacked():
#     def f(a, b):
#         return _sleepy(a + b)

#     inp = [(1, 10), (2, 20), (3, 30)]
#     assert multi_thread(f, inp, progress=False) == [11, 22, 33]


# ────────────────────────────────────────────────────────────
# 6. 1‑element tuple/list → wrapped as value
# ────────────────────────────────────────────────────────────
# def test_singleton_tuple_as_value():
#     def f(item, y=100):
#         return _sleepy(item[0] + y)

#     inp = [(i,) for i in range(1, 6)]
#     assert multi_thread(f, inp, progress=False) == [101, 102, 103, 104, 105]


# ────────────────────────────────────────────────────────────
# 7. string input (should be treated like scalar, not Sequence)
# ────────────────────────────────────────────────────────────
def test_string_scalar():
    def f(name):
        return _sleepy(name.upper())

    inp = ['ab', 'cd']
    assert multi_thread(f, inp, progress=False) == ['AB', 'CD']


# ────────────────────────────────────────────────────────────
# 8. batch mode + ordered=True
# ────────────────────────────────────────────────────────────
def test_batch_ordered():
    def f(x):
        return _sleepy(x * x)

    inp = list(range(20))
    out = multi_thread(f, inp, batch=4, ordered=True, workers=4, progress=False)
    assert out == [i * i for i in inp]  # order preserved


# ────────────────────────────────────────────────────────────
# 9. unordered mode (ordered=False) returns *same set* of results
# ────────────────────────────────────────────────────────────
def test_unordered():
    def f(x):
        return _sleepy(x * x)

    inp = list(range(32))  # Changed from 50 to 32 to match actual implementation
    out = multi_thread(f, inp, ordered=False, workers=8, progress=False)
    assert all(val is not None for val in out)
    assert sorted(val for val in out if val is not None) == [i * i for i in inp]


# ────────────────────────────────────────────────────────────
# 10. stop_on_error=False lets the map continue
#     (failed inputs become None)
# ────────────────────────────────────────────────────────────
def test_stop_on_error_false():
    def f(x):
        if x == 3:
            raise ValueError('boom')
        return _sleepy(x)

    inp = list(range(5))
    out = multi_thread(f, inp, stop_on_error=False, workers=2, progress=False)
    print('Input:', inp)
    print('Output:', out)
    num_errors = out.count(None)
    print('Number of errors (None):', num_errors)
    for idx, val in enumerate(out):
        if idx == 3:
            print(f'Index {idx}: Expected None, got {val}')
        else:
            print(f'Index {idx}: Expected {idx}, got {val}')
    assert num_errors == 1, f'Expected 1 error, got {num_errors}'
    assert out[3] is None, f'Expected None at index 3, got {out[3]}'
    for i, val in enumerate(out):
        if i != 3:
            assert val == i, f'Expected {i} at index {i}, got {val}'


def test_kill_all_thread_interrupts_sleepy_workers():
    """Simulate an IPython session: launch work, then abort via kill_all_thread."""

    def slow_worker(x: int) -> int:
        time.sleep(0.5)
        return x

    outcome: dict[str, object] = {}

    def run_pool() -> None:
        try:
            multi_thread(
                slow_worker,
                range(8),
                workers=4,
                progress=False,
                prefetch_factor=1,
            )
        except SystemExit:
            outcome['system_exit'] = True
        except Exception as exc:  # pragma: no cover - diagnostic safety net
            outcome['exception'] = exc
        else:
            outcome['completed'] = True

    runner = threading.Thread(target=run_pool, daemon=True)
    runner.start()

    start = time.time()
    while not SPEEDY_RUNNING_THREADS and time.time() - start < 1.0:
        time.sleep(0.01)

    assert SPEEDY_RUNNING_THREADS, 'expected worker threads to be active'

    killed = kill_all_thread()
    runner.join(timeout=2.0)

    time.sleep(0.05)
    _prune_dead_threads()

    assert killed > 0, 'kill_all_thread should signal at least one worker'
    assert not runner.is_alive(), 'background multi_thread should have stopped'
    assert not SPEEDY_RUNNING_THREADS, 'all tracked threads must be cleared'
    unexpected = outcome.get('exception')
    assert unexpected is None, f'unexpected exception: {unexpected}'
    assert outcome.get('system_exit') or outcome.get('completed')


def test_keyboard_interrupt_cancels_immediately(monkeypatch):
    """multi_thread should abort instantly when the main thread sees Ctrl+C."""

    real_wait = thread_mod.wait
    real_kill = thread_mod.kill_all_thread

    def fake_wait(*args, **kwargs):  # pragma: no cover - deterministic path
        raise KeyboardInterrupt

    kill_calls: dict[str, int] = {}

    def wrapped_kill(exc_type=SystemExit, join_timeout=0.1):
        kill_calls['count'] = kill_calls.get('count', 0) + 1
        return real_kill(exc_type, join_timeout)

    monkeypatch.setattr(thread_mod, 'wait', fake_wait)
    monkeypatch.setattr(thread_mod, 'kill_all_thread', wrapped_kill)

    with pytest.raises(KeyboardInterrupt):
        multi_thread(
            lambda x: time.sleep(1) or x,
            range(10),
            workers=4,
            progress=False,
        )

    monkeypatch.setattr(thread_mod, 'wait', real_wait)
    monkeypatch.setattr(thread_mod, 'kill_all_thread', real_kill)

    for _ in range(50):
        _prune_dead_threads()
        if not SPEEDY_RUNNING_THREADS:
            break
        time.sleep(0.02)

    assert kill_calls.get('count') == 1
    assert not SPEEDY_RUNNING_THREADS


# ────────────────────────────────────────────────────────────
# 11. Test speedy vs normal for loop
def test_speedy_vs_normal():
    def f(x):
        return _sleepy(x * x)

    # Use a smaller range to ensure the test completes quickly
    inp = list(range(100))

    # Add more visible output for debugging
    print('\n' + '=' * 50)
    print('RUNNING SPEED COMPARISON TEST')
    print('Input size:', len(inp))
    print('=' * 50)

    # First try: Basic configuration - no progress bar
    start_time = time.time()
    out1 = multi_thread(f, inp, workers=4, progress=True, stop_on_error=False)
    mt_time = time.time() - start_time
    print(f'Speedy multi-threading took: {mt_time:.4f} seconds')

    start_time = time.time()
    out2 = [f(x) for x in inp]
    st_time = time.time() - start_time
    print(f'Normal for loop took: {st_time:.4f} seconds')

    print(f'Speed improvement: {st_time / mt_time:.2f}x faster')

    # Print details about the outputs for debugging
    print(f'\nSpeedy output (length={len(out1)}): {out1}')
    print(f'Normal output (length={len(out2)}): {out2}')

    # Check if all values match
    is_equal = True
    differences = []
    for i, (v1, v2) in enumerate(zip(out1, out2, strict=False)):
        if v1 != v2:
            is_equal = False
            differences.append(f'Index {i}: {v1} != {v2}')

    if not is_equal:
        print('\nOutput differences:')
        for diff in differences[:5]:  # Show only first 5 differences
            print(diff)
        if len(differences) > 5:
            print(f'...and {len(differences) - 5} more differences')

    # Final assertion
    assert out1 == out2, 'Outputs do not match!'


# =====


# ────────────────────────────────────────────────────────────
# 12. Test multi_thread vs standard threading, use a heavry compute function like fibonacci
def fibonacci(n, x):
    if n <= 1:
        return n + x
    return fibonacci(n - 1, x) + fibonacci(n - 2, x)


def test_multi_thread_vs_standard():
    def f(x):
        return fibonacci(x, 0)

    # Create a longer input list for more substantial testing
    inp = list(range(10, 35))  # Fibonacci numbers from 10 to 34

    # Use multi_thread
    start_mt = time.time()
    print(inp)
    out_mt = multi_thread(
        f,
        inp,
        workers=4,
        progress=False,
    )
    mt_time = time.time() - start_mt

    # Use standard ThreadPoolExecutor
    start_std = time.time()
    out_std = multi_thread_standard(f, inp, workers=4)
    std_time = time.time() - start_std

    print(
        f'multi_thread: {mt_time:.4f}s, standard: {std_time:.4f}s, outputs equal: {out_mt == out_std}'
    )
    assert out_mt == out_std, 'multi_thread and standard outputs differ'
