# tests/test_multi_process.py
"""
Unit‑tests for speedy_utils.multi_worker.process.multi_process

The test cases mirror those in test_multi_thread.py.  All target functions
are declared at *module scope* to guarantee they are picklable under the
“spawn” start‑method that Windows uses for multiprocessing.
"""
import time
from speedy_utils.multi_worker.process import multi_process
from speedy_utils.multi_worker.thread import multi_thread   # used in the last test


# ────────────────────────────────────────────────────────────
# helpers (top‑level ⇒ picklable)
# ────────────────────────────────────────────────────────────
def _sleepy(x, delay=0.001):
    """Tiny deterministic sleep to keep processes busy but tests fast."""
    time.sleep(delay)
    return x


# 1. scalar – single positional parameter
def double(x):
    return _sleepy(x * 2)


# 2. scalar – extra parameter with default
def add_default(x, y=5):
    return _sleepy(x + y)


# 3. dict → kwargs   (keys ⊆ signature)
def mul(a, b):
    return _sleepy(a * b)


# 4. dict → single value (keys NOT in signature)
def dict_plus(item, y=100):
    return _sleepy(item["x"] + y)


# 5. 2‑element tuple/list → positional unpacking
def add_pair(a, b):
    return _sleepy(a + b)


# 6. 1‑element tuple/list → wrapped as value
def tuple_first_plus(item, y=100):
    return _sleepy(item[0] + y)


# 7. string input (treated like scalar)
def to_upper(name):
    return _sleepy(name.upper())


# 8 & 9. square helper for batching / ordering tests
def square(x):
    return _sleepy(x * x)


# 10. function that raises for x == 3
def maybe_fail(x):
    if x == 3:
        raise ValueError("boom")
    return _sleepy(x)


# 12. heavy CPU job
def fibonacci(n):
    """Inefficient recursive Fibonacci – intentionally CPU heavy."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# ────────────────────────────────────────────────────────────
# actual tests
# ────────────────────────────────────────────────────────────
def test_scalar_single_param():
    inp = [1, 2, 3]
    assert multi_process(double, inp, workers=2, progress=False) == [2, 4, 6]


def test_scalar_extra_default():
    inp = [1, 2, 3]
    assert multi_process(add_default, inp, progress=False) == [6, 7, 8]


def test_dict_as_kwargs():
    inp = [{"a": 2, "b": 4}, {"a": 3, "b": 5}]
    assert multi_process(mul, inp, progress=False) == [8, 15]


def test_dict_as_value():
    inp = [{"x": i} for i in range(1, 6)]
    assert multi_process(dict_plus, inp, progress=False) == [101, 102, 103, 104, 105]


def test_tuple_unpacked():
    inp = [(1, 10), (2, 20), (3, 30)]
    assert multi_process(add_pair, inp, progress=False) == [11, 22, 33]


def test_singleton_tuple_as_value():
    inp = [(i,) for i in range(1, 6)]
    assert multi_process(tuple_first_plus, inp, progress=False) == [
        101,
        102,
        103,
        104,
        105,
    ]


def test_string_scalar():
    inp = ["ab", "cd"]
    assert multi_process(to_upper, inp, progress=False) == ["AB", "CD"]


def test_batch_ordered():
    inp = list(range(20))
    out = multi_process(square, inp, batch=4, ordered=True, workers=4, progress=False)
    assert out == [i * i for i in inp]  # order preserved


def test_unordered():
    inp = list(range(32))
    out = multi_process(square, inp, ordered=False, workers=8, progress=False)
    assert sorted(out) == [i * i for i in inp]  # may reorder


def test_stop_on_error_false():
    inp = list(range(5))
    out = multi_process(
        maybe_fail, inp, stop_on_error=False, workers=2, progress=False
    )
    assert out.count(None) == 1 and out[3] is None
    for i, val in enumerate(out):
        if i != 3:
            assert val == i


def test_multi_process_vs_serial():
    """Outputs must match a plain Python for‑loop."""
    inp = list(range(40))
    out_mp = multi_process(square, inp, workers=4, progress=False)
    out_serial = [square(x) for x in inp]
    assert out_mp == out_serial


def forloop(inputs):
    ys = []
    for x in inputs:
        ys.append(fibonacci(x))
    return ys

def test_process_vs_thread_heavy():
    """Heavy CPU: compare multi_process against multi_thread for correctness and speed."""
    inp = [22]*1000

    start_proc = time.perf_counter()
    out_proc = multi_process(fibonacci, inp, workers=30, progress=True)
    dur_proc = time.perf_counter() - start_proc

    start_thread = time.perf_counter()
    out_thread = multi_thread(fibonacci, inp, workers=30, progress=False)
    dur_thread = time.perf_counter() - start_thread

    # test for loop
    start_forloop = time.perf_counter()
    out_for = forloop(inp)
    dur_forloop = time.perf_counter() - start_forloop

    assert out_proc == out_thread
    assert out_proc == out_for

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
    print(f"multi_process: {dur_proc:.2f}s")
    print(f"multi_thread: {dur_thread:.2f}s")
    print(f"for loop: {dur_forloop:.2f}s")
