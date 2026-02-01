import contextlib
import multiprocessing
import time


if hasattr(multiprocessing, "set_start_method"):
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)

from speedy_utils import multi_thread
from speedy_utils.multi_worker.process import multi_process, tqdm


# ────────────────────────────────────────────────────────────
# helpers (top‑level ⇒ picklable)
# ────────────────────────────────────────────────────────────
def _sleepy(x, delay=0.00001):
    """Tiny deterministic sleep to keep processes busy but tests fast."""
    time.sleep(delay)
    return x


def double(x):
    return _sleepy(x * 2)


def add_default(x, y=5):
    return _sleepy(x + y)


def mul(a, b):
    return _sleepy(a * b)


def dict_plus(item, y=100):
    return _sleepy(item["x"] + y)


def add_pair(a, b):
    return _sleepy(a + b)


def tuple_first_plus(item, y=100):
    return _sleepy(item[0] + y)


def to_upper(name):
    return _sleepy(name.upper())


def square(x):
    return _sleepy(x * x)


def maybe_fail(x):
    if x == 3:
        raise ValueError("boom")
    return _sleepy(x)


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
    assert multi_process(double, inp, workers=2, progress=False, backend="safe") == [
        2,
        4,
        6,
    ]


def test_string_scalar():
    inp = ["ab", "cd"]
    assert multi_process(to_upper, inp, progress=False, backend="safe") == ["AB", "CD"]


def test_batch_ordered():
    inp = list(range(20))
    out = multi_process(square, inp, batch=4, ordered=True, workers=4, progress=False, backend="safe")
    assert out == [i * i for i in inp]


def test_unordered():
    inp = list(range(32))
    out = multi_process(square, inp, ordered=False, workers=8, progress=False, backend="safe")
    assert sorted(out) == [i * i for i in inp]


def test_stop_on_error_false():
    """Test that with stop_on_error=False, errors don't halt processing."""
    inp = list(range(5))
    # With error_handler='log' (default), errors are logged but processing continues
    # Items that error return None
    result = multi_process(
        maybe_fail,
        inp,
        stop_on_error=False,
        workers=2,
        progress=False,
        backend="safe",
    )
    # Item at index 3 should fail and return None (maybe_fail raises at x==3)
    assert result[3] is None
    # Other items should process successfully
    assert result[0] == 0
    assert result[1] == 1
    assert result[2] == 2
    assert result[4] == 4


def test_multi_process_vs_serial():
    inp = list(range(20))
    start_mp = time.perf_counter()
    out_mp = multi_process(square, inp, workers=4, progress=False, backend="safe")
    dur_mp = time.perf_counter() - start_mp

    start_serial = time.perf_counter()
    out_serial = [square(x) for x in inp]
    dur_serial = time.perf_counter() - start_serial

    print(f"multi_process duration: {dur_mp:.6f} seconds")
    print(f"serial duration: {dur_serial:.6f} seconds")

    assert out_mp == out_serial


def forloop(func, inp):
    out = []
    for i in tqdm(inp, desc="forloop"):
        out.append(func(i))
    return out


def test_process_vs_thread_heavy():
    inp = [18] * 10

    start_proc = time.perf_counter()
    out_proc = multi_process(fibonacci, inp, workers=4, progress=False, backend="safe")
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
