import time
import multiprocessing

if hasattr(multiprocessing, "set_start_method"):
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # It's already set, which is fine
        pass

from speedy_utils import multi_thread
from speedy_utils.multi_worker.process import multi_process, tqdm


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
    out = multi_process(maybe_fail, inp, stop_on_error=False, workers=2, progress=False)
    assert out.count(None) == 1
    assert out[3] is None
    for i, val in enumerate(out):
        if i != 3:
            assert val == i


def test_multi_process_vs_serial():
    """Outputs must match a plain Python for‑loop. Also log the time for each."""
    inp = list(range(200))

    start_mp = time.perf_counter()
    out_mp = multi_process(square, inp, workers=4, progress=True)
    dur_mp = time.perf_counter() - start_mp

    start_serial = time.perf_counter()
    out_serial = [square(x) for x in tqdm(inp, desc="serial")]
    dur_serial = time.perf_counter() - start_serial

    print(f"multi_process duration: {dur_mp:.6f} seconds")
    print(f"serial duration: {dur_serial:.6f} seconds")

    assert out_mp == out_serial

def forloop(func, inp):
    """Plain Python for loop to compare against multi_process."""
    out = []
    for i in tqdm(inp, desc="forloop"):
        out.append(func(i))
    return out

def test_process_vs_thread_heavy():
    """Heavy CPU: compare multi_process against multi_thread for correctness and speed."""
    inp = [22]*1000

    start_proc = time.perf_counter()
    out_proc = multi_process(fibonacci, inp, workers=4, progress=True)
    dur_proc = time.perf_counter() - start_proc

    start_thread = time.perf_counter()
    out_thread = multi_thread(fibonacci, inp, workers=100, progress=True)
    dur_thread = time.perf_counter() - start_thread

    # test for loop
    start_forloop = time.perf_counter()
    out_for = forloop(fibonacci, inp)
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
