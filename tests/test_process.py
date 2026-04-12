import contextlib
import multiprocessing
import os
import pickle
import ssl
import time
import types

import pytest


if hasattr(multiprocessing, "set_start_method"):
    with contextlib.suppress(RuntimeError):
        multiprocessing.set_start_method("spawn", force=True)

from speedy_utils import multi_thread
from speedy_utils.common.utils_print import flatten_dict
from speedy_utils.multi_worker import _multi_process as mp_mod
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


def always_fail(x):
    raise RuntimeError(f"bad item: {x}")


def passthrough_with_client(item, client=None):
    return item


def fibonacci(n):
    """Inefficient recursive Fibonacci – intentionally CPU heavy."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


def clone_as_main(func):
    cloned = types.FunctionType(
        func.__code__,
        func.__globals__,
        name=func.__name__,
        argdefs=func.__defaults__,
        closure=func.__closure__,
    )
    cloned.__module__ = "__main__"
    return cloned


# ────────────────────────────────────────────────────────────
# actual tests
# ────────────────────────────────────────────────────────────
def test_scalar_single_param():
    inp = [1, 2, 3]
    assert multi_process(
        double,
        inp,
        num_threads=2,
        progress=False,
        backend="thread",
    ) == [
        2,
        4,
        6,
    ]


def test_string_scalar():
    inp = ["ab", "cd"]
    assert multi_process(to_upper, inp, progress=False, backend="thread") == [
        "AB",
        "CD",
    ]


def test_batch_ordered():
    inp = list(range(20))
    out = multi_process(
        square,
        inp,
        num_threads=4,
        progress=False,
        backend="thread",
    )
    assert out == [i * i for i in inp]


def test_unordered():
    inp = list(range(32))
    out = multi_process(
        square,
        inp,
        num_threads=8,
        progress=False,
        backend="thread",
    )
    assert out == [i * i for i in inp]


def test_stop_on_error_false():
    """Test that log-mode error handling continues processing."""
    inp = list(range(5))
    result = multi_process(
        maybe_fail,
        inp,
        error_handler="log",
        num_threads=2,
        progress=False,
        backend="thread",
    )
    assert result[3] is None
    assert result[0] == 0
    assert result[1] == 1
    assert result[2] == 2
    assert result[4] == 4


def test_multi_process_vs_serial():
    inp = list(range(20))
    start_mp = time.perf_counter()
    out_mp = multi_process(
        square,
        inp,
        num_threads=4,
        progress=False,
        backend="thread",
    )
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
    out_proc = multi_process(
        fibonacci,
        inp,
        num_threads=4,
        progress=False,
        backend="thread",
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


def test_mp_default_num_threads_matches_sequential_results():
    inp = list(range(12))
    out_mp = multi_process(
        square,
        inp,
        num_procs=2,
        progress=False,
        backend="mp",
    )
    assert out_mp == [square(x) for x in inp]


def test_mp_nested_process_and_thread_fanout():
    inp = list(range(24))
    out = multi_process(
        square,
        inp,
        num_procs=2,
        num_threads=4,
        progress=False,
        backend="mp",
    )
    assert out == [i * i for i in inp]


def test_mp_unordered_returns_full_result_set():
    inp = list(range(24))
    out = multi_process(
        square,
        inp,
        num_procs=2,
        num_threads=3,
        progress=False,
        backend="mp",
    )
    assert out == [i * i for i in inp]


def test_mp_notebook_style_main_callable():
    notebook_func = types.FunctionType(
        square.__code__,
        globals(),
        name="forward_one",
        argdefs=square.__defaults__,
        closure=square.__closure__,
    )
    notebook_func.__module__ = "__main__"

    out = multi_process(
        notebook_func,
        list(range(8)),
        num_procs=2,
        progress=False,
        backend="mp",
    )
    assert out == [square(x) for x in range(8)]


def test_infer_importable_module_for_main_clone():
    main_flatten = clone_as_main(flatten_dict)

    assert mp_mod._infer_importable_module(main_flatten) == (
        "speedy_utils.common.utils_print",
        "flatten_dict",
    )


def test_serialize_spawn_callable_uses_import_ref_for_main_clone():
    main_flatten = clone_as_main(flatten_dict)

    payload = mp_mod._serialize_spawn_callable(main_flatten)
    data = pickle.loads(payload)

    assert data == {
        "_v": 1,
        "import_ref": True,
        "module_name": "speedy_utils.common.utils_print",
        "qualname": "flatten_dict",
    }

    restored = mp_mod._deserialize_spawn_callable(payload)
    assert restored({"outer": {"inner": 3}}) == {"outer.inner": 3}


def test_mp_importable_main_callable_uses_import_ref_path():
    main_flatten = clone_as_main(flatten_dict)

    out = multi_process(
        main_flatten,
        [{"a": {"b": 1}}, {"x": 2}],
        num_procs=2,
        progress=False,
        backend="mp",
    )

    assert out == [{"a.b": 1}, {"x": 2}]


def test_mp_local_closure_callable():
    offset = 7

    def forward_one(x):
        return x + offset

    out = multi_process(
        forward_one,
        [1, 2, 3],
        num_procs=2,
        progress=False,
        backend="mp",
    )
    assert out == [8, 9, 10]


def test_mp_kwargs_with_ssl_context_should_work_with_spawn_regression():
    """Regression: mp spawn should degrade safely with non-picklable kwargs."""
    with pytest.warns(RuntimeWarning, match="Falling back to thread backend"):
        out = multi_process(
            passthrough_with_client,
            [1, 2, 3],
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
            client=ssl.create_default_context(),
        )
    assert out == [1, 2, 3]


def test_mp_notebook_callable_with_sslcontext_global_regression():
    """Red test: spawn path should fall back gracefully for non-picklable notebook globals."""
    namespace: dict[str, object] = {}
    exec("def notebook_gate(x):\n    return x if ctx is not None else -1", namespace)
    notebook_gate_impl = namespace["notebook_gate"]
    assert isinstance(notebook_gate_impl, types.FunctionType)
    notebook_gate = types.FunctionType(
        notebook_gate_impl.__code__,
        {
            "__builtins__": __builtins__,
            "ctx": ssl.create_default_context(),
        },
        name="notebook_gate",
    )
    notebook_gate.__module__ = "__main__"

    with pytest.warns(RuntimeWarning, match="Falling back to thread backend"):
        out = multi_process(
            notebook_gate,
            [1, 2, 3],
            num_procs=2,
            progress=False,
            backend="mp",
        )
    assert out == [1, 2, 3]


def test_safe_backend_ignores_num_procs_and_honors_num_threads():
    inp = list(range(10))
    out = multi_process(
        square,
        inp,
        num_procs=8,
        num_threads=2,
        progress=False,
        backend="thread",
    )
    assert out == [i * i for i in inp]


def test_mp_error_handler_ignore_preserves_none_placeholders():
    inp = list(range(6))
    out = multi_process(
        maybe_fail,
        inp,
        num_procs=2,
        num_threads=3,
        progress=False,
        backend="mp",
        error_handler="ignore",
    )
    assert out[3] is None
    assert out[:3] == [0, 1, 2]
    assert out[4:] == [4, 5]


def test_mp_error_handler_raise_aborts():
    with pytest.raises(SystemExit):
        multi_process(
            always_fail,
            [1, 2, 3, 4],
            num_procs=2,
            num_threads=2,
            progress=False,
            backend="mp",
            error_handler="raise",
        )


def test_mp_lazy_output_returns_paths():
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

    for path in out:
        os.unlink(path)

    cache_dir = os.path.dirname(out[0])
    if cache_dir and os.path.isdir(cache_dir):
        with contextlib.suppress(OSError):
            os.rmdir(cache_dir)
