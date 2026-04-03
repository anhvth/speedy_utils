from __future__ import annotations

import importlib.util
import json
import os
import sys
import textwrap
import threading
import time
import uuid
from pathlib import Path

import pytest # type: ignore

import speedy_utils.multi_worker.parallel as pmod
from speedy_utils import parallel


# ---------------------------------------------------------------------------
# Worker functions used by multiprocessing tests must stay at module scope.
# ---------------------------------------------------------------------------

def square(x):
    return x * x


def delayed_echo(x):
    time.sleep(0.03 * (5 - x))
    return x


def add(a, b):
    return a + b


def combine(a, b, c=0):
    return a + b + c


def marker_square(x):
    marker_dir = Path(os.environ["PARALLEL_MARKER_DIR"])
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"{uuid.uuid4().hex}.txt"
    marker_path.write_text(str(x), encoding="utf-8")
    return x * x


def maybe_fail_resume(x):
    marker_dir = Path(os.environ["PARALLEL_MARKER_DIR"])
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker_path = marker_dir / f"{uuid.uuid4().hex}.txt"
    marker_path.write_text(str(x), encoding="utf-8")
    fail_on = os.environ.get("PARALLEL_FAIL_ON")
    if fail_on is not None and x == int(fail_on):
        raise ValueError(f"boom:{x}")
    return x + 100


def always_fail_for_two(x):
    if x == 2:
        raise RuntimeError("bad-two")
    return x


def pid_tid(x):
    time.sleep(0.01)
    return (os.getpid(), threading.get_ident(), x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_parallel_tmp(monkeypatch, tmp_path):
    root = tmp_path / "parallel-cache"
    monkeypatch.setenv("PARALLEL_TMP_ROOT", str(root))
    return root


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parallel_basic_scalar_execution():
    result = parallel(square, [1, 2, 3, 4], num_procs=2, num_threads=2, progress=False)
    assert result == [1, 4, 9, 16]


def test_parallel_preserves_input_order_when_runtime_finishes_out_of_order():
    result = parallel(delayed_echo, [1, 2, 3, 4], num_procs=2, num_threads=2, progress=False)
    assert result == [1, 2, 3, 4]


def test_parallel_supports_tuple_unpacking():
    result = parallel(add, [(1, 2), (3, 4), (10, -2)], num_procs=2, num_threads=2, progress=False)
    assert result == [3, 7, 8]


def test_parallel_supports_dict_unpacking():
    result = parallel(
        combine,
        [{"a": 1, "b": 2}, {"a": 3, "b": 4, "c": 5}, {"a": -1, "b": 1, "c": 10}],
        num_procs=2,
        num_threads=2,
        progress=False,
    )
    assert result == [3, 12, 10]


def test_parallel_accepts_auto_worker_values():
    data = list(range(20))
    result = parallel(square, data, num_procs="auto", num_threads="auto", progress=False)
    assert result == [x * x for x in data]


def test_parallel_zero_inputs_returns_empty_list():
    assert parallel(square, [], num_procs=2, num_threads=2, progress=False) == []


def test_parallel_deduplicates_identical_inputs_within_a_run(monkeypatch, tmp_path):
    marker_dir = tmp_path / "markers"
    monkeypatch.setenv("PARALLEL_MARKER_DIR", str(marker_dir))

    data = [1, 1, 2, 2, 3, 3, 3]
    result = parallel(marker_square, data, num_procs=2, num_threads=3, progress=False)

    assert result == [1, 1, 4, 4, 9, 9, 9]
    assert len(list(marker_dir.iterdir())) == len(set(data))


def test_parallel_reuses_cached_outputs_across_runs(monkeypatch, tmp_path):
    marker_dir = tmp_path / "markers"
    monkeypatch.setenv("PARALLEL_MARKER_DIR", str(marker_dir))

    data = [1, 2, 3, 2, 1]
    first = parallel(marker_square, data, num_procs=2, num_threads=2, progress=False)
    count_after_first = len(list(marker_dir.iterdir()))
    second = parallel(marker_square, data, num_procs=2, num_threads=2, progress=False)
    count_after_second = len(list(marker_dir.iterdir()))

    assert first == [1, 4, 9, 4, 1]
    assert second == first
    assert count_after_first == len(set(data))
    assert count_after_second == count_after_first


def test_parallel_resumes_only_missing_outputs_after_failure(monkeypatch, tmp_path):
    marker_dir = tmp_path / "markers"
    monkeypatch.setenv("PARALLEL_MARKER_DIR", str(marker_dir))
    monkeypatch.setenv("PARALLEL_FAIL_ON", "2")

    with pytest.raises(RuntimeError) as exc_info:
        parallel(maybe_fail_resume, [0, 1, 2, 3], num_procs=2, num_threads=2, progress=False)
    assert "boom:2" in str(exc_info.value)

    count_after_failure = len(list(marker_dir.iterdir()))
    monkeypatch.delenv("PARALLEL_FAIL_ON")

    result = parallel(maybe_fail_resume, [0, 1, 2, 3], num_procs=2, num_threads=2, progress=False)
    count_after_resume = len(list(marker_dir.iterdir()))

    assert result == [100, 101, 102, 103]
    assert count_after_resume == count_after_failure + 1


def test_parallel_invalidates_cache_when_function_source_changes(tmp_path, monkeypatch):
    module_dir = tmp_path / "dynmods"
    module_dir.mkdir()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    monkeypatch.setenv("PARALLEL_MARKER_DIR", str(marker_dir))

    def load_module(version_text: str, module_name: str):
        module_path = module_dir / f"{module_name}.py"
        module_path.write_text(version_text, encoding="utf-8")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    module_v1 = load_module(
        textwrap.dedent(
            """
            import os
            import uuid
            from pathlib import Path

            def compute(x):
                marker_dir = Path(os.environ[\"PARALLEL_MARKER_DIR\"])
                marker_dir.mkdir(parents=True, exist_ok=True)
                (marker_dir / f\"{uuid.uuid4().hex}.txt\").write_text(f\"v1:{x}\", encoding=\"utf-8\")
                return x + 1
            """
        ),
        "dynamic_parallel_mod_v1",
    )
    result_v1 = parallel(module_v1.compute, [1, 2], num_procs=1, num_threads=2, progress=False)
    count_after_v1 = len(list(marker_dir.iterdir()))

    module_v2 = load_module(
        textwrap.dedent(
            """
            import os
            import uuid
            from pathlib import Path

            def compute(x):
                marker_dir = Path(os.environ[\"PARALLEL_MARKER_DIR\"])
                marker_dir.mkdir(parents=True, exist_ok=True)
                (marker_dir / f\"{uuid.uuid4().hex}.txt\").write_text(f\"v2:{x}\", encoding=\"utf-8\")
                return x + 2
            """
        ),
        "dynamic_parallel_mod_v2",
    )
    result_v2 = parallel(module_v2.compute, [1, 2], num_procs=1, num_threads=2, progress=False)
    count_after_v2 = len(list(marker_dir.iterdir()))

    assert result_v1 == [2, 3]
    assert result_v2 == [3, 4]
    assert count_after_v2 == count_after_v1 + 2


def test_parallel_persists_error_trace_for_failures(isolated_parallel_tmp):
    with pytest.raises(RuntimeError) as exc_info:
        parallel(always_fail_for_two, [1, 2, 3], num_procs=2, num_threads=2, progress=False)

    assert "bad-two" in str(exc_info.value)

    func_name, func_signature = pmod._get_function_identity(always_fail_for_two)
    func_dir = isolated_parallel_tmp / f"{func_name}_{func_signature}"
    failing_hash = pmod._hash_input(2)
    error_path = func_dir / failing_hash / "error.txt"
    status_path = func_dir / failing_hash / "status.json"

    assert error_path.exists()
    assert "bad-two" in error_path.read_text(encoding="utf-8")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "error"


def test_parallel_creates_expected_cache_layout(isolated_parallel_tmp):
    result = parallel(square, [5], num_procs=1, num_threads=1, progress=False)
    assert result == [25]

    func_name, func_signature = pmod._get_function_identity(square)
    func_dir = isolated_parallel_tmp / f"{func_name}_{func_signature}"
    input_hash = pmod._hash_input(5)
    input_dir = func_dir / input_hash

    assert func_dir.exists()
    assert input_dir.exists()
    assert (input_dir / "input.pkl").exists()
    assert (input_dir / "input_meta.json").exists()
    assert (input_dir / "output.pkl").exists()
    assert (input_dir / "output_meta.json").exists()
    assert (input_dir / "status.json").exists()
    runs_dir = func_dir / "runs"
    assert runs_dir.exists()
    assert any((run / "manifest.json").exists() for run in runs_dir.iterdir())


def test_parallel_threads_live_inside_processes_and_results_are_returned():
    data = list(range(12))
    result = parallel(pid_tid, data, num_procs=2, num_threads=3, progress=False)

    assert [item[2] for item in result] == data
    pids = {item[0] for item in result}
    assert 1 <= len(pids) <= 2


@pytest.mark.parametrize(
    ("num_procs", "num_threads"),
    [(-1, 1), (1, -1), ("bad", 1), (1, "bad")],
)
def test_parallel_rejects_invalid_worker_values(num_procs, num_threads):
    with pytest.raises(ValueError):
        parallel(square, [1, 2], num_procs=num_procs, num_threads=num_threads, progress=False)
