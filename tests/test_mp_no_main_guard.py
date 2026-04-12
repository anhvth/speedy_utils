"""Test that multi_process works without an if __name__ == '__main__' guard.

This is the TDD red→green test for the spawn-recursion hang bug:
  - With 'spawn', Python re-executes __main__ in every child process.
  - Without a guard, the script calls multi_process() again → infinite spawn.
  - Fix: detect we're in a child worker and return early.

Key: the bug only triggers when the script is run as `python script.py`,
because that is when multiprocessing.spawn sets init_main_from_path and
re-executes the file in every child.  Running via `python -c "..."` does NOT
trigger the bug because __main__ has no __file__ in that case.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

import pytest


# ---------------------------------------------------------------------------
# Helper: run a script written to a real temp file (NOT python -c)
# This is the only way to reproduce the spawn re-execution of __main__.
# ---------------------------------------------------------------------------

_SCRIPT_NO_MAIN_GUARD = textwrap.dedent("""\
    from time import sleep
    from speedy_utils import multi_process

    a = 1

    def f(x):
        sleep(0.01)
        return x * 2 + a

    # Top-level call — NO if __name__ == '__main__': guard
    items = list(range(8))
    results = multi_process(f, items, num_procs=2, desc="no-guard", progress=False)
    assert results == [i * 2 + 1 for i in range(8)], f"wrong results: {results}"
    print("OK:", results)
""")

_SCRIPT_NOTEBOOK_STYLE = textwrap.dedent("""\
    # Simulates notebook: code runs at module level, function captures globals
    from speedy_utils import multi_process

    FACTOR = 3

    def compute(x):
        return x * FACTOR

    results = multi_process(compute, list(range(6)), num_procs=2, progress=False)
    assert results == [i * 3 for i in range(6)], f"wrong: {results}"
    print("OK:", results)
""")


def _run_script_file(src: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Write src to a real .py file and run it with `python script.py`.

    This triggers multiprocessing.spawn's init_main_from_path mechanism,
    which re-executes the file in every child process.  Using python -c
    does NOT reproduce that behaviour.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="speedy_test_"
    ) as f:
        f.write(src)
        fpath = f.name
    try:
        return subprocess.run(
            [sys.executable, fpath],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        os.unlink(fpath)


def test_multi_process_no_main_guard():
    """multi_process must complete without hanging when there is no __main__ guard."""
    proc = _run_script_file(_SCRIPT_NO_MAIN_GUARD)
    assert proc.returncode == 0, (
        f"Script failed (rc={proc.returncode}).\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "OK:" in proc.stdout


def test_multi_process_notebook_style():
    """multi_process must work for notebook-style top-level calls with global captures."""
    proc = _run_script_file(_SCRIPT_NOTEBOOK_STYLE)
    assert proc.returncode == 0, (
        f"Script failed (rc={proc.returncode}).\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    assert "OK:" in proc.stdout


def test_multi_process_child_env_flag_is_no_op():
    """When _SPEEDY_MP_CHILD is set (simulating a spawned worker), multi_process returns []."""
    from speedy_utils import multi_process

    old = os.environ.pop("_SPEEDY_MP_CHILD", None)
    try:
        os.environ["_SPEEDY_MP_CHILD"] = "1"
        result = multi_process(lambda x: x, list(range(5)), num_procs=2, progress=False)
        assert result == [], f"Expected [] in child context, got {result}"
    finally:
        if old is None:
            os.environ.pop("_SPEEDY_MP_CHILD", None)
        else:
            os.environ["_SPEEDY_MP_CHILD"] = old
