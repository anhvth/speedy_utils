# tests/test_import_time.py
"""Regression test to ensure speedy_utils import time stays under 1 second."""
import subprocess
import sys
import time

import pytest


def test_import_time_star_import():
    """
    Ensure that `from speedy_utils import *` completes in under 1 second.

    This is a regression test to catch any accidental additions of slow
    module-level imports that would degrade the import performance.

    The test runs in a subprocess to ensure a clean import environment.
    """
    code = """
import time
start = time.perf_counter()
from speedy_utils import *
elapsed = time.perf_counter() - start
print(f"IMPORT_TIME:{elapsed}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    # Parse the import time from output
    import_time = 0.0
    for line in result.stdout.splitlines():
        if line.startswith("IMPORT_TIME:"):
            import_time = float(line.split(":")[1])
            break
    else:
        pytest.fail(f"Could not parse import time from output: {result.stdout}\nstderr: {result.stderr}")

    # Assert import time is under 1 second (subprocess variance; hook uses 0.4s)
    max_time = 1.0
    assert import_time < max_time, (
        f"Import time ({import_time:.3f}s) exceeds maximum allowed ({max_time}s). "
        "Check for slow module-level imports in speedy_utils."
    )


def test_import_time_module_import():
    """
    Ensure that `import speedy_utils` completes in under 1 second.

    This tests the basic module import without the star import.
    """
    code = """
import time
start = time.perf_counter()
import speedy_utils
elapsed = time.perf_counter() - start
print(f"IMPORT_TIME:{elapsed}")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    # Parse the import time from output
    import_time = 0.0
    for line in result.stdout.splitlines():
        if line.startswith("IMPORT_TIME:"):
            import_time = float(line.split(":")[1])
            break
    else:
        pytest.fail(f"Could not parse import time from output: {result.stdout}\nstderr: {result.stderr}")

    # Assert import time is under 1 second (subprocess variance; hook uses 0.4s)
    max_time = 1.0
    assert import_time < max_time, (
        f"Import time ({import_time:.3f}s) exceeds maximum allowed ({max_time}s). "
        "Check for slow module-level imports in speedy_utils."
    )
