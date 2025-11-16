#!/usr/bin/env python
"""
test_process.py - Tests for the multi-process functionality
that properly handles the multiprocessing context
"""

import multiprocessing as mp
import time

from speedy_utils.multi_worker.process import multi_process


def multiply(x, factor=3):
    """Test function that multiplies input by a factor."""
    time.sleep(0.01)  # Short sleep to simulate work
    return x * factor


def multiply_by_three(x):
    """Helper function that multiplies input by 3."""
    return multiply(x, factor=3)


def test_multi_process_success():
    """Test multi_process with a function that multiplies input by a factor."""
    # Use 'spawn' method to avoid fork warnings in Python 3.12+
    if hasattr(mp, "get_context"):
        mp.set_start_method("spawn", force=True)

    inputs = range(5)
    results = multi_process(
        func=multiply_by_three, inputs=inputs, workers=2, progress=False
    )
    expected = [x * 3 for x in inputs]
    assert results == expected, f"Expected {expected}, got {results}"


def test_multi_process_with_kwargs():
    """Test multi_process with a function that uses kwargs."""
    inputs = range(5)
    results = multi_process(
        func=multiply,
        inputs=inputs,
        workers=2,
        progress=False,
        factor=4,  # Pass as kwarg
    )
    expected = [x * 4 for x in inputs]
    assert results == expected, f"Expected {expected}, got {results}"


if __name__ == "__main__":
    test_multi_process_success()
    test_multi_process_with_kwargs()
    print("All multi-process tests passed.")
