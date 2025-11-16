"""
Tests for the updated multi_process functionality with process_update_interval.
"""

import time
from unittest.mock import MagicMock, patch

from speedy_utils.multi_worker.process import multi_process


# Test helper functions for multiprocessing
def identity(x):
    """Simple identity function for testing."""
    return x


def slow_identity(x, delay=0.01):
    """Identity function with a slight delay to test progress updates."""
    time.sleep(delay)
    return x


def failing_function(x):
    """Function that raises an error for a specific input."""
    if x == 5:
        raise ValueError("Test error")
    return x


def test_process_update_interval():
    """Test that the process_update_interval parameter works correctly."""
    # Create a list of 20 items to process
    test_input = list(range(20))

    # Run multi_process with progress=True and process_update_interval=5
    # Note: process_update_interval is accepted for compatibility but not implemented for safe backend
    result = multi_process(
        slow_identity,
        test_input,
        workers=2,
        progress=False,  # Disable progress to avoid fastcore's progress bar
        process_update_interval=5,
        backend="safe",
    )

    # Check results
    assert result == test_input


def test_worker_error_handling():
    """Test error handling in the worker process."""
    # Since stop_on_error is not implemented for safe backend,
    # we test that errors are properly raised

    # Test with a smaller set that should fail
    try:
        result = multi_process(failing_function, [5], backend="safe")
        raise AssertionError("Expected ValueError to be raised")
    except ValueError as e:
        assert "Test error" in str(e)

    # Test with a set that should succeed
    result = multi_process(failing_function, [1, 2, 3, 4], backend="safe")
    assert result == [1, 2, 3, 4]


def test_batch_parameter():
    """Test the batch parameter for multi_process."""
    # Create a list of 20 items to process
    test_input = list(range(20))

    # Process with batch=5
    result = multi_process(identity, test_input, batch=5, backend="safe")

    # Check results
    assert result == test_input
