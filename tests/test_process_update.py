"""
Tests for the updated multi_process functionality with process_update_interval.
"""

import time
from unittest.mock import patch

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

    # Run multi_process with progress=True and process_update_interval=5.
    result = multi_process(
        slow_identity,
        test_input,
        num_threads=2,
        progress=False,  # Disable progress to avoid fastcore's progress bar
        process_update_interval=5,
        backend="thread",
    )

    # Check results
    assert result == test_input


def test_worker_error_handling():
    """Test error handling in the worker process."""
    # Errors are logged but not raised (error_handler='log' by default).

    # Test with a smaller set that should fail (returns None for failed items)
    result = multi_process(failing_function, [5], backend="thread", progress=False)
    assert result == [None]  # Failed item returns None

    # Test with a set that should succeed
    result = multi_process(
        failing_function,
        [1, 2, 3, 4],
        backend="thread",
        progress=False,
    )
    assert result == [1, 2, 3, 4]


def test_batch_parameter():
    """Test the batch parameter for multi_process."""
    # Create a list of 20 items to process
    test_input = list(range(20))

    # Process with batch=5
    result = multi_process(identity, test_input, batch=5, backend="thread")

    # Check results
    assert result == test_input


class _FakeTqdm:
    created: list["_FakeTqdm"] = []

    def __init__(self, *args, total=None, desc=None, **kwargs):
        self.total = total
        self.desc = desc
        self.kwargs = kwargs
        self.updated = 0
        self.postfix = None
        self.postfix_calls: list[tuple[dict | None, bool | None]] = []
        _FakeTqdm.created.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, value):
        self.updated += value

    def set_postfix(self, postfix, refresh=None):
        self.postfix = postfix
        self.postfix_calls.append((postfix, refresh))

    def close(self):
        return None


def test_mp_progress_uses_single_parent_bar():
    _FakeTqdm.created.clear()
    with patch("speedy_utils.multi_worker._multi_process.tqdm", _FakeTqdm):
        result = multi_process(
            slow_identity,
            list(range(8)),
            num_procs=2,
            num_threads=2,
            progress=True,
            backend="mp",
        )

    assert result == list(range(8))
    assert len(_FakeTqdm.created) == 1
    bar = _FakeTqdm.created[0]
    assert bar.updated == 8
    assert bar.total == 8
    assert bar.desc == "Multi-process [mp: 2p x 2t]"
    assert bar.kwargs["dynamic_ncols"] is True
    assert bar.postfix is not None
    assert bar.postfix["proc"].endswith("/2")
    assert all(refresh is False for _, refresh in bar.postfix_calls)
