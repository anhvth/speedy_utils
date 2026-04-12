"""
Tests for the updated multi_process functionality with process_update_interval.
"""

import queue
import time
from unittest.mock import MagicMock, patch

from speedy_utils.multi_worker.process import multi_process
from speedy_utils.multi_worker import _multi_process as mp_mod


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
        self.refresh_calls = 0
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

    def refresh(self):
        self.refresh_calls += 1

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
    assert bar.postfix["start"] == 8
    assert bar.postfix["active"] == 0
    assert all(refresh is False for _, refresh in bar.postfix_calls)


class _FakeQueue:
    def __init__(self):
        self._calls = 0

    def get(self, timeout=None):
        time.sleep(min(timeout or 0, 0.02))
        self._calls += 1
        if self._calls == 1:
            return ("started", 1)
        raise queue.Empty


class _FakeProcess:
    next_pid = 1000

    def __init__(self, *args, **kwargs):
        self.pid = _FakeProcess.next_pid
        _FakeProcess.next_pid += 1
        self.exitcode = None
        self._thread = None

    def start(self):
        _FAKE_MP_ORDER.append("start")

        def _finish():
            time.sleep(0.2)
            self.exitcode = 0

        import threading

        self._thread = threading.Thread(target=_finish, daemon=True)
        self._thread.start()

    def is_alive(self):
        return self.exitcode is None

    def terminate(self):
        self.exitcode = -15

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)


class _FakeContext:
    def Queue(self):
        return _FakeQueue()

    def Process(self, *args, **kwargs):
        return _FakeProcess(*args, **kwargs)


_FAKE_MP_ORDER: list[str] = []


def test_mp_progress_bar_appears_before_process_start_and_refreshes_while_idle():
    _FakeTqdm.created.clear()
    _FAKE_MP_ORDER.clear()

    original_init = _FakeTqdm.__init__

    def _recording_init(self, *args, **kwargs):
        _FAKE_MP_ORDER.append("bar")
        original_init(self, *args, **kwargs)

    with patch.object(_FakeTqdm, "__init__", _recording_init):
        with patch("speedy_utils.multi_worker._multi_process.tqdm", _FakeTqdm):
            with patch("speedy_utils.multi_worker._multi_process.mp.get_context", return_value=_FakeContext()):
                with patch("speedy_utils.multi_worker._multi_process.psutil.Process", side_effect=lambda pid: pid):
                    with patch("speedy_utils.multi_worker._multi_process._track_processes", return_value=None):
                        result = mp_mod._run_multiprocess_backend(
                            func=identity,
                            cache_dir=None,
                            dump_in_thread=True,
                            items=[1, 2],
                            total=2,
                            num_procs=2,
                            num_threads=1,
                            desc="Test MP",
                            progress=True,
                            func_kwargs={},
                            log_worker="first",
                            log_gate_path=None,
                            error_handler="log",
                            error_stats=mp_mod.ErrorStats(
                                func_name="identity",
                                max_error_files=10,
                                write_logs=True,
                            ),
                            func_name="identity",
                            max_error_files=10,
                        )

    assert result == [None, None]
    assert _FAKE_MP_ORDER[0] == "bar"
    assert _FAKE_MP_ORDER.count("start") == 2
    bar = _FakeTqdm.created[0]
    assert bar.total == 2
    assert bar.postfix is not None
    assert bar.postfix["proc"].endswith("/2")
    assert bar.postfix["start"] >= 1
    assert bar.postfix["active"] >= 1
    assert bar.refresh_calls >= 2


def test_error_stats_logs_first_error_path():
    with patch("speedy_utils.multi_worker.common.logger") as mock_logger:
        mock_logger.opt.return_value.warning = MagicMock()
        error_stats = mp_mod.ErrorStats(
            func_name="identity",
            max_error_files=10,
            write_logs=True,
        )

        with patch.object(error_stats, "_write_error_log", return_value="/tmp/demo.log"):
            error_stats.record_error(0, ValueError("boom"), {"x": 1}, "identity")

    mock_logger.opt.return_value.warning.assert_called_once_with(
        "Error log: {}",
        "/tmp/demo.log",
    )


def test_mp_parent_logs_first_error_path():
    class _ErrorLogQueue:
        def __init__(self):
            self._calls = 0

        def get(self, timeout=None):
            time.sleep(min(timeout or 0, 0.02))
            self._calls += 1
            if self._calls == 1:
                return ("error_log", "/tmp/mp-error.log")
            raise queue.Empty

    class _ErrorLogContext:
        def Queue(self):
            return _ErrorLogQueue()

        def Process(self, *args, **kwargs):
            return _FakeProcess(*args, **kwargs)

    _FakeTqdm.created.clear()
    _FAKE_MP_ORDER.clear()

    with patch("speedy_utils.multi_worker._multi_process.tqdm", _FakeTqdm):
        with patch(
            "speedy_utils.multi_worker._multi_process.mp.get_context",
            return_value=_ErrorLogContext(),
        ):
            with patch(
                "speedy_utils.multi_worker._multi_process.psutil.Process",
                side_effect=lambda pid: pid,
            ):
                with patch(
                    "speedy_utils.multi_worker._multi_process._track_processes",
                    return_value=None,
                ):
                    with patch(
                        "speedy_utils.multi_worker._multi_process.logger"
                    ) as mock_logger:
                        mock_logger.opt.return_value.warning = MagicMock()
                        result = mp_mod._run_multiprocess_backend(
                            func=identity,
                            cache_dir=None,
                            dump_in_thread=True,
                            items=[1, 2],
                            total=2,
                            num_procs=2,
                            num_threads=1,
                            desc="Test MP",
                            progress=True,
                            func_kwargs={},
                            log_worker="first",
                            log_gate_path=None,
                            error_handler="log",
                            error_stats=mp_mod.ErrorStats(
                                func_name="identity",
                                max_error_files=10,
                                write_logs=True,
                            ),
                            func_name="identity",
                            max_error_files=10,
                        )

    assert result == [None, None]
    mock_logger.opt.return_value.warning.assert_called_once_with(
        "Error log: {}",
        "/tmp/mp-error.log",
    )
