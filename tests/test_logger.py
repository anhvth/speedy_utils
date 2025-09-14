import time
from unittest.mock import MagicMock, patch

import pytest  # type: ignore[import-not-found]
from freezegun import freeze_time

from speedy_utils.common.logger import (
    _last_log_intervals,
    _last_log_times,
    _logged_once_set,
    log,
    setup_logger,
)


@pytest.fixture
def reset_log_state():
    """Reset all global state used by the logger between tests."""
    _logged_once_set.clear()
    _last_log_intervals.clear()
    _last_log_times.clear()
    yield
    _logged_once_set.clear()
    _last_log_intervals.clear()
    _last_log_times.clear()


class TestSetupLogger:
    @patch("speedy_utils.common.logger.logger")
    def test_setup_logger_with_default_values(self, mock_logger):
        """Test setup_logger with default parameters."""
        setup_logger()

        # Verify logger was configured correctly
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()

        # Check that INFO level was set
        add_call_args = mock_logger.add.call_args[1]
        assert "filter" in add_call_args
        mock_logger.level.assert_called_once_with("INFO")

    @patch("speedy_utils.common.logger.logger")
    def test_setup_logger_with_custom_level(self, mock_logger):
        """Test setup_logger with custom log level."""
        setup_logger(level="Debug")
        mock_logger.level.assert_called_once_with("DEBUG")

    @patch("speedy_utils.common.logger.logger")
    def test_setup_logger_with_shorthand_level(self, mock_logger):
        """Test setup_logger with shorthand log level notation."""
        setup_logger(level="E")
        mock_logger.level.assert_called_once_with("ERROR")

    @patch("speedy_utils.common.logger.logger")
    def test_setup_logger_disable_logging(self, mock_logger):
        """Test setup_logger with DISABLE level."""
        setup_logger(level="Disable")
        mock_logger.disable.assert_called_once_with("")
        mock_logger.info.assert_called_once_with("Logging disabled")

    @patch("speedy_utils.common.logger.logger")
    def test_rate_limiting_filter(self, mock_logger, reset_log_state):
        """Test that the log filter correctly rate-limits messages."""
        min_interval = 2.0
        setup_logger(min_interval=min_interval)

        # Extract the filter function
        filter_fn = mock_logger.add.call_args[1]["filter"]

        # Create mock record with file and line info
        record1 = {
            "file": "test.py",
            "line": 10,
            "level": MagicMock(),
            "function": "test_func",
        }
        record1["level"].no = 30  # INFO level

        with patch("speedy_utils.common.logger.logger.level") as mock_level:
            mock_level.return_value.no = 20  # DEBUG level threshold

            # First call should pass
            assert filter_fn(record1) is True

            # Second call within min_interval should be filtered out
            assert filter_fn(record1) is False

            # Advance time beyond min_interval
            with patch("time.time") as mock_time:
                mock_time.return_value = time.time() + min_interval + 0.1
                # Should pass again after interval
                assert filter_fn(record1) is True

    @patch("speedy_utils.common.logger.logger")
    def test_grep_filtering(self, mock_logger):
        """Test that the log filter correctly applies grep patterns."""
        # Enable only logs from 'foo.py'
        setup_logger(enable_grep="foo.py")

        # Extract the filter function
        filter_fn = mock_logger.add.call_args[1]["filter"]

        with patch("speedy_utils.common.logger.logger.level") as mock_level:
            mock_level.return_value.no = 20  # DEBUG level threshold

            # Create records for different files
            record_foo = {
                "file": "foo.py",
                "line": 10,
                "level": MagicMock(),
                "function": "test_func",
            }
            record_foo["level"].no = 30  # INFO level

            record_bar = {
                "file": "bar.py",
                "line": 10,
                "level": MagicMock(),
                "function": "test_func",
            }
            record_bar["level"].no = 30  # INFO level

            # foo.py should pass the filter
            assert filter_fn(record_foo) is True
            # bar.py should be filtered out
            assert filter_fn(record_bar) is False

    @patch("speedy_utils.common.logger.logger")
    def test_disable_grep_filtering(self, mock_logger):
        """Test that the log filter correctly applies disable_grep patterns."""
        # Disable logs from 'secret.py'
        setup_logger(disable_grep="secret.py")

        # Extract the filter function
        filter_fn = mock_logger.add.call_args[1]["filter"]

        with patch("speedy_utils.common.logger.logger.level") as mock_level:
            mock_level.return_value.no = 20  # DEBUG level threshold

            # Create records for different files
            record_normal = {
                "file": "normal.py",
                "line": 10,
                "level": MagicMock(),
                "function": "test_func",
            }
            record_normal["level"].no = 30  # INFO level

            record_secret = {
                "file": "secret.py",
                "line": 10,
                "level": MagicMock(),
                "function": "test_func",
            }
            record_secret["level"].no = 30  # INFO level

            # normal.py should pass the filter
            assert filter_fn(record_normal) is True
            # secret.py should be filtered out
            assert filter_fn(record_secret) is False


class TestLogFunction:
    @patch("speedy_utils.common.logger.logger.opt")
    def test_basic_logging(self, mock_opt, reset_log_state):
        """Test that the log function correctly delegates to loguru."""
        mock_info = MagicMock()
        mock_opt.return_value.info = mock_info

        log("Test message")

        mock_opt.assert_called_once_with(depth=1)
        mock_info.assert_called_once_with("Test message")

    @patch("speedy_utils.common.logger.logger.opt")
    def test_different_log_levels(self, mock_opt, reset_log_state):
        """Test that the log function respects the level parameter."""

        for level in ["info", "warning", "error", "critical", "success"]:
            mock_level_fn = MagicMock()
            setattr(mock_opt.return_value, level, mock_level_fn)

            log("Test message", level=level)  # type: ignore[arg-type]

            mock_level_fn.assert_called_once_with("Test message")
            mock_level_fn.reset_mock()

    @patch("speedy_utils.common.logger.logger.opt")
    @patch("speedy_utils.common.logger._get_call_site_id")
    def test_log_once(self, mock_get_id, mock_opt, reset_log_state):
        """Test that the once parameter prevents duplicate logs."""
        # Return a consistent call site ID for testing
        mock_get_id.return_value = "test_file.py:100"

        mock_info = MagicMock()
        mock_opt.return_value.info = mock_info

        # First call should log
        log("Test message", once=True)
        assert mock_info.call_count == 1

        # Second call should be suppressed
        log("Test message again", once=True)
        assert mock_info.call_count == 1

        # Different call site should log
        mock_get_id.return_value = "different_file.py:20"
        log("Different call site", once=True)
        assert mock_info.call_count == 2

    @patch("speedy_utils.common.logger.logger.opt")
    @patch("speedy_utils.common.logger._get_call_site_id")
    def test_log_interval(self, mock_get_id, mock_opt, reset_log_state):
        """Test that the interval parameter rate-limits logs."""
        # Return a consistent call site ID for testing
        mock_get_id.return_value = "test_file.py:100"

        mock_info = MagicMock()
        mock_opt.return_value.info = mock_info

        # First call should log
        with freeze_time("2023-01-01 12:00:00"):
            log("Test message", interval=5.0)
            assert mock_info.call_count == 1

            # Call within interval should be suppressed
            log("Too soon", interval=5.0)
            assert mock_info.call_count == 1

        # Call after interval should log
        with freeze_time("2023-01-01 12:00:06"):
            log("After interval", interval=5.0)
            assert mock_info.call_count == 2


class TestRateLimitCache:
    def test_rate_limit_cache_max_size(self, reset_log_state):
        """Test that the _RateLimitCache correctly evicts oldest items."""
        # Set a small cache size for testing
        setup_logger(max_cache_entries=3)

        # Fill the cache
        _last_log_times["test1.py:10"] = 100
        _last_log_times["test2.py:20"] = 200
        _last_log_times["test3.py:30"] = 300

        # Cache should have 3 items
        assert len(_last_log_times) == 3
        assert "test1.py:10" in _last_log_times

        # Add one more, the oldest should be evicted
        _last_log_times["test4.py:40"] = 400

        assert len(_last_log_times) == 3
        assert "test1.py:10" not in _last_log_times
        assert "test4.py:40" in _last_log_times

    def test_rate_limit_cache_reuse(self, reset_log_state):
        """Test that accessing an existing item moves it to the end of eviction order."""
        # Set a small cache size for testing
        setup_logger(max_cache_entries=3)

        # Fill the cache
        _last_log_times["test1.py:10"] = 100
        _last_log_times["test2.py:20"] = 200
        _last_log_times["test3.py:30"] = 300

        # Access the oldest item to refresh it
        _last_log_times["test1.py:10"] = 400

        # Add one more, test2 should be evicted now instead of test1
        _last_log_times["test4.py:40"] = 500

        assert len(_last_log_times) == 3
        assert "test1.py:10" in _last_log_times
        assert "test2.py:20" not in _last_log_times
        assert "test3.py:30" in _last_log_times
        assert "test4.py:40" in _last_log_times
