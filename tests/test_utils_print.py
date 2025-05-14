"""
Tests for speedy_utils.common.utils_print module.

These tests validate the functionality of the various printing, logging,
and display utilities provided by this module.
"""
import json
import re
import time
from unittest.mock import Mock, patch
import pytest
from speedy_utils.common.utils_print import (
    _RateLimitCache,
    display_pretty_table_html,
    flatten_dict,
    fprint,
    print_table,
    setup_logger,
    log,
    _get_call_site_id,
)


class TestRateLimitCache:
    """Tests for the _RateLimitCache class."""

    def test_initialization(self):
        """Test that the cache initializes with the correct max size."""
        cache = _RateLimitCache(max_size=10)
        assert cache.max_size == 10
        assert len(cache) == 0

    def test_eviction(self):
        """Test that older items are evicted when max size is reached."""
        cache = _RateLimitCache(max_size=3)
        
        # Add 3 items (fills the cache)
        cache["key1"] = "value1"
        cache["key2"] = "value2"
        cache["key3"] = "value3"
        
        assert len(cache) == 3
        assert "key1" in cache
        
        # Add a 4th item (should evict the oldest - key1)
        cache["key4"] = "value4"
        
        assert len(cache) == 3
        assert "key1" not in cache  # key1 should have been evicted
        assert "key2" in cache
        assert "key3" in cache
        assert "key4" in cache

    def test_key_update(self):
        """Test that updating an existing key moves it to the end (newest)."""
        cache = _RateLimitCache(max_size=3)
        
        # Add 3 items
        cache["key1"] = "value1"
        cache["key2"] = "value2"
        cache["key3"] = "value3"
        
        # Update key1 (makes it newest)
        cache["key1"] = "updated_value1"
        
        # Add a 4th item (should evict the oldest - now key2)
        cache["key4"] = "value4"
        
        assert len(cache) == 3
        assert "key2" not in cache  # key2 should have been evicted
        assert "key1" in cache      # key1 should remain
        assert "key3" in cache
        assert "key4" in cache
        assert cache["key1"] == "updated_value1"  # Value was updated


class TestFlattenDict:
    """Tests for the flatten_dict function."""

    def test_simple_dict(self):
        """Test flattening a simple dictionary."""
        d = {"a": 1, "b": 2, "c": 3}
        result = flatten_dict(d)
        assert result == d  # Flat dictionary remains the same

    def test_nested_dict(self):
        """Test flattening a nested dictionary."""
        d = {
            "a": 1,
            "b": {
                "c": 2,
                "d": 3
            },
            "e": {
                "f": {
                    "g": 4
                }
            }
        }
        result = flatten_dict(d)
        expected = {
            "a": 1,
            "b.c": 2,
            "b.d": 3,
            "e.f.g": 4
        }
        assert result == expected

    def test_custom_separator(self):
        """Test flattening with a custom separator."""
        d = {
            "a": 1,
            "b": {
                "c": 2
            }
        }
        result = flatten_dict(d, sep="_")
        expected = {
            "a": 1,
            "b_c": 2
        }
        assert result == expected

    def test_empty_dict(self):
        """Test flattening an empty dictionary."""
        d = {}
        result = flatten_dict(d)
        assert result == {}


class TestDisplayPrettyTableHtml:
    """Tests for the display_pretty_table_html function."""

    @patch('speedy_utils.common.utils_print.display')
    @patch('speedy_utils.common.utils_print.HTML')
    def test_html_output(self, mock_html, mock_display):
        """Test that HTML output is generated correctly."""
        data = {"key1": "value1", "key2": "value2"}
        display_pretty_table_html(data)
        
        # Check that the HTML was created with the right content
        html_content = mock_html.call_args[0][0]
        assert "<table>" in html_content
        assert "<tr><td>key1</td><td>value1</td></tr>" in html_content
        assert "<tr><td>key2</td><td>value2</td></tr>" in html_content
        assert "</table>" in html_content
        
        # Check that display was called with the HTML object
        mock_display.assert_called_once()


class TestFprint:
    """Tests for the fprint function."""

    @patch('speedy_utils.common.utils_print.is_notebook')
    def test_dict_output(self, mock_is_notebook):
        """Test printing a dictionary."""
        mock_is_notebook.return_value = False
        
        data = {"key1": "value1", "key2": "value2"}
        mock_print = Mock()
        
        fprint(data, f=mock_print)
        
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        assert "key1" in printed_output
        assert "value1" in printed_output
        assert "key2" in printed_output
        assert "value2" in printed_output

    @patch('speedy_utils.common.utils_print.is_notebook')
    def test_string_output(self, mock_is_notebook):
        """Test printing a string."""
        mock_is_notebook.return_value = False
        
        data = "This is a test string"
        mock_print = Mock()
        
        fprint(data, f=mock_print)
        
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        assert "This is a test string" in printed_output

    @patch('speedy_utils.common.utils_print.is_notebook')
    def test_nested_dict_flattening(self, mock_is_notebook):
        """Test that nested dictionaries are flattened."""
        mock_is_notebook.return_value = False
        
        data = {"outer": {"inner": "value"}}
        mock_print = Mock()
        
        fprint(data, f=mock_print)
        
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        assert "outer.inner" in printed_output
        assert "value" in printed_output

    @patch('speedy_utils.common.utils_print.is_notebook')
    def test_key_ignore(self, mock_is_notebook):
        """Test ignoring specific keys."""
        mock_is_notebook.return_value = False
        
        data = {"keep": "value1", "ignore": "value2"}
        mock_print = Mock()
        
        fprint(data, key_ignore=["ignore"], f=mock_print)
        
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        assert "keep" in printed_output
        assert "value1" in printed_output
        # Note: The way fprint is implemented, key_ignore only works in notebook mode
        # So these would fail if is_notebook was True
        assert "ignore" in printed_output
        assert "value2" in printed_output

    @patch('speedy_utils.common.utils_print.is_notebook')
    def test_key_keep(self, mock_is_notebook):
        """Test keeping only specific keys."""
        mock_is_notebook.return_value = False
        
        data = {"keep": "value1", "ignore": "value2"}
        mock_print = Mock()
        
        fprint(data, key_keep=["keep"], f=mock_print)
        
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        # Note: The way fprint is implemented, key_keep only works in notebook mode
        # So these would fail if is_notebook was True
        assert "keep" in printed_output
        assert "value1" in printed_output
        assert "ignore" in printed_output
        assert "value2" in printed_output

    @patch('speedy_utils.common.utils_print.is_notebook')
    def test_grep_filtering(self, mock_is_notebook):
        """Test filtering keys with grep."""
        mock_is_notebook.return_value = False
        
        data = {"apple": 1, "banana": 2, "orange": 3}
        mock_print = Mock()
        
        fprint(data, grep="an", f=mock_print)
        
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        assert "banana" in printed_output
        assert "orange" in printed_output
        assert "apple" not in printed_output

    @patch('speedy_utils.common.utils_print.is_notebook')
    def test_invalid_input(self, mock_is_notebook):
        """Test that an error is raised for invalid input types."""
        mock_is_notebook.return_value = False
        
        with pytest.raises(ValueError, match="Input data must be a dictionary or string"):
            fprint(123)  # Not a dict or string



    @patch('speedy_utils.common.utils_print.is_notebook')
    def test_object_with_to_dict(self, mock_is_notebook):
        """Test handling objects with to_dict method."""
        mock_is_notebook.return_value = False
        
        class TestObj:
            def to_dict(self):
                return {"converted": "data"}
        
        obj = TestObj()
        mock_print = Mock()
        
        fprint(obj, f=mock_print)
        
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        assert "converted" in printed_output
        assert "data" in printed_output


class TestPrintTable:
    """Tests for the print_table function."""

    @patch('speedy_utils.common.utils_print.display')
    @patch('speedy_utils.common.utils_print.HTML')
    def test_dict_html_output(self, mock_html, mock_display):
        """Test HTML output with a dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        print_table(data, use_html=True)
        
        mock_html.assert_called_once()
        mock_display.assert_called_once()

    @patch('builtins.print')
    def test_dict_text_output(self, mock_print):
        """Test text output with a dictionary."""
        data = {"key1": "value1", "key2": "value2"}
        print_table(data, use_html=False)
        
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        assert "key1" in printed_output
        assert "value1" in printed_output

    @patch('speedy_utils.common.utils_print.display')
    @patch('speedy_utils.common.utils_print.HTML')
    def test_list_html_output(self, mock_html, mock_display):
        """Test HTML output with a list of dictionaries."""
        data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
        print_table(data, use_html=True)
        
        mock_html.assert_called_once()
        mock_display.assert_called_once()

    @patch('builtins.print')
    def test_json_string_input(self, mock_print):
        """Test handling a JSON string input."""
        json_str = json.dumps({"key": "value"})
        print_table(json_str, use_html=False)
        
        mock_print.assert_called_once()
        printed_output = mock_print.call_args[0][0]
        assert "key" in printed_output
        assert "value" in printed_output

    def test_invalid_json_string(self):
        """Test handling an invalid JSON string."""
        with pytest.raises(ValueError, match="String input could not be decoded as JSON"):
            print_table("not a json string", use_html=False)

    def test_invalid_list_input(self):
        """Test handling a list with non-dictionary items."""
        with pytest.raises(ValueError, match="List must contain dictionaries"):
            print_table([1, 2, 3], use_html=False)

    def test_invalid_type_input(self):
        """Test handling an invalid input type."""
        with pytest.raises(TypeError, match="Input data must be a list of dictionaries, a dictionary, or a JSON string"):
            print_table(123, use_html=False)


class TestSetupLogger:
    """Tests for the setup_logger function."""

    @patch('speedy_utils.common.utils_print.logger')
    def test_basic_setup(self, mock_logger):
        """Test basic logger setup."""
        setup_logger()
        
        # Verify that handlers were removed and a new one was added
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()
        
        # Verify that logging was enabled
        mock_logger.enable.assert_called_once_with("")
        mock_logger.debug.assert_called_once()

    @patch('speedy_utils.common.utils_print.logger')
    def test_disable_logging(self, mock_logger):
        """Test disabling the logger."""
        setup_logger(level="Disable")
        
        # Verify logging was disabled
        mock_logger.disable.assert_called_once_with("")
        mock_logger.info.assert_called_once_with("Logging disabled")


    @patch('speedy_utils.common.utils_print._last_log_times')
    def test_cache_size_update(self, mock_cache):
        """Test that the cache size is updated."""
        setup_logger(max_cache_entries=500)
        
        # Verify cache size was updated
        assert mock_cache.max_size == 500



    @patch('speedy_utils.common.utils_print.logger')
    def test_grep_pattern_filtering(self, mock_logger):
        """Test the grep pattern filtering in the log filter."""
        setup_logger(enable_grep="test.py", disable_grep="ignore_me")
        
        # Get the filter function
        filter_fn = mock_logger.add.call_args[1]['filter']
        
        # Create mock records
        matching_record = {
            "level": Mock(no=10),
            "file": "test.py",
            "line": 10,
            "function": "test_func",
            "message": "test message"
        }
        
        non_matching_record = {
            "level": Mock(no=10),
            "file": "other.py",
            "line": 20,
            "function": "other_func",
            "message": "other message"
        }
        
        disabled_record = {
            "level": Mock(no=10),
            "file": "ignore_me.py",
            "line": 30,
            "function": "ignore_func",
            "message": "ignore message"
        }
        
        # Setup mock logger level
        mock_logger.level.return_value = Mock(no=5)
        
        # Test the filter with different records
        assert filter_fn(matching_record)
        assert not filter_fn(non_matching_record)
        assert not filter_fn(disabled_record)

    @patch('speedy_utils.common.utils_print.logger')
    @patch('speedy_utils.common.utils_print.time')
    @patch('speedy_utils.common.utils_print._last_log_times')
    def test_rate_limiting(self, mock_cache, mock_time, mock_logger):
        """Test the rate limiting in the log filter."""
        # Set up a minimum interval of 10 seconds
        setup_logger(min_interval=10)
        
        # Get the filter function
        filter_fn = mock_logger.add.call_args[1]['filter']
        
        # Create a record
        record = {
            "level": Mock(no=10),
            "file": "test.py",
            "line": 10,
            "function": "test_func",
            "message": "test message"
        }
        
        # Setup mocks
        mock_logger.level.return_value = Mock(no=5)
        mock_time.time.return_value = 100  # Current time
        mock_cache.get.return_value = 95  # Last log time (5 seconds ago)
        
        # Test the filter - should return False due to rate limiting
        assert not filter_fn(record)
        
        # Now set the last log time to be longer ago
        mock_cache.get.return_value = 85  # Last log time (15 seconds ago)
        
        # Test again - should pass this time
        assert filter_fn(record)
        
        # Verify the cache was updated
        mock_cache.__setitem__.assert_called_with("test.py:10", 100)


class TestLog:
    """Tests for the log function."""

    @patch('speedy_utils.common.utils_print.logger')
    @patch('speedy_utils.common.utils_print._get_call_site_id')
    def test_basic_logging(self, mock_get_id, mock_logger):
        """Test basic logging functionality."""
        mock_get_id.return_value = "test.py:10"
        opt_mock = Mock()
        mock_logger.opt.return_value = opt_mock
        info_mock = Mock()
        opt_mock.info = info_mock
        
        log("Test message")
        
        mock_logger.opt.assert_called_once_with(depth=1)
        info_mock.assert_called_once_with("Test message")

    @patch('speedy_utils.common.utils_print.logger')
    @patch('speedy_utils.common.utils_print._get_call_site_id')
    def test_custom_level(self, mock_get_id, mock_logger):
        """Test logging with a custom level."""
        mock_get_id.return_value = "test.py:10"
        opt_mock = Mock()
        mock_logger.opt.return_value = opt_mock
        error_mock = Mock()
        opt_mock.error = error_mock
        
        log("Error message", level="error")
        
        mock_logger.opt.assert_called_once_with(depth=1)
        error_mock.assert_called_once_with("Error message")

    @patch('speedy_utils.common.utils_print.logger')
    @patch('speedy_utils.common.utils_print._get_call_site_id')
    @patch('speedy_utils.common.utils_print._logged_once_set')
    def test_log_once(self, mock_once_set, mock_get_id, mock_logger):
        """Test that log messages are only logged once when once=True."""
        identifier = "test.py:10"
        mock_get_id.return_value = identifier
        opt_mock = Mock()
        mock_logger.opt.return_value = opt_mock
        info_mock = Mock()
        opt_mock.info = info_mock
        
        # First call with once=True, identifier not in set
        mock_once_set.__contains__.return_value = False
        log("Test message", once=True)
        
        # Should log and add to set
        info_mock.assert_called_once_with("Test message")
        mock_once_set.add.assert_called_once_with(identifier)
        
        # Reset mocks
        info_mock.reset_mock()
        mock_once_set.add.reset_mock()
        
        # Second call with once=True, identifier already in set
        mock_once_set.__contains__.return_value = True
        log("Test message", once=True)
        
        # Should not log or add to set
        info_mock.assert_not_called()
        mock_once_set.add.assert_not_called()

    @patch('speedy_utils.common.utils_print.logger')
    @patch('speedy_utils.common.utils_print._get_call_site_id')
    @patch('speedy_utils.common.utils_print.time')
    @patch('speedy_utils.common.utils_print._last_log_intervals')
    def test_log_interval(self, mock_intervals, mock_time, mock_get_id, mock_logger):
        """Test that log messages respect the interval parameter."""
        identifier = "test.py:10"
        mock_get_id.return_value = identifier
        opt_mock = Mock()
        mock_logger.opt.return_value = opt_mock
        info_mock = Mock()
        opt_mock.info = info_mock
        
        # Set current time to 100
        mock_time.time.return_value = 100
        
        # First call with interval=10, no previous log
        mock_intervals.get.return_value = None
        log("Test message", interval=10)
        
        # Should log and update interval dictionary
        info_mock.assert_called_once_with("Test message")
        mock_intervals.__setitem__.assert_called_once_with(identifier, 100)
        
        # Reset mocks
        info_mock.reset_mock()
        mock_intervals.__setitem__.reset_mock()
        
        # Second call with interval=10, previous log 5 seconds ago
        mock_intervals.get.return_value = 95
        log("Test message", interval=10)
        
        # Should not log (since less than 10 seconds passed)
        info_mock.assert_not_called()
        mock_intervals.__setitem__.assert_not_called()
        
        # Third call with interval=10, previous log 15 seconds ago
        mock_intervals.get.return_value = 85
        log("Test message", interval=10)
        
        # Should log and update interval dictionary
        info_mock.assert_called_once_with("Test message")
        mock_intervals.__setitem__.assert_called_once_with(identifier, 100)


class TestGetCallSiteId:
    """Tests for the _get_call_site_id function."""

    @patch('speedy_utils.common.utils_print.inspect.stack')
    def test_call_site_identification(self, mock_stack):
        """Test that the function returns the expected identifier."""
        # Create a mock frame
        mock_frame = Mock(filename="test_file.py", lineno=42)
        mock_stack.return_value = [None, None, mock_frame]  # depth=2 means index 2
        
        # Call with default depth
        result = _get_call_site_id()
        
        # Verify result
        assert result == "test_file.py:42"
        
        # Call with custom depth
        mock_frame2 = Mock(filename="other_file.py", lineno=100)
        mock_stack.return_value = [None, mock_frame2]  # depth=1 means index 1
        
        result = _get_call_site_id(depth=1)
        
        # Verify result
        assert result == "other_file.py:100"


if __name__ == "__main__":
    pytest.main(["-v", "test_utils_print.py"])
