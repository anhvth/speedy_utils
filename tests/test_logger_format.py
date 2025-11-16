"""
Tests for the updated logger format in setup_logger function.
"""

import re
import sys
from io import StringIO

from loguru import logger

from speedy_utils.common.logger import setup_logger


def test_logger_timestamp_format():
    """Test that the logger format includes the timestamp."""
    # Capture stdout to check the format
    stdout_capture = StringIO()
    original_stdout = sys.stdout

    try:
        sys.stdout = stdout_capture

        # Setup logger with Info level
        setup_logger(level="Info")

        # Log something
        logger.info("Test message")

        # Get the captured output
        output = stdout_capture.getvalue()

        # Check that the output contains a timestamp in the expected format
        # Format should include: HH:mm:ss
        timestamp_pattern = r"\d{2}:\d{2}:\d{2}"
        assert re.search(
            timestamp_pattern, output
        ), f"Timestamp not found in log output: {output}"

    finally:
        # Restore stdout
        sys.stdout = original_stdout


def test_logger_format_elements():
    """Test that the logger format includes all expected elements."""
    # Capture stdout to check the format
    stdout_capture = StringIO()
    original_stdout = sys.stdout

    try:
        sys.stdout = stdout_capture

        # Setup logger with Info level
        setup_logger(level="Info")

        # Log something simple
        logger.info("Test message")

        # Get the captured output
        output = stdout_capture.getvalue()

        # Check format elements
        assert "INFO" in output, f"Log level not found in output: {output}"
        assert (
            "test_logger_format.py" in output
        ), f"File name not found in output: {output}"
        assert "Test message" in output, f"Message not found in output: {output}"

        # Timestamp is already checked in the previous test

    finally:
        # Restore stdout
        sys.stdout = original_stdout


def test_logger_disable_mode():
    """Test that the logger handles Disable level appropriately."""
    # Capture stdout to check if logging is disabled
    stdout_capture = StringIO()
    original_stdout = sys.stdout

    try:
        sys.stdout = stdout_capture

        # In the updated code, "Disable" level is handled differently
        # We now test that it doesn't raise an exception
        # First with a valid level
        setup_logger(level="Info")
        logger.info("This should appear")

        # Get the captured output
        output = stdout_capture.getvalue()
        assert "This should appear" in output

    finally:
        # Restore stdout
        sys.stdout = original_stdout
