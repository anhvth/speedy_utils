"""Tests for load_jsonl and fast_load_jsonl functions."""

import json
from pathlib import Path

import pytest

from speedy_utils.common.utils_io import fast_load_jsonl, load_jsonl


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def simple_jsonl_file(tmp_path):
    """Create a simple JSONL file for testing."""
    file_path = tmp_path / "simple.jsonl"
    data = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Charlie", "age": 35},
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return file_path


@pytest.fixture
def empty_jsonl_file(tmp_path):
    """Create an empty JSONL file for testing."""
    file_path = tmp_path / "empty.jsonl"
    file_path.write_text("", encoding="utf-8")
    return file_path


@pytest.fixture
def jsonl_with_various_types(tmp_path):
    """Create a JSONL file with various JSON value types."""
    file_path = tmp_path / "various.jsonl"
    data = [
        {"type": "string", "value": "hello"},
        {"type": "number", "value": 42.5},
        {"type": "integer", "value": 42},
        {"type": "boolean", "value": True},
        {"type": "null", "value": None},
        {"type": "array", "value": [1, 2, 3]},
        {"type": "nested", "value": {"a": {"b": {"c": 1}}}},
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return file_path


# ----------------------------------------------------------------------
# load_jsonl tests
# ----------------------------------------------------------------------


def test_load_jsonl_simple(simple_jsonl_file):
    """Test loading a simple JSONL file."""
    result = load_jsonl(simple_jsonl_file)

    assert len(result) == 3
    assert result[0] == {"name": "Alice", "age": 30}
    assert result[1] == {"name": "Bob", "age": 25}
    assert result[2] == {"name": "Charlie", "age": 35}


def test_load_jsonl_empty(empty_jsonl_file):
    """Test loading an empty JSONL file."""
    result = load_jsonl(empty_jsonl_file)

    assert result == []


def test_load_jsonl_file_not_found():
    """Test error handling when file is not found."""
    with pytest.raises(Exception):
        load_jsonl("/nonexistent/path/file.jsonl")


# ----------------------------------------------------------------------
# fast_load_jsonl tests
# ----------------------------------------------------------------------


def test_fast_load_jsonl_simple(simple_jsonl_file):
    """Test fast_load_jsonl with a simple file."""
    result = list(fast_load_jsonl(simple_jsonl_file))

    assert len(result) == 3
    assert result[0] == {"name": "Alice", "age": 30}
    assert result[1] == {"name": "Bob", "age": 25}
    assert result[2] == {"name": "Charlie", "age": 35}


def test_fast_load_jsonl_empty(empty_jsonl_file):
    """Test fast_load_jsonl with an empty file."""
    result = list(fast_load_jsonl(empty_jsonl_file))

    assert result == []


def test_fast_load_jsonl_various_types(jsonl_with_various_types):
    """Test fast_load_jsonl with various JSON value types."""
    result = list(fast_load_jsonl(jsonl_with_various_types))

    assert len(result) == 7

    # String
    assert result[0] == {"type": "string", "value": "hello"}
    # Number (float)
    assert result[1] == {"type": "number", "value": 42.5}
    # Integer
    assert result[2] == {"type": "integer", "value": 42}
    # Boolean
    assert result[3] == {"type": "boolean", "value": True}
    # Null
    assert result[4] == {"type": "null", "value": None}
    # Array
    assert result[5] == {"type": "array", "value": [1, 2, 3]}
    # Nested dict
    assert result[6] == {"type": "nested", "value": {"a": {"b": {"c": 1}}}}


def test_fast_load_jsonl_file_not_found():
    """Test error handling when file is not found."""
    with pytest.raises(Exception):
        list(fast_load_jsonl("/nonexistent/path/file.jsonl"))


def test_fast_load_jsonl_with_max_lines(simple_jsonl_file):
    """Test fast_load_jsonl with max_lines parameter."""
    result = list(fast_load_jsonl(simple_jsonl_file, max_lines=2))

    assert len(result) == 2
    assert result[0] == {"name": "Alice", "age": 30}
    assert result[1] == {"name": "Bob", "age": 25}


def test_fast_load_jsonl_skip_empty_lines(tmp_path):
    """Test fast_load_jsonl skips empty lines by default."""
    file_path = tmp_path / "with_empty.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"name": "Alice"}) + "\n")
        f.write("\n")
        f.write(json.dumps({"name": "Bob"}) + "\n")
        f.write("   \n")
        f.write(json.dumps({"name": "Charlie"}) + "\n")

    result = list(fast_load_jsonl(file_path))

    assert len(result) == 3
    assert result[0] == {"name": "Alice"}
    assert result[1] == {"name": "Bob"}
    assert result[2] == {"name": "Charlie"}


def test_fast_load_jsonl_on_error_raise(tmp_path):
    """Test fast_load_jsonl with on_error='raise' on malformed lines."""
    file_path = tmp_path / "malformed.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"valid": True}) + "\n")
        f.write("not valid json\n")
        f.write(json.dumps({"also": "valid"}) + "\n")

    with pytest.raises(Exception):
        list(fast_load_jsonl(file_path, on_error="raise"))


def test_fast_load_jsonl_on_error_skip(tmp_path):
    """Test fast_load_jsonl with on_error='skip' on malformed lines."""
    file_path = tmp_path / "malformed.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"valid": True}) + "\n")
        f.write("not valid json\n")
        f.write(json.dumps({"also": "valid"}) + "\n")

    result = list(fast_load_jsonl(file_path, on_error="skip"))

    assert len(result) == 2
    assert result[0] == {"valid": True}
    assert result[1] == {"also": "valid"}


def test_fast_load_jsonl_preserves_line_order(simple_jsonl_file):
    """Test that fast_load_jsonl preserves the order of lines."""
    result = list(fast_load_jsonl(simple_jsonl_file))

    for i, item in enumerate(result):
        if i == 0:
            assert item["name"] == "Alice"
        elif i == 1:
            assert item["name"] == "Bob"
        elif i == 2:
            assert item["name"] == "Charlie"


# ----------------------------------------------------------------------
# Both functions should produce equivalent results
# ----------------------------------------------------------------------


def test_load_jsonl_and_fast_load_jsonl_equivalence(simple_jsonl_file):
    """Test that load_jsonl and fast_load_jsonl produce the same results."""
    result_load_jsonl = load_jsonl(simple_jsonl_file)
    result_fast_load_jsonl = list(fast_load_jsonl(simple_jsonl_file))

    assert result_load_jsonl == result_fast_load_jsonl