"""Tests for identify function."""

import json

import pytest

from speedy_utils.common.utils_cache import identify


# ----------------------------------------------------------------------
# Basic type tests
# ----------------------------------------------------------------------


def test_identify_basic_string():
    """Test identify with basic string."""
    result = identify("hello")
    assert isinstance(result, str)
    assert len(result) > 0


def test_identify_basic_int():
    """Test identify with basic integer."""
    result1 = identify(42)
    result2 = identify(42)

    assert isinstance(result1, str)
    assert result1 == result2


def test_identify_basic_float():
    """Test identify with float."""
    result1 = identify(3.14)
    result2 = identify(3.14)

    assert isinstance(result1, str)
    assert result1 == result2


def test_identify_boolean():
    """Test identify with boolean values."""
    result_true = identify(True)
    result_false = identify(False)

    assert isinstance(result_true, str)
    assert isinstance(result_false, str)
    assert result_true != result_false


def test_identify_none():
    """Test identify with None."""
    result = identify(None)

    assert isinstance(result, str)
    assert len(result) > 0


# ----------------------------------------------------------------------
# Container type tests
# ----------------------------------------------------------------------


def test_identify_list():
    """Test identify with list."""
    result1 = identify([1, 2, 3])
    result2 = identify([1, 2, 3])

    assert isinstance(result1, str)
    assert result1 == result2


def test_identify_different_lists():
    """Test that different lists produce different identifiers."""
    result1 = identify([1, 2, 3])
    result2 = identify([1, 2, 4])

    assert result1 != result2


def test_identify_tuple():
    """Test identify with tuple."""
    result1 = identify((1, 2, 3))
    result2 = identify((1, 2, 3))

    assert isinstance(result1, str)
    assert result1 == result2


def test_identify_dict():
    """Test identify with dictionary."""
    result1 = identify({"a": 1, "b": 2})
    result2 = identify({"a": 1, "b": 2})

    assert isinstance(result1, str)
    assert result1 == result2


def test_identify_different_dicts():
    """Test that different dicts produce different identifiers."""
    result1 = identify({"a": 1, "b": 2})
    result2 = identify({"a": 1, "b": 3})

    assert result1 != result2


# ----------------------------------------------------------------------
# Nested structure tests
# ----------------------------------------------------------------------


def test_identify_nested_list():
    """Test identify with nested list."""
    result1 = identify([[1, 2], [3, 4]])
    result2 = identify([[1, 2], [3, 4]])

    assert isinstance(result1, str)
    assert result1 == result2


def test_identify_nested_dict():
    """Test identify with nested dictionary."""
    result1 = identify({"a": {"b": {"c": 1}}})
    result2 = identify({"a": {"b": {"c": 1}}})

    assert isinstance(result1, str)
    assert result1 == result2


def test_identify_mixed_nested():
    """Test identify with mixed nested structures."""
    data = {"list": [1, 2, {"nested": "value"}], "tuple": (3, 4)}
    result1 = identify(data)
    result2 = identify(data)

    assert isinstance(result1, str)
    assert result1 == result2


# ----------------------------------------------------------------------
# Stability tests
# ----------------------------------------------------------------------


def test_identify_stability_same_input():
    """Test that identify returns the same output for the same input."""
    data = {"key": "value", "number": 42}

    result1 = identify(data)
    result2 = identify(data)
    result3 = identify(data)

    assert result1 == result2 == result3


def test_identify_stability_across_calls():
    """Test identify stability across multiple calls with same data."""

    def get_identify_result():
        return identify({"a": 1, "b": [1, 2, 3]})

    result1 = get_identify_result()
    result2 = get_identify_result()
    result3 = get_identify_result()

    assert result1 == result2 == result3


# ----------------------------------------------------------------------
# Different input produces different output
# ----------------------------------------------------------------------


def test_identify_different_inputs_different_outputs():
    """Test that different inputs produce different identifiers."""
    inputs = [
        "hello",
        "world",
        42,
        42.0,
        True,
        False,
        None,
        [1, 2, 3],
        [3, 2, 1],
        {"a": 1},
        {"b": 1},
        (1, 2),
        (2, 1),
    ]

    results = [identify(x) for x in inputs]
    assert len(results) == len(set(results)), "All different inputs should have unique identifiers"


def test_identify_string_vs_int():
    """Test that string and integer with same value produce different identifiers."""
    result_str = identify("42")
    result_int = identify(42)

    assert result_str != result_int


# ----------------------------------------------------------------------
# Key type handling tests
# ----------------------------------------------------------------------


def test_identify_with_int_keys():
    """Test identify with integer dictionary keys."""
    result1 = identify({1: "one", 2: "two"})
    result2 = identify({1: "one", 2: "two"})

    assert result1 == result2


def test_identify_with_mixed_keys():
    """Test identify with mixed type dictionary keys."""
    result1 = identify({1: "one", "two": 2})
    result2 = identify({1: "one", "two": 2})

    assert result1 == result2


def test_identify_empty_container():
    """Test identify with empty containers."""
    result_empty_list = identify([])
    result_empty_tuple = identify(())
    result_empty_dict = identify({})

    assert isinstance(result_empty_list, str)
    assert isinstance(result_empty_tuple, str)
    assert isinstance(result_empty_dict, str)
    # All should be valid non-empty strings
    assert len(result_empty_list) > 0
    assert len(result_empty_tuple) > 0
    assert len(result_empty_dict) > 0


# ----------------------------------------------------------------------
# Dict key ordering independence
# ----------------------------------------------------------------------


def test_identify_dict_key_order_independence():
    """Test that dict key order doesn't affect identify result."""
    # Create two dicts with same keys/values but different insertion order
    dict1 = {}
    dict1["b"] = 1
    dict1["a"] = 2

    dict2 = {}
    dict2["a"] = 2
    dict2["b"] = 1

    result1 = identify(dict1)
    result2 = identify(dict2)

    assert result1 == result2