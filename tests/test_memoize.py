"""Tests for memoize and imemoize decorators."""

import asyncio
from pathlib import Path

import pytest

from speedy_utils.common.utils_cache import imemoize, memoize


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path):
    """Provide a temporary cache directory."""
    return str(tmp_path / "cache")


@pytest.fixture(autouse=True)
def clear_global_memory_cache():
    """Clear the global memory cache before and after each test."""
    from speedy_utils.common import utils_cache

    utils_cache._GLOBAL_MEMORY_CACHE.clear()
    yield
    utils_cache._GLOBAL_MEMORY_CACHE.clear()


# ----------------------------------------------------------------------
# Basic memoize tests (memory cache)
# ----------------------------------------------------------------------


def test_memoize_same_args_returns_cached_result():
    """Test that same arguments return the cached result."""

    call_count = 0

    @memoize(cache_type="memory")
    def expensive_computation(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * x

    result1 = expensive_computation(5)
    result2 = expensive_computation(5)

    assert result1 == 25
    assert result2 == 25
    assert call_count == 1


def test_memoize_different_args_cache_miss():
    """Test that different arguments cause cache misses."""

    call_count = 0

    @memoize(cache_type="memory")
    def expensive_computation(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * x

    result1 = expensive_computation(5)
    result2 = expensive_computation(10)

    assert result1 == 25
    assert result2 == 100
    assert call_count == 2


def test_memoize_with_string_args():
    """Test memoization with string arguments."""
    call_count = 0

    @memoize(cache_type="memory")
    def process_string(s: str) -> str:
        nonlocal call_count
        call_count += 1
        return s.upper()

    result1 = process_string("hello")
    result2 = process_string("hello")
    result3 = process_string("world")

    assert result1 == "HELLO"
    assert result2 == "HELLO"
    assert result3 == "WORLD"
    assert call_count == 2


def test_memoize_with_list_args():
    """Test memoization with list arguments."""
    call_count = 0

    @memoize(cache_type="memory")
    def process_list(lst: list) -> int:
        nonlocal call_count
        call_count += 1
        return sum(lst)

    result1 = process_list([1, 2, 3])
    result2 = process_list([1, 2, 3])
    result3 = process_list([4, 5])

    assert result1 == 6
    assert result2 == 6
    assert result3 == 9
    assert call_count == 2


def test_memoize_with_dict_args():
    """Test memoization with dict arguments."""
    call_count = 0

    @memoize(cache_type="memory")
    def process_dict(d: dict) -> int:
        nonlocal call_count
        call_count += 1
        return sum(d.values())

    result1 = process_dict({"a": 1, "b": 2})
    result2 = process_dict({"a": 1, "b": 2})
    result3 = process_dict({"x": 10})

    assert result1 == 3
    assert result2 == 3
    assert result3 == 10
    assert call_count == 2


# ----------------------------------------------------------------------
# Disk cache tests
# ----------------------------------------------------------------------


def test_memoize_disk_cache_persists(cache_dir):
    """Test that disk cache persists across calls."""
    from speedy_utils.common import utils_cache

    # Clear any existing caches to ensure clean state
    utils_cache._MEM_CACHES.clear()

    call_count = 0

    @memoize(cache_type="disk", cache_dir=cache_dir)
    def expensive_computation(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * x

    result1 = expensive_computation(5)

    # Clear the memory cache to force disk read
    utils_cache._MEM_CACHES.clear()

    result2 = expensive_computation(5)

    assert result1 == 25
    assert result2 == 25
    assert call_count == 1


def test_memoize_disk_and_memory_cache(cache_dir):
    """Test both memory and disk caching together."""
    from speedy_utils.common import utils_cache

    utils_cache._MEM_CACHES.clear()

    call_count = 0

    @memoize(cache_type="both", cache_dir=cache_dir, size=10)
    def expensive_computation(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * x

    result1 = expensive_computation(5)
    result2 = expensive_computation(5)

    # Clear memory cache but disk cache should still work
    utils_cache._MEM_CACHES.clear()

    result3 = expensive_computation(5)

    assert result1 == 25
    assert result2 == 25
    assert result3 == 25
    assert call_count == 1


# ----------------------------------------------------------------------
# imemoize tests
# ----------------------------------------------------------------------


def test_imemoize_basic():
    """Test basic imemoize functionality."""
    call_count = 0

    @imemoize
    def expensive_computation(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * x

    result1 = expensive_computation(5)
    result2 = expensive_computation(5)

    assert result1 == 25
    assert result2 == 25
    assert call_count == 1


def test_imemoize_different_args():
    """Test imemoize with different arguments."""
    call_count = 0

    @imemoize
    def expensive_computation(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * x

    result1 = expensive_computation(5)
    result2 = expensive_computation(10)

    assert result1 == 25
    assert result2 == 100
    assert call_count == 2


def test_imemoize_with_key_parameter():
    """Test imemoize with custom key function."""
    call_count = 0

    @imemoize(key=lambda x: x)
    def expensive_computation(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * x

    result1 = expensive_computation(5)
    result2 = expensive_computation(5)

    assert result1 == 25
    assert result2 == 25
    assert call_count == 1


# ----------------------------------------------------------------------
# Key function tests
# ----------------------------------------------------------------------


def test_memoize_custom_key_function():
    """Test memoize with custom key function."""
    call_count = 0

    @memoize(cache_type="memory", key=lambda x: x)
    def expensive_computation(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x * x

    result1 = expensive_computation(5)
    result2 = expensive_computation(5)

    assert result1 == 25
    assert result2 == 25
    assert call_count == 1


# ----------------------------------------------------------------------
# Method memoization tests
# ----------------------------------------------------------------------


def test_memoize_method_with_ignore_self():
    """Test memoization on methods with ignore_self=True."""

    class Calculator:
        @memoize(cache_type="memory", ignore_self=True)
        def multiply(self, x: int, y: int) -> int:
            return x * y

    calc = Calculator()
    result1 = calc.multiply(3, 4)
    result2 = calc.multiply(3, 4)

    assert result1 == 12
    assert result2 == 12


def test_imemoize_method_with_ignore_self():
    """Test imemoize on methods with ignore_self=True."""

    class Calculator:
        @imemoize(ignore_self=True)
        def multiply(self, x: int, y: int) -> int:
            return x * y

    calc = Calculator()
    result1 = calc.multiply(3, 4)
    result2 = calc.multiply(3, 4)

    assert result1 == 12
    assert result2 == 12