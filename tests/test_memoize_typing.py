"""Test that memoize preserves type information properly."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
from typing import Any, Protocol

try:
    from openai import AsyncOpenAI, OpenAI
    from llm_utils.lm.openai_memoize import MAsyncOpenAI, MOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    MOpenAI = None  # type: ignore[assignment,misc]
    MAsyncOpenAI = None  # type: ignore[assignment,misc]
    OPENAI_AVAILABLE = False

from speedy_utils.common.utils_cache import memoize


def test_basic_memoize_typing():
    """Test that memoize preserves basic function signatures."""

    @memoize
    def add(x: int, y: int) -> int:
        return x + y

    # Type information should be preserved
    result = add(1, 2)
    assert result == 3
    assert isinstance(result, int)


def test_method_memoize_typing():
    """Test that memoize works correctly on methods."""

    class Calculator:
        @memoize
        def multiply(self, x: int, y: int) -> int:
            return x * y

    calc = Calculator()
    result = calc.multiply(3, 4)
    assert result == 12
    assert isinstance(result, int)


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
def test_openai_memoize_no_cast():
    """Test that MOpenAI works without needing cast."""

    # This should work without any type errors
    # Note: We can't actually test with real API calls in tests
    # but we can at least verify the class initializes correctly
    try:
        assert MOpenAI is not None
        client = MOpenAI(api_key='dummy-key', cache=True)
        # The post method should be properly typed
        assert hasattr(client, 'post')
        assert callable(client.post)
        print('✅ MOpenAI initialization with memoization successful')
    except Exception as e:
        print(f'❌ MOpenAI initialization failed: {e}')
        raise


@pytest.mark.skipif(not OPENAI_AVAILABLE, reason="OpenAI not available")
def test_async_openai_memoize_no_cast():
    """Test that MAsyncOpenAI works without needing cast."""

    try:
        assert MAsyncOpenAI is not None
        client = MAsyncOpenAI(api_key='dummy-key', cache=True)
        # The post method should be properly typed
        assert hasattr(client, 'post')
        assert callable(client.post)
        print('✅ MAsyncOpenAI initialization with memoization successful')
    except Exception as e:
        print(f'❌ MAsyncOpenAI initialization failed: {e}')
        raise


if __name__ == '__main__':
    test_basic_memoize_typing()
    test_method_memoize_typing()
    test_openai_memoize_no_cast()
    test_async_openai_memoize_no_cast()
    print('🎉 All tests passed!')
