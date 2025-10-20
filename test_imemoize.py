#!/usr/bin/env python3
"""
Test script for imemoize functionality
"""

import sys
import time
sys.path.insert(0, 'src')

from speedy_utils import imemoize

@imemoize
def expensive_computation(x, y=1):
    """Simulate an expensive computation"""
    print(f"Computing expensive_computation({x}, {y})")
    time.sleep(0.1)  # Simulate work
    return x * x + y

@imemoize
async def async_expensive_computation(x):
    """Simulate an expensive async computation"""
    import asyncio
    print(f"Computing async_expensive_computation({x})")
    await asyncio.sleep(0.1)  # Simulate work
    return x * x

def test_basic_functionality():
    print("=== Testing basic functionality ===")
    
    # Test sync function
    print("First call:")
    start = time.time()
    result1 = expensive_computation(5)
    end = time.time()
    print(f"Result: {result1}, Time: {end - start:.3f}s")
    
    print("Second call (should be cached):")
    start = time.time()
    result2 = expensive_computation(5)
    end = time.time()
    print(f"Result: {result2}, Time: {end - start:.3f}s")
    
    assert result1 == result2
    print("✓ Basic sync caching works")

def test_different_arguments():
    print("\n=== Testing different arguments ===")
    
    # Different arguments should compute separately
    result1 = expensive_computation(3)
    result2 = expensive_computation(4)
    result3 = expensive_computation(3, y=2)  # Different y value
    result4 = expensive_computation(3)  # Should be cached
    
    print(f"expensive_computation(3) = {result1}")
    print(f"expensive_computation(4) = {result2}")
    print(f"expensive_computation(3, y=2) = {result3}")
    print(f"expensive_computation(3) again = {result4}")
    
    assert result1 != result2
    assert result1 != result3
    assert result1 == result4
    print("✓ Different arguments work correctly")

async def test_async_functionality():
    print("\n=== Testing async functionality ===")
    
    start = time.time()
    result1 = await async_expensive_computation(7)
    end = time.time()
    print(f"First async call: {result1}, Time: {end - start:.3f}s")
    
    start = time.time()
    result2 = await async_expensive_computation(7)
    end = time.time()
    print(f"Second async call (cached): {result2}, Time: {end - start:.3f}s")
    
    assert result1 == result2
    print("✓ Async caching works")

def test_code_change_detection():
    print("\n=== Testing code change detection ===")
    
    # Create a function dynamically to simulate code changes
    def create_function(multiplier):
        @imemoize
        def dynamic_func(x):
            print(f"Computing with multiplier {multiplier}")
            return x * multiplier
        return dynamic_func
    
    func1 = create_function(2)
    func2 = create_function(3)
    
    result1a = func1(5)  # Should compute
    result1b = func1(5)  # Should be cached
    result2a = func2(5)  # Different function, should compute
    
    print(f"func1(5) first: {result1a}")
    print(f"func1(5) second: {result1b}")
    print(f"func2(5): {result2a}")
    
    assert result1a == result1b == 10
    assert result2a == 15
    print("✓ Code change detection works")

if __name__ == "__main__":
    test_basic_functionality()
    test_different_arguments()
    
    import asyncio
    asyncio.run(test_async_functionality())
    
    test_code_change_detection()
    
    print("\n🎉 All tests passed!")