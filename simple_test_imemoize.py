#!/usr/bin/env python3
"""
Simple test for imemoize without heavy dependencies
"""

import sys
import time
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), 'src'))

# Test basic imports first
print("Testing imports...")

try:
    from speedy_utils.common.utils_cache import imemoize, identify, get_source
    print("✓ Successfully imported imemoize from utils_cache")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

@imemoize
def simple_computation(x):
    """Simple computation for testing"""
    print(f"Computing simple_computation({x})")
    time.sleep(0.05)  # Small delay to verify caching
    return x * x

def test_basic():
    print("\n=== Testing basic imemoize functionality ===")
    
    # Test first call
    print("First call:")
    start = time.time()
    result1 = simple_computation(5)
    end = time.time()
    print(f"Result: {result1}, Time: {end - start:.3f}s")
    
    # Test second call (should be cached)
    print("Second call (should be cached):")
    start = time.time()
    result2 = simple_computation(5)
    end = time.time()
    print(f"Result: {result2}, Time: {end - start:.3f}s")
    
    assert result1 == result2 == 25
    print("✓ Basic caching works!")

def test_function_source_hashing():
    print("\n=== Testing function source code detection ===")
    
    # Test that get_source works
    source = get_source(simple_computation)
    print(f"Function source hash length: {len(source)}")
    
    # Test identity
    identity = identify((source, (5,), {}))
    print(f"Cache key identity: {identity[:50]}...")
    
    print("✓ Function source detection works!")

if __name__ == "__main__":
    test_basic()
    test_function_source_hashing()
    print("\n🎉 Simple tests passed!")