#!/usr/bin/env python3
"""
Test imemoize persistence across function redefinition (simulating IPython reload)
"""

import sys
import time
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), 'src'))

from speedy_utils.common.utils_cache import imemoize, _GLOBAL_MEMORY_CACHE

def test_ipython_reload_simulation():
    """
    Test that imemoize persists across function redefinition,
    simulating what happens in IPython when %load is used
    """
    print("=== Testing IPython reload simulation ===")
    
    # Clear any existing cache
    _GLOBAL_MEMORY_CACHE.clear()
    
    # First definition of the function
    @imemoize
    def computation_v1(x):
        print(f"Computing v1({x})")
        time.sleep(0.05)
        return x * x
    
    # Call it first time
    print("1. First call to v1:")
    result1 = computation_v1(10)
    print(f"   Result: {result1}")
    
    # Call it second time (should be cached)
    print("2. Second call to v1 (should be cached):")
    start = time.time()
    result2 = computation_v1(10)
    end = time.time()
    print(f"   Result: {result2}, Time: {end - start:.6f}s")
    
    assert result1 == result2
    assert end - start < 0.01  # Should be very fast (cached)
    
    # Check cache has content
    print(f"3. Cache now contains {len(_GLOBAL_MEMORY_CACHE)} entries")
    
    # Now simulate IPython %load - redefine the SAME function
    @imemoize 
    def computation_v1(x):  # Same name, same logic
        print(f"Computing v1({x})")
        time.sleep(0.05)
        return x * x
    
    # This should still be cached because source code is the same
    print("4. Call after 'reload' with same code (should be cached):")
    start = time.time()
    result3 = computation_v1(10)
    end = time.time()
    print(f"   Result: {result3}, Time: {end - start:.6f}s")
    
    assert result3 == result1
    assert end - start < 0.01  # Should still be fast (cached)
    
    # Now simulate a code change
    @imemoize
    def computation_v1(x):  # Same name, different logic
        print(f"Computing v1 modified({x})")
        time.sleep(0.05)
        return x * x * 2  # Different computation!
    
    # This should NOT be cached because source code changed
    print("5. Call after code change (should NOT be cached):")
    start = time.time()
    result4 = computation_v1(10)
    end = time.time()
    print(f"   Result: {result4}, Time: {end - start:.3f}s")
    
    assert result4 == 200  # New computation
    assert result4 != result1  # Different from before
    assert end - start > 0.04  # Should be slow (not cached)
    
    # Second call to modified function should now be cached
    print("6. Second call to modified function (should be cached):")
    start = time.time()
    result5 = computation_v1(10)
    end = time.time()
    print(f"   Result: {result5}, Time: {end - start:.6f}s")
    
    assert result5 == result4
    assert end - start < 0.01  # Should be fast (cached)
    
    print("✓ IPython reload simulation works correctly!")

def test_cross_function_caching():
    """Test that different functions have separate cache entries"""
    print("\n=== Testing cross-function caching ===")
    
    @imemoize
    def func_a(x):
        print(f"Computing func_a({x})")
        return x + 1
    
    @imemoize 
    def func_b(x):
        print(f"Computing func_b({x})")
        return x + 1  # Same logic, different function
    
    # Both should compute independently
    result_a = func_a(5)
    result_b = func_b(5)
    
    print(f"func_a(5) = {result_a}")
    print(f"func_b(5) = {result_b}")
    
    assert result_a == result_b == 6
    print("✓ Different functions cache independently!")

if __name__ == "__main__":
    test_ipython_reload_simulation()
    test_cross_function_caching()
    print("\n🎉 All persistence tests passed!")