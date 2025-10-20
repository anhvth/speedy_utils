#!/usr/bin/env python3
"""
Proper test for imemoize with realistic timing expectations
"""

import sys
import time
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), 'src'))

from speedy_utils.common.utils_cache import imemoize, _GLOBAL_MEMORY_CACHE

def test_ipython_simulation():
    """Test imemoize behavior that simulates IPython %load scenario"""
    print("=== Testing IPython %load simulation ===")
    
    _GLOBAL_MEMORY_CACHE.clear()
    
    # Simulate first file load
    print("1. First 'file load' - define function:")
    @imemoize
    def expensive_computation(x):
        print(f"   [COMPUTING] expensive_computation({x})")
        time.sleep(0.1)  # Longer delay to be more obvious
        return x * x + x
    
    print("2. First execution:")
    start = time.time()
    result1 = expensive_computation(7)
    duration1 = time.time() - start
    print(f"   Result: {result1}, Duration: {duration1:.3f}s")
    
    print("3. Second execution (should be cached):")
    start = time.time()
    result2 = expensive_computation(7)
    duration2 = time.time() - start
    print(f"   Result: {result2}, Duration: {duration2:.3f}s")
    
    # Verify caching worked
    assert result1 == result2
    assert duration2 < 0.01, f"Expected cache hit to be fast, got {duration2:.3f}s"
    print("   ✓ Caching within same session works")
    
    # Simulate %load file.py (function redefinition with same source)
    print("\n4. Simulate '%load file.py' - redefine same function:")
    @imemoize
    def expensive_computation(x):
        print(f"   [COMPUTING] expensive_computation({x})")
        time.sleep(0.1)  # Same delay, same logic
        return x * x + x  # Same computation
    
    print("5. First call after 'reload' (should use cached result):")
    start = time.time()
    result3 = expensive_computation(7)
    duration3 = time.time() - start
    print(f"   Result: {result3}, Duration: {duration3:.3f}s")
    
    # This should be cached because source code is identical
    assert result3 == result1
    assert duration3 < 0.01, f"Expected reload to use cache, got {duration3:.3f}s"
    print("   ✓ Cache persists across function redefinition")
    
    # Simulate code change
    print("\n6. Simulate code change:")
    @imemoize
    def expensive_computation(x):
        print(f"   [COMPUTING] expensive_computation MODIFIED({x})")
        time.sleep(0.1)
        return x * x + x + 1  # Different computation!
    
    print("7. Call after code change (should recompute):")
    start = time.time()
    result4 = expensive_computation(7)
    duration4 = time.time() - start
    print(f"   Result: {result4}, Duration: {duration4:.3f}s")
    
    # Should recompute because code changed
    assert result4 == result1 + 1  # New computation
    assert duration4 > 0.05, f"Expected recomputation, got {duration4:.3f}s"
    print("   ✓ Code changes trigger recomputation")
    
    # Second call to modified function should be cached
    print("8. Second call to modified function (should be cached):")
    start = time.time()
    result5 = expensive_computation(7)
    duration5 = time.time() - start
    print(f"   Result: {result5}, Duration: {duration5:.3f}s")
    
    assert result5 == result4
    assert duration5 < 0.01, f"Expected cache hit, got {duration5:.3f}s"
    print("   ✓ Modified function caches correctly")
    
    print(f"\nFinal cache size: {len(_GLOBAL_MEMORY_CACHE)} entries")

def test_different_args():
    """Test that different arguments work correctly"""
    print("\n=== Testing different arguments ===")
    
    @imemoize
    def compute(a, b=1):
        print(f"   [COMPUTING] compute({a}, b={b})")
        return a * b
    
    # Different arguments should compute separately
    r1 = compute(5)       # a=5, b=1
    r2 = compute(5, 2)    # a=5, b=2  
    r3 = compute(6)       # a=6, b=1
    r4 = compute(5)       # a=5, b=1 (should be cached)
    
    assert r1 == 5
    assert r2 == 10
    assert r3 == 6
    assert r4 == 5  # Should equal r1
    print("   ✓ Different arguments handled correctly")

if __name__ == "__main__":
    test_ipython_simulation()
    test_different_args()
    print("\n🎉 All tests passed! imemoize works as expected.")