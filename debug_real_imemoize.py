#!/usr/bin/env python3
"""
Debug the real imemoize to see what's happening
"""

import sys
import time
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), 'src'))

from speedy_utils.common.utils_cache import imemoize, _GLOBAL_MEMORY_CACHE

# Monkey patch to add debugging
original_imemoize = imemoize

def debug_imemoize_call(func_name, cache_key, action, result=None):
    """Debug helper"""
    if action == "check":
        in_cache = cache_key in _GLOBAL_MEMORY_CACHE
        print(f"[{func_name}] CHECK cache key {cache_key[:16]}... -> {'HIT' if in_cache else 'MISS'}")
        return in_cache
    elif action == "store":
        print(f"[{func_name}] STORE cache key {cache_key[:16]}... -> {result}")

# Temporarily patch the global cache access
import speedy_utils.common.utils_cache as cache_module
original_global_cache = cache_module._GLOBAL_MEMORY_CACHE

def test_real_imemoize():
    print("=== Testing real imemoize with debugging ===")
    
    # Clear cache  
    _GLOBAL_MEMORY_CACHE.clear()
    print("Cache cleared")
    
    # First function definition
    @imemoize
    def test_computation(x):
        print(f"    COMPUTING test_computation({x})")
        time.sleep(0.05)
        return x * x
    
    print(f"\n1. First call:")
    start = time.time()
    result1 = test_computation(10)
    end = time.time()
    print(f"   Result: {result1}, Time: {end - start:.6f}s")
    print(f"   Cache size: {len(_GLOBAL_MEMORY_CACHE)}")
    
    print(f"\n2. Second call (same function object):")
    start = time.time()
    result2 = test_computation(10)
    end = time.time()
    print(f"   Result: {result2}, Time: {end - start:.6f}s")
    print(f"   Cache size: {len(_GLOBAL_MEMORY_CACHE)}")
    
    # Redefine the function
    print(f"\n3. Redefining function...")
    @imemoize
    def test_computation(x):
        print(f"    COMPUTING test_computation({x})")
        time.sleep(0.05)
        return x * x
    
    print(f"4. Call after redefinition:")
    start = time.time()
    result3 = test_computation(10)
    end = time.time()
    print(f"   Result: {result3}, Time: {end - start:.6f}s")
    print(f"   Cache size: {len(_GLOBAL_MEMORY_CACHE)}")
    
    # Check cache contents
    print(f"\nCache contents:")
    for i, (key, value) in enumerate(_GLOBAL_MEMORY_CACHE.items()):
        print(f"  {i+1}. {key[:30]}... -> {value}")

if __name__ == "__main__":
    test_real_imemoize()