#!/usr/bin/env python3
"""
Debug the imemoize cache key generation
"""

import sys
import time
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), 'src'))

from speedy_utils.common.utils_cache import imemoize, _GLOBAL_MEMORY_CACHE, get_source, identify, _compute_cache_components

def debug_cache_keys():
    """Debug what cache keys are generated"""
    print("=== Debugging cache key generation ===")
    
    # Clear cache
    _GLOBAL_MEMORY_CACHE.clear()
    
    # First function
    @imemoize
    def test_func(x):
        return x * x
    
    # Get cache components
    func_source1, sub_dir1, key_id1 = _compute_cache_components(
        test_func, (5,), {}, True, None, None
    )
    cache_key1 = identify((func_source1, sub_dir1, key_id1))
    
    print(f"Function 1 source length: {len(func_source1)}")
    print(f"Function 1 source: {func_source1[:100]}...")
    print(f"Function 1 cache key: {cache_key1}")
    
    # Call function
    result1 = test_func(5)
    print(f"After call 1, cache size: {len(_GLOBAL_MEMORY_CACHE)}")
    
    # Redefine same function 
    @imemoize
    def test_func(x):
        return x * x
    
    # Get cache components for redefined function
    func_source2, sub_dir2, key_id2 = _compute_cache_components(
        test_func, (5,), {}, True, None, None
    )
    cache_key2 = identify((func_source2, sub_dir2, key_id2))
    
    print(f"\nFunction 2 source length: {len(func_source2)}")
    print(f"Function 2 source: {func_source2[:100]}...")
    print(f"Function 2 cache key: {cache_key2}")
    
    print(f"\nSource code same: {func_source1 == func_source2}")
    print(f"Cache keys same: {cache_key1 == cache_key2}")
    
    # Call function again
    result2 = test_func(5)
    print(f"After call 2, cache size: {len(_GLOBAL_MEMORY_CACHE)}")
    
    print(f"\nCache contents:")
    for i, (key, value) in enumerate(_GLOBAL_MEMORY_CACHE.items()):
        print(f"  {i+1}. {key[:50]}... -> {value}")

if __name__ == "__main__":
    debug_cache_keys()