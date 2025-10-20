#!/usr/bin/env python3
"""
Debug the imemoize execution flow
"""

import sys
import time
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(__file__), 'src'))

from speedy_utils.common.utils_cache import _GLOBAL_MEMORY_CACHE, identify, _compute_cache_components
from speedy_utils.common.utils_cache import mem_lock

def create_debug_imemoize():
    """Create a debug version of imemoize with detailed logging"""
    import functools
    import inspect
    
    def debug_imemoize(func):
        print(f"Creating imemoize wrapper for {func.__name__}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n--- Calling {func.__name__}({args}, {kwargs}) ---")
            
            # Compute cache key
            func_source, sub_dir, key_id = _compute_cache_components(
                func, args, kwargs, True, None, None
            )
            cache_key = identify((func_source, sub_dir, key_id))
            
            print(f"Function source: {func_source[:50]}...")
            print(f"Cache key: {cache_key}")
            print(f"Cache size before lookup: {len(_GLOBAL_MEMORY_CACHE)}")
            
            # Check cache
            with mem_lock:
                if cache_key in _GLOBAL_MEMORY_CACHE:
                    cached_result = _GLOBAL_MEMORY_CACHE[cache_key]
                    print(f"CACHE HIT: Found {cached_result}")
                    return cached_result
                else:
                    print(f"CACHE MISS: Key not found")
                    print(f"Available keys:")
                    for i, key in enumerate(_GLOBAL_MEMORY_CACHE.keys()):
                        print(f"  {i+1}. {key}")
            
            # Compute result
            print(f"Computing result...")
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"Computation took {end - start:.6f}s")
            
            # Store in cache
            with mem_lock:
                _GLOBAL_MEMORY_CACHE[cache_key] = result
                print(f"Stored result in cache. Cache size now: {len(_GLOBAL_MEMORY_CACHE)}")
            
            return result
        
        return wrapper
    
    return debug_imemoize

def test_debug():
    debug_imemoize = create_debug_imemoize()
    
    # Clear cache
    _GLOBAL_MEMORY_CACHE.clear()
    print("Cache cleared")
    
    # First function
    @debug_imemoize
    def test_func(x):
        time.sleep(0.05)
        return x * x
    
    print("\n=== FIRST CALL ===")
    result1 = test_func(5)
    
    print("\n=== SECOND CALL (same function object) ===")
    result2 = test_func(5)
    
    # Redefine function
    @debug_imemoize
    def test_func(x):
        time.sleep(0.05) 
        return x * x
    
    print("\n=== THIRD CALL (new function object, same source) ===")
    result3 = test_func(5)

if __name__ == "__main__":
    test_debug()