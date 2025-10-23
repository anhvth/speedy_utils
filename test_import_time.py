#!/usr/bin/env python3
"""
Test script to measure import times for speedy_utils and identify slow components.
"""

import time
import sys
import importlib
from typing import Dict, List, Tuple, Optional


def time_import(module_name: str, from_list: Optional[List[str]] = None) -> float:
    """Time the import of a module or specific items from a module."""
    start_time = time.time()
    
    try:
        if from_list:
            # Import specific items from module
            module = importlib.import_module(module_name)
            for item in from_list:
                getattr(module, item)
        else:
            # Import entire module
            importlib.import_module(module_name)
        
        end_time = time.time()
        return end_time - start_time
    except Exception as e:
        print(f"Error importing {module_name}: {e}")
        return -1


def test_individual_imports() -> Dict[str, float]:
    """Test individual imports to identify slow components."""
    
    # Clear any existing imports
    modules_to_clear = [mod for mod in sys.modules.keys() if mod.startswith('speedy_utils')]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    import_times = {}
    
    print("Testing individual third-party library imports...")
    
    # Test third-party libraries that speedy_utils imports
    third_party_libs = [
        'numpy',
        'pandas', 
        'xxhash',
        'IPython.core.getipython',
        'IPython.display',
        'loguru',
        'pydantic',
        'tabulate',
        'tqdm'
    ]
    
    for lib in third_party_libs:
        # Clear the library from cache if it exists
        if lib in sys.modules:
            del sys.modules[lib]
        
        import_time = time_import(lib)
        import_times[f"third_party.{lib}"] = import_time
        print(f"{lib}: {import_time:.4f}s")
    
    # Clear speedy_utils modules again
    modules_to_clear = [mod for mod in sys.modules.keys() if mod.startswith('speedy_utils')]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    print("\nTesting speedy_utils submodules...")
    
    # Test speedy_utils submodules
    speedy_modules = [
        'speedy_utils.common.logger',
        'speedy_utils.common.clock',
        'speedy_utils.common.function_decorator',
        'speedy_utils.common.notebook_utils',
        'speedy_utils.common.utils_cache',
        'speedy_utils.common.utils_io',
        'speedy_utils.common.utils_misc',
        'speedy_utils.common.utils_print',
        'speedy_utils.multi_worker.process',
        'speedy_utils.multi_worker.thread',
    ]
    
    for module in speedy_modules:
        import_time = time_import(module)
        import_times[module] = import_time
        print(f"{module}: {import_time:.4f}s")
    
    return import_times


def test_full_import() -> float:
    """Test the full 'from speedy_utils import *' import."""
    
    # Clear any existing speedy_utils imports
    modules_to_clear = [mod for mod in sys.modules.keys() if mod.startswith('speedy_utils')]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    print("\nTesting full 'from speedy_utils import *'...")
    
    start_time = time.time()
    
    try:
        exec("from speedy_utils import *")
        end_time = time.time()
        import_time = end_time - start_time
        print(f"Full import time: {import_time:.4f}s")
        return import_time
    except Exception as e:
        print(f"Error with full import: {e}")
        return -1


def analyze_slow_imports(import_times: Dict[str, float], threshold: float = 0.1) -> None:
    """Analyze and report slow imports."""
    
    print(f"\n=== ANALYSIS: Imports slower than {threshold}s ===")
    
    slow_imports = [(name, time_val) for name, time_val in import_times.items() 
                   if time_val > threshold and time_val > 0]
    
    if not slow_imports:
        print(f"No imports took longer than {threshold}s")
        return
    
    # Sort by time (slowest first)
    slow_imports.sort(key=lambda x: x[1], reverse=True)
    
    total_slow_time = sum(time_val for _, time_val in slow_imports)
    
    print(f"Found {len(slow_imports)} slow imports (total time: {total_slow_time:.4f}s):")
    
    for name, time_val in slow_imports:
        percentage = (time_val / total_slow_time) * 100
        print(f"  {name}: {time_val:.4f}s ({percentage:.1f}%)")


def main():
    """Main test function."""
    
    print("=== SPEEDY_UTILS IMPORT TIME ANALYSIS ===\n")
    
    # Test individual imports
    import_times = test_individual_imports()
    
    # Test full import
    full_import_time = test_full_import()
    if full_import_time > 0:
        import_times['FULL_IMPORT'] = full_import_time
    
    # Analyze results
    analyze_slow_imports(import_times, threshold=0.05)  # 50ms threshold
    
    print("\n=== SUMMARY ===")
    total_individual = sum(t for t in import_times.values() if t > 0 and 'FULL_IMPORT' not in str(t))
    print(f"Total individual import time: {total_individual:.4f}s")
    if full_import_time > 0:
        print(f"Full import time: {full_import_time:.4f}s")
        overhead = full_import_time - total_individual
        print(f"Import overhead: {overhead:.4f}s")


if __name__ == "__main__":
    main()