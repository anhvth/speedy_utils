#!/usr/bin/env python3
"""
Detailed analysis of logger import slowness.
"""

import time
import sys
import importlib


def time_step(step_name: str, func):
    """Time a specific step and print the result."""
    start = time.time()
    result = func()
    end = time.time()
    print(f"{step_name}: {end - start:.4f}s")
    return result


def test_logger_import_steps():
    """Break down logger import into steps to find the bottleneck."""
    
    # Clear any existing imports
    modules_to_clear = [mod for mod in sys.modules.keys() if 'speedy_utils' in mod or 'loguru' in mod]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    print("=== DETAILED LOGGER IMPORT ANALYSIS ===\n")
    
    # Step 1: Import loguru alone
    print("Step 1: Import loguru")
    time_step("loguru import", lambda: importlib.import_module('loguru'))
    
    # Step 2: Import the logger module 
    print("\nStep 2: Import speedy_utils.common.logger")
    time_step("logger module import", lambda: importlib.import_module('speedy_utils.common.logger'))
    
    # Step 3: Test calling setup_logger
    print("\nStep 3: Call setup_logger function")
    from speedy_utils.common.logger import setup_logger
    time_step("setup_logger call", lambda: setup_logger())
    
    # Step 4: Test importing from speedy_utils main
    print("\nStep 4: Import from main speedy_utils")
    # Clear again
    modules_to_clear = [mod for mod in sys.modules.keys() if 'speedy_utils' in mod]
    for mod in modules_to_clear:
        del sys.modules[mod]
    
    time_step("speedy_utils main import", lambda: importlib.import_module('speedy_utils'))


def test_logger_initialization():
    """Test if logger setup is happening during import."""
    
    print("\n=== TESTING LOGGER INITIALIZATION ===\n")
    
    # Clear loguru completely
    if 'loguru' in sys.modules:
        del sys.modules['loguru']
    
    # Import loguru and check its state
    print("Testing loguru initialization...")
    
    start = time.time()
    from loguru import logger
    end = time.time()
    print(f"Loguru import: {end - start:.4f}s")
    
    # Check if logger has handlers
    print(f"Logger has {len(logger._core.handlers)} handlers after import")
    
    # Test adding a handler
    start = time.time()
    logger.add(sys.stdout, format="{message}")
    end = time.time()
    print(f"Adding handler: {end - start:.4f}s")
    
    # Test removing handlers
    start = time.time()
    logger.remove()
    end = time.time()
    print(f"Removing handlers: {end - start:.4f}s")


def main():
    test_logger_import_steps()
    test_logger_initialization()


if __name__ == "__main__":
    main()