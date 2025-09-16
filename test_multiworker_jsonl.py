#!/usr/bin/env python3
"""Test script for the multi-worker JSONL functionality."""

import json
import tempfile
import os
from pathlib import Path
import sys

# Add the project to path
sys.path.insert(0, "/home/anhvth5/projects/speedy_utils/src")

from speedy_utils.common.utils_io import fast_load_jsonl


def create_test_jsonl(num_lines: int) -> str:
    """Create a test JSONL file with specified number of lines."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for i in range(num_lines):
            data = {"id": i, "value": f"test_value_{i}", "number": i * 2}
            f.write(json.dumps(data) + '\n')
        return f.name


def test_small_file():
    """Test with a small file (should use single-threaded)."""
    print("Testing small file (100 lines)...")
    file_path = create_test_jsonl(100)
    
    try:
        results = list(fast_load_jsonl(file_path, progress=True))
        print(f"Loaded {len(results)} objects")
        assert len(results) == 100
        assert results[0]['id'] == 0
        assert results[99]['id'] == 99
        print("✓ Small file test passed")
    finally:
        os.unlink(file_path)


def test_large_file():
    """Test with a large file (should use multi-worker)."""
    print("\nTesting large file (60,000 lines)...")
    file_path = create_test_jsonl(60000)
    
    try:
        results = list(fast_load_jsonl(file_path, progress=True, multiworker_threshold=50000))
        print(f"Loaded {len(results)} objects")
        assert len(results) == 60000
        assert results[0]['id'] == 0
        assert results[59999]['id'] == 59999
        print("✓ Large file test passed")
    finally:
        os.unlink(file_path)


def test_disable_multiworker():
    """Test with multiworker disabled."""
    print("\nTesting large file with multiworker disabled...")
    file_path = create_test_jsonl(60000)
    
    try:
        results = list(fast_load_jsonl(file_path, progress=True, use_multiworker=False))
        print(f"Loaded {len(results)} objects")
        assert len(results) == 60000
        print("✓ Multiworker disabled test passed")
    finally:
        os.unlink(file_path)


if __name__ == "__main__":
    test_small_file()
    test_large_file() 
    test_disable_multiworker()
    print("\n🎉 All tests passed!")