#!/usr/bin/env python3
"""
Import Time Analysis Script

This script measures the import time of individual modules used by the speedy_utils project.
Each import is executed inside a separate function to avoid global import timing artifacts
that can occur when modules are imported at the module level.

Usage:
    python tests/import_time_report.py

The script will display a table showing import times sorted from slowest to fastest.
"""

import time
from typing import Callable, Dict, List, Tuple

try:
    from tabulate import tabulate
except ImportError:
    print("tabulate not available, falling back to basic formatting")
    tabulate = None


def time_import(import_func: Callable, module_name: str, iterations: int = 1) -> Tuple[str, float, bool]:
    """
    Time an import function by running it multiple times and taking the average.

    Args:
        import_func: Function that performs the import
        module_name: Name of the module for reporting
        iterations: Number of times to run the import

    Returns:
        Tuple of (module_name, average_time_seconds, success)
    """
    times = []

    for _ in range(iterations):
        try:
            start_time = time.perf_counter()
            import_func()
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        except ImportError:
            # Module not available
            return module_name, 0.0, False
        except Exception as e:
            print(f"Error importing {module_name}: {e}")
            return module_name, 0.0, False

    avg_time = sum(times) / len(times)
    return module_name, avg_time, True


# Import functions - each import is isolated in its own function
def import_numpy():
    import numpy

def import_pandas():
    import pandas

def import_torch():
    import torch

def import_ray():
    import ray

def import_matplotlib():
    import matplotlib
    import matplotlib.pyplot

def import_ipython_core():
    from IPython.core.getipython import get_ipython

def import_ipython_display():
    from IPython.display import HTML, display

def import_pil():
    from PIL import Image

def import_pydantic():
    from pydantic import BaseModel

def import_tabulate():
    import tabulate

def import_xxhash():
    import xxhash

def import_tqdm():
    import tqdm

def import_cachetools():
    import cachetools

def import_psutil():
    import psutil

def import_fastcore():
    from fastcore.parallel import parallel

def import_json_repair():
    from json_repair import loads

def import_loguru():
    from loguru import logger

def import_requests():
    import requests

def import_scikit_learn():
    import sklearn

def import_openai():
    import openai

def import_aiohttp():
    import aiohttp

def import_multiprocessing():
    import multiprocessing

def import_threading():
    import threading

def import_concurrent_futures():
    from concurrent.futures import ThreadPoolExecutor

def import_json():
    import json

def import_pickle():
    import pickle

def import_re():
    import re

def import_collections():
    from collections import defaultdict

def import_pathlib():
    from pathlib import Path

def import_functools():
    import functools

def import_itertools():
    import itertools


def main():
    """Main function to run import timing analysis."""

    # Define modules to test with their import functions
    modules_to_test = [
        # Heavy third-party libraries used in speedy_utils
        ("numpy", import_numpy),
        ("pandas", import_pandas),
        ("torch", import_torch),
        ("ray", import_ray),
        ("matplotlib", import_matplotlib),

        # IPython/Jupyter (optional)
        ("IPython.core", import_ipython_core),
        ("IPython.display", import_ipython_display),

        # Image processing (optional)
        ("PIL", import_pil),

        # Data validation
        ("pydantic", import_pydantic),

        # Utilities
        ("tabulate", import_tabulate),
        ("xxhash", import_xxhash),
        ("tqdm", import_tqdm),
        ("cachetools", import_cachetools),
        ("psutil", import_psutil),
        ("fastcore", import_fastcore),
        ("json-repair", import_json_repair),
        ("loguru", import_loguru),
        ("requests", import_requests),

        # Standard library modules that might be slow
        ("multiprocessing", import_multiprocessing),
        ("threading", import_threading),
        ("concurrent.futures", import_concurrent_futures),

        # Common stdlib modules
        ("json", import_json),
        ("pickle", import_pickle),
        ("re", import_re),
        ("pathlib", import_pathlib),
        ("functools", import_functools),
    ]

    print("Analyzing import times for speedy_utils modules...")
    print("=" * 60)

    results = []

    for module_name, import_func in modules_to_test:
        print(f"Testing {module_name}...", end=" ", flush=True)
        name, avg_time, success = time_import(import_func, module_name)

        if success:
            print(".3f")
            results.append((name, avg_time))
        else:
            print("NOT AVAILABLE")
            results.append((name, 0.0))

    # Sort by time (slowest first)
    results.sort(key=lambda x: x[1], reverse=True)

    # Separate available and unavailable modules
    available_results = [(name, time_val) for name, time_val in results if time_val > 0]
    unavailable_modules = [name for name, time_val in results if time_val == 0]

    print("\n" + "=" * 60)
    print("IMPORT TIME ANALYSIS RESULTS")
    print("=" * 60)

    if unavailable_modules:
        print(f"Unavailable modules (not installed): {', '.join(unavailable_modules)}")
        print()

    if tabulate and available_results:
        # Use tabulate for nice table formatting
        table_data = [
            [f"{i+1}", name, ".4f", ".2f"]
            for i, (name, time_val) in enumerate(available_results)
        ]

        headers = ["Rank", "Module", "Time (seconds)", "Time (ms)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    elif available_results:
        # Fallback formatting without tabulate
        print("<4")
        print("-" * 50)
        for i, (name, time_val) in enumerate(available_results, 1):
            print("<4")

    else:
        print("No modules were successfully imported for timing.")

    # Summary statistics
    if available_results:
        total_time = sum(time for _, time in available_results)
        avg_time = total_time / len(available_results)
        max_time = max(time for _, time in available_results)
        min_time = min(time for _, time in available_results)

        print("\nSUMMARY STATISTICS:")
        print(f"Total import time: {total_time:.4f} seconds")
        print(f"Average import time: {avg_time:.4f} seconds")
        print(f"Slowest import: {max_time:.4f} seconds")
        print(f"Fastest import: {min_time:.4f} seconds")
        print(f"Total modules tested: {len(results)}")
        print(f"Available modules: {len(available_results)}")
        print(f"Unavailable modules: {len(results) - len(available_results)}")

        # Show top 5 slowest
        print("\nTOP 5 SLOWEST MODULES:")
        for i, (name, time_val) in enumerate(available_results[:5], 1):
            print(f"{i:2d}. {name:<20} {time_val:.4f}s")


if __name__ == "__main__":
    main()