# ----------------------------------------------------------------------------
# speedy_utils/__init__.py
#
# Main entry point and public API for the Speedy Utils library
#
# This module exports the primary utilities for enhanced Python development
# productivity including caching mechanisms, parallel processing, file I/O,
# timing utilities, and data manipulation functions. It provides a convenient
# single-import interface for the most commonly used functionality.
#
# Public API / Data Contracts:
# • setup_logger(min_interval: int = 5) -> None - Configure logging system
# • log(*args, **kwargs) -> None - Rate-limited logging function
# • Clock() - Timing and performance measurement utility
# • speedy_timer: Clock - Pre-configured global timer instance
# • timef(func) -> Callable - Function execution time decorator
# • retry_runtime(sleep_seconds: int, max_retry: int, exceptions) -> Callable
# • memoize(func) -> Callable - Function result caching decorator
# • identify(obj: Any) -> str - Generate unique object identifier
# • identify_uuid(obj: Any) -> str - Generate UUID-based object identifier
# • load_by_ext(fname: str | list[str]) -> Any - Auto-detect file format loader
# • dump_json_or_pickle(obj: Any, fname: str) -> None - Smart file serializer
# • load_json_or_pickle(fname: str) -> Any - Smart file deserializer
# • multi_thread(func, items, **kwargs) -> list - Parallel thread execution
# • multi_process(func, items, **kwargs) -> list - Parallel process execution
#
# Invariants / Constraints:
# • MUST import only stable, tested utilities into public namespace
# • SHOULD maintain backward compatibility across minor versions
# • MUST provide consistent error handling across all public functions
# • SHOULD use lazy imports for heavy dependencies when possible
#
# Usage Example:
# ```python
# from speedy_utils import Clock, memoize, multi_thread, load_by_ext
#
# @memoize
# def expensive_computation(x):
#     return x ** 2
#
# timer = Clock()
# timer.start()
# results = multi_thread(expensive_computation, range(100))
# timer.end()
# data = load_by_ext("config.json")
# ```
#
# TODO & Future Work:
# • Add async variants for I/O operations
# • Implement distributed caching backend
# • Add GPU acceleration utilities
# ----------------------------------------------------------------------------

# Import specific functions and classes from modules
# Logger
# Standard library imports
import copy
import functools
import gc
import inspect
import json
import multiprocessing
import os
import os.path as osp
import pickle
import pprint
import random
import re
import sys
import textwrap
import threading
import time
import traceback
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

# Third-party imports
import numpy as np
import pandas as pd
import xxhash
from IPython.core.getipython import get_ipython
from IPython.display import HTML, display
from loguru import logger
from pydantic import BaseModel
from tabulate import tabulate
from tqdm import tqdm

from speedy_utils.common.logger import log, setup_logger

# Clock module
from .common.clock import Clock, speedy_timer, timef

# Function decorators
from .common.function_decorator import retry_runtime

# notebook
from .common.notebook_utils import (
    change_dir,
    display_pretty_table_html,
    print_table,
)

# Cache utilities
from .common.utils_cache import identify, identify_uuid, memoize

# IO utilities
from .common.utils_io import (
    dump_json_or_pickle,
    dump_jsonl,
    jdumps,
    jloads,
    load_by_ext,
    load_json_or_pickle,
    load_jsonl,
)

# Misc utilities
from .common.utils_misc import (
    convert_to_builtin_python,
    flatten_list,
    get_arg_names,
    is_notebook,
    mkdir_or_exist,
)

# Print utilities
from .common.utils_print import (
    flatten_dict,
    fprint,
)

# Multi-worker processing
from .multi_worker.process import multi_process
from .multi_worker.thread import multi_thread

# Define __all__ explicitly
__all__ = [
    # Standard library
    "random",
    "copy",
    "functools",
    "gc",
    "inspect",
    "json",
    "multiprocessing",
    "os",
    "osp",
    "pickle",
    "pprint",
    "re",
    "sys",
    "textwrap",
    "threading",
    "time",
    "traceback",
    "uuid",
    "Counter",
    "ThreadPoolExecutor",
    "as_completed",
    "glob",
    "Pool",
    "Path",
    "Lock",
    "defaultdict",
    # Typing
    "Any",
    "Callable",
    "Dict",
    "Generic",
    "List",
    "Literal",
    "Optional",
    "TypeVar",
    "Union",
    # Third-party
    "pd",
    "xxhash",
    "get_ipython",
    "HTML",
    "display",
    "logger",
    "BaseModel",
    "tabulate",
    "tqdm",
    "np",
    # Clock module
    "Clock",
    "speedy_timer",
    "timef",
    # Function decorators
    "retry_runtime",
    # Cache utilities
    "memoize",
    "identify",
    "identify_uuid",
    # IO utilities
    "dump_json_or_pickle",
    "dump_jsonl",
    "load_by_ext",
    "load_json_or_pickle",
    "load_jsonl",
    "jdumps",
    "jloads",
    # Misc utilities
    "mkdir_or_exist",
    "flatten_list",
    "get_arg_names",
    "is_notebook",
    "convert_to_builtin_python",
    # Print utilities
    "display_pretty_table_html",
    "flatten_dict",
    "fprint",
    "print_table",
    "setup_logger",
    "log",
    # Multi-worker processing
    "multi_process",
    "multi_thread",
    # Notebook utilities
    "change_dir",
]
