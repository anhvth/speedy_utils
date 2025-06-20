# ----------------------------------------------------------------------------
# speedy_utils/all.py
#
# Consolidated import collection for comprehensive library access
#
# This module provides a unified collection of standard library, third-party,
# and internal imports commonly used across data science and development
# workflows. It serves as a convenience module for interactive environments
# and rapid prototyping by reducing boilerplate import statements.
#
# Public API / Data Contracts:
# • All standard library modules: collections, concurrent.futures, pathlib, etc.
# • Third-party dependencies: loguru.logger, pydantic.BaseModel, tqdm, tabulate
# • Core utilities: Counter, defaultdict, ThreadPoolExecutor, as_completed
# • Development tools: IPython.display.HTML, get_ipython for notebook detection
# • Type system: Any, Dict, List, Optional, Union, TypeVar, Generic, Literal
#
# Invariants / Constraints:
# • MUST only import stable, widely-used packages
# • SHOULD handle import failures gracefully for optional dependencies
# • MUST maintain consistent import aliases across the library
# • SHOULD group imports by category (stdlib, third-party, internal)
#
# Usage Example:
# ```python
# from speedy_utils.all import *
#
# # Now have access to common utilities without individual imports
# data = defaultdict(list)
# results = []
# for item in tqdm(items):
#     results.append(process(item))
#
# df = tabulate(results, headers=['Item', 'Result'])
# display(HTML(df))
# ```
#
# TODO & Future Work:
# • Add conditional imports for ML libraries (torch, numpy, pandas)
# • Implement import health checking
# • Add version compatibility warnings
# ----------------------------------------------------------------------------

# speedy_utils/all.py

# Provide a consolidated set of imports for convenience

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

# Import specific functions from speedy_utils
from speedy_utils import (  # Clock module; Function decorators; Cache utilities; IO utilities; Misc utilities; Print utilities; Multi-worker processing
    Clock,
    convert_to_builtin_python,
    display_pretty_table_html,
    dump_json_or_pickle,
    dump_jsonl,
    flatten_dict,
    flatten_list,
    fprint,
    get_arg_names,
    identify,
    identify_uuid,
    is_notebook,
    jdumps,
    jloads,
    load_by_ext,
    load_json_or_pickle,
    load_jsonl,
    log,
    memoize,
    mkdir_or_exist,
    multi_process,
    multi_thread,
    print_table,
    retry_runtime,
    setup_logger,
    speedy_timer,
    timef,
)

# Define __all__ explicitly with all exports
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
]
