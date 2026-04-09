# Standard library
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
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from collections.abc import Callable as TypingCallable
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from threading import Lock
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

# Third-party (fast)
import xxhash
from loguru import logger
from tqdm import tqdm

tabulate = __import__("tabulate").tabulate

# Heavy third-party modules — imported lazily on first access to keep
# `import speedy_utils` under 0.4 s.  Access them like any other attribute;
# Python calls __getattr__ below the first time each name is used.
_HEAVY = {
    "np": ("numpy", None),
    "pd": ("pandas", None),
    "matplotlib": ("matplotlib", None),
    "plt": ("matplotlib.pyplot", None),
    "get_ipython": ("IPython.core.getipython", "get_ipython"),
    "HTML": ("IPython.display", "HTML"),
    "display": ("IPython.display", "display"),
    "BaseModel": ("pydantic", "BaseModel"),
}


def __getattr__(name: str):
    if name in _HEAVY:
        module_path, attr = _HEAVY[name]
        try:
            import importlib
            mod = importlib.import_module(module_path)
            value = getattr(mod, attr) if attr else mod
        except ImportError:
            value = None
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Clock
from speedy_utils.common.clock import Clock, speedy_timer, timef

# Function decorators / logging
from speedy_utils.common.function_decorator import retry_runtime
from speedy_utils.common.logger import log, setup_logger

# Notebook helpers
from speedy_utils.common.notebook_utils import (
    change_dir,
    display_pretty_table_html,
    print_table,
)

# Cache utilities
from speedy_utils.common.utils_cache import identify, identify_uuid, imemoize, memoize

# IO utilities
from speedy_utils.common.utils_io import (
    dump_json_or_pickle,
    dump_jsonl,
    jdumps,
    jloads,
    load_by_ext,
    load_json_or_pickle,
    load_jsonl,
)

# Misc utilities
from speedy_utils.common.utils_misc import (
    convert_to_builtin_python,
    dedup,
    flatten_list,
    get_arg_names,
    is_notebook,
    mkdir_or_exist,
)

# Print utilities
from speedy_utils.common.utils_print import flatten_dict, fprint

# Error handling utilities
from speedy_utils.common.utils_error import (
    clean_traceback,
    handle_exceptions_with_clean_traceback,
)

# Multi-worker processing
from speedy_utils.multi_worker.process import multi_process
from speedy_utils.multi_worker.thread import kill_all_thread, multi_thread
from speedy_utils.multi_worker.dataset_sharding import multi_process_dataset

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
    "Awaitable",
    "Callable",
    "TypingCallable",
    "Dict",
    "Generic",
    "Iterable",
    "List",
    "Literal",
    "Mapping",
    "Optional",
    "Sequence",
    "Set",
    "Tuple",
    "Type",
    "TypeVar",
    "Union",
    # Third-party
    "xxhash",
    "logger",
    "tabulate",
    "tqdm",
    # Heavy third-party — omitted from __all__ so `from speedy_utils import *`
    # does NOT eagerly load them.  Access via `speedy_utils.pd` or
    # `from speedy_utils import pd` (triggers __getattr__ lazily).
    # "np", "pd", "matplotlib", "plt", "get_ipython", "HTML", "display", "BaseModel"
    # Clock
    "Clock",
    "speedy_timer",
    "timef",
    # Function decorators
    "retry_runtime",
    # Cache utilities
    "memoize",
    "imemoize",
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
    "dedup",
    # Print utilities
    "display_pretty_table_html",
    "flatten_dict",
    "fprint",
    "print_table",
    "setup_logger",
    "log",
    # Error handling utilities
    "clean_traceback",
    "handle_exceptions_with_clean_traceback",
    # Multi-worker processing
    "multi_process",
    "multi_thread",
    "kill_all_thread",
    "multi_process_dataset",
    # Notebook utilities
    "change_dir",
]
