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
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Generic, List, Literal, Optional, TypeVar, Union

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
from speedy_utils import (
    # Clock module
    Clock,
    speedy_timer,
    timef,
    # Function decorators
    retry_runtime,
    # Cache utilities
    memoize,
    identify,
    identify_uuid,
    # IO utilities
    dump_json_or_pickle,
    dump_jsonl,
    load_by_ext,
    load_json_or_pickle,
    load_jsonl,
    jdumps,
    jloads,
    # Misc utilities
    mkdir_or_exist,
    flatten_list,
    get_arg_names,
    is_notebook,
    convert_to_builtin_python,
    # Print utilities
    display_pretty_table_html,
    flatten_dict,
    fprint,
    print_table,
    setup_logger,
    log,
    # Multi-worker processing
    multi_process,
    multi_thread,
    multi_threaad_standard,
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
    "multi_threaad_standard",
]
