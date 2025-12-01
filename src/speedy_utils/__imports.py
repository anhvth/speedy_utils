# type: ignore
from __future__ import annotations

import time
import warnings


# Third-party imports
try:
    # Python 3.10+
    from typing import ParamSpec
except ImportError:  # pragma: no cover
    from typing_extensions import ParamSpec  # type: ignore


import asyncio
import contextlib
import copy
import ctypes
import datetime
import functools
import gc
import inspect
import io
import json
import multiprocessing
import os
import os.path as osp
import pathlib
import pickle
import pprint
import random
import re
import sys
import textwrap
import threading
import time
import traceback
import types
import uuid
import weakref
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from collections.abc import Callable as TypingCallable
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from datetime import datetime  # noqa: F811
from glob import glob
from heapq import heappop, heappush
from itertools import islice
from multiprocessing import Pool
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import (
    IO,
    TYPE_CHECKING,
    Annotated,
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
    cast,
    overload,
)

import cachetools
import psutil
from fastcore.parallel import parallel
from json_repair import loads as jloads
from loguru import logger
from tqdm import tqdm


# Direct imports (previously lazy-loaded)
import numpy as np
tabulate = __import__('tabulate').tabulate
import xxhash

# Optional imports - lazy loaded for performance
def _get_pandas():
    """Lazy import pandas."""
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None

def _get_ray():
    """Lazy import ray."""
    try:
        import ray
        return ray
    except ImportError:
        return None

def _get_matplotlib():
    """Lazy import matplotlib."""
    try:
        import matplotlib
        return matplotlib
    except ImportError:
        return None

def _get_matplotlib_pyplot():
    """Lazy import matplotlib.pyplot."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None

def _get_ipython_core():
    """Lazy import IPython.core.getipython."""
    try:
        from IPython.core.getipython import get_ipython
        return get_ipython
    except ImportError:
        return None

# Cache for lazy imports
_pandas_cache = None
_ray_cache = None
_matplotlib_cache = None
_plt_cache = None
_get_ipython_cache = None

# Lazy import classes for performance-critical modules
class _LazyModule:
    """Lazy module loader that imports only when accessed."""
    def __init__(self, import_func, cache_var_name):
        self._import_func = import_func
        self._cache_var_name = cache_var_name
        self._module = None

    def __call__(self):
        """Allow calling as a function to get the module."""
        if self._module is None:
            # Use global cache
            cache = globals().get(self._cache_var_name)
            if cache is None:
                cache = self._import_func()
                globals()[self._cache_var_name] = cache
            self._module = cache
        return self._module

    def __getattr__(self, name):
        """Lazy attribute access."""
        if self._module is None:
            self()  # Load the module
        return getattr(self._module, name)

    def __bool__(self):
        """Support truthiness checks."""
        return self._module is not None

    def __repr__(self):
        if self._module is None:
            return f"<LazyModule: not loaded>"
        return repr(self._module)

# Create lazy loaders for top slow imports (import only when accessed)
pd = _LazyModule(_get_pandas, '_pandas_cache')
ray = _LazyModule(_get_ray, '_ray_cache')
matplotlib = _LazyModule(_get_matplotlib, '_matplotlib_cache')
plt = _LazyModule(_get_matplotlib_pyplot, '_plt_cache')
get_ipython = _LazyModule(_get_ipython_core, '_get_ipython_cache')

# Other optional imports (not lazy loaded as they're not in top slow imports)
try:
    import torch
except ImportError:
    torch = None

try:
    from IPython.display import HTML, display
except ImportError:
    HTML = None
    display = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import ray
    import torch
    import matplotlib.pyplot as plt
    # xxhash
    import xxhash  # type: ignore
    from IPython.core.getipython import get_ipython  # type: ignore
    from IPython.display import HTML, display  # type: ignore
    from loguru import logger  # type: ignore
    from PIL import Image
    from pydantic import BaseModel  # type: ignore
    from tabulate import tabulate  # type: ignore
    from tqdm import tqdm  # type: ignore

__all__ = [
    # ------------------------------------------------------------------
    # Direct imports (previously lazy-loaded)
    # ------------------------------------------------------------------
    'torch',
    'np',
    'pd',
    'tqdm',
    'tabulate',
    'xxhash',
    'get_ipython',
    'HTML',
    'display',
    'BaseModel',
    'Image',
    'ray',
    'matplotlib',
    'plt',
    # ------------------------------------------------------------------
    # Standard library modules imported
    # ------------------------------------------------------------------
    'asyncio',
    'contextlib',
    'copy',
    'ctypes',
    'datetime',
    'functools',
    'gc',
    'inspect',
    'io',
    'json',
    'multiprocessing',
    'os',
    'osp',
    'pathlib',
    'pickle',
    'pprint',
    'random',
    're',
    'sys',
    'textwrap',
    'threading',
    'time',
    'traceback',
    'types',
    'uuid',
    'weakref',
    'warnings',
    # ------------------------------------------------------------------
    # Data structures
    # ------------------------------------------------------------------
    'Counter',
    'OrderedDict',
    'defaultdict',
    'MappingProxyType',
    # ------------------------------------------------------------------
    # File & path utilities
    # ------------------------------------------------------------------
    'Path',
    'glob',
    # ------------------------------------------------------------------
    # Concurrency / parallelism
    # ------------------------------------------------------------------
    'ThreadPoolExecutor',
    'as_completed',
    'wait',
    'FIRST_COMPLETED',
    'Future',
    'Pool',
    'Lock',
    # ------------------------------------------------------------------
    # Algorithms / heap helpers
    # ------------------------------------------------------------------
    'heappop',
    'heappush',
    'islice',
    # ------------------------------------------------------------------
    # Typing
    # ------------------------------------------------------------------
    'Annotated',
    'Any',
    'Awaitable',
    'Callable',
    'Dict',
    'Generic',
    'IO',
    'Iterable',
    'List',
    'Literal',
    'Mapping',
    'Optional',
    'ParamSpec',
    'Sequence',
    'Set',
    'Tuple',
    'Type',
    'TYPE_CHECKING',
    'TypeVar',
    'TypingCallable',
    'Union',
    'cast',
    'overload',
    # ------------------------------------------------------------------
    # Third-party modules
    # ------------------------------------------------------------------
    'cachetools',
    'psutil',
    'parallel',
    'jloads',
    'logger',
    'plt',
]
