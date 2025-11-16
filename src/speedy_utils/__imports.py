# type: ignore
from __future__ import annotations

import time
import warnings


# Suppress lazy_loader subpackage warning
warnings.filterwarnings(
    'ignore',
    message='subpackages can technically be lazily loaded',
    category=RuntimeWarning,
    module='lazy_loader',
)

t = time.time()
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
from datetime import datetime
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
import lazy_loader as lazy
import psutil
from fastcore.parallel import parallel
from json_repair import loads as jloads
from loguru import logger
from tqdm import tqdm


# Resolve long-import-time dependencies lazily

torch = lazy.load('torch')  # lazy at runtime
np = lazy.load('numpy')
pd = lazy.load('pandas')
tqdm = lazy.load('tqdm').tqdm  # type: ignore  # noqa: F811
pd = lazy.load('pandas')
tabulate = lazy.load('tabulate').tabulate
xxhash = lazy.load('xxhash')
get_ipython = lazy.load('IPython.core.getipython')
HTML = lazy.load('IPython.display').HTML
display = lazy.load('IPython.display').display
# logger = lazy.load('loguru').logger
BaseModel = lazy.load('pydantic').BaseModel
_pil = lazy.load('PIL.Image')
Image = _pil.Image

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import torch

    # xxhash
    import xxhash  # type: ignore
    from IPython.core.getipython import get_ipython  # type: ignore
    from IPython.display import HTML, display  # type: ignore
    from loguru import logger  # type: ignore
    from PIL import Image
    from pydantic import BaseModel  # type: ignore
    from tabulate import tabulate  # type: ignore
    from tqdm import tqdm  # type: ignore


# Import specific functions from speedy_utils
load_time = time.time() - t
