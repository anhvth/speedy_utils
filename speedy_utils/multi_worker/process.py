import gc
import inspect
import random
import time
import traceback
import dill
from multiprocessing import Manager, Process
from threading import Thread
from typing import List, Literal

from fastcore.all import threaded, parallel, defaults
from loguru import logger
from tqdm import tqdm

from speedy_utils.common.clock import Clock
from speedy_utils.common.report_manager import ReportManager


def multi_process(
    func: callable,
    items: List[any],
    workers=defaults.cpus,
    progress=True,
    **kwargs,
):
    results = parallel(
        func,
        items,
        n_workers=workers,
        threadpool=False,
        progress=progress,
        method="fork",
        **kwargs,
    )

    return results
