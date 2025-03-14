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


from fastcore.parallel import parallel
from typing import List, Callable, Any
import math

def multi_process(
    func: Callable[[Any], Any], 
    items: List[Any], 
    workers: int = None, 
    progress: bool = True, 
    chunksize: int = None,  # Optional chunksize
    **kwargs,
) -> List[Any]:
    """
    A parallel processing function using fastcore.parallel.parallel.

    Args:
        func (Callable): The function to apply in parallel.
        items (List[Any]): List of items to process.
        workers (int, optional): Number of worker processes (default: CPU count).
        progress (bool, optional): Show progress bar (default: True).
        chunksize (int, optional): Size of task chunks (auto-calculated if None).
        **kwargs: Additional keyword arguments for parallel.

    Returns:
        List[Any]: Processed results.
    """
    if not items:
        return []  # Handle empty input list

    # Set workers dynamically if not provided
    if workers is None:
        from fastcore.foundation import defaults
        workers = defaults.cpus  # Default to available CPUs

    # Ensure we don't assign more workers than tasks
    workers = min(len(items), workers)

    # Compute optimal chunksize dynamically if not provided
    if chunksize is None:
        chunksize = max(1, math.ceil(len(items) / (workers * 4)))

    results = parallel(
        func,
        items,
        n_workers=workers,
        threadpool=False,
        progress=progress,
        # method="fork",  # "fork" is faster but might not work on Windows
        chunksize=chunksize,  # Auto-computed chunksize
        **kwargs,
    )

    return results
