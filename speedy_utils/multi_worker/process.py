import gc
import random
import time
from multiprocessing import Manager, Process
from threading import Thread
from typing import List
import joblib
from functools import partial

from fastcore.all import threaded
from loguru import logger
from tqdm import tqdm

from speedy_utils.common.utils_print import setup_logger


def multi_process(
    func: callable,
    inputs: List[any],
    workers: int = 4,
    verbose: bool = True,
    show_traceback: bool = True,
    desc: str = "",
) -> List[any]:
    """
    Process inputs in parallel using multiple processes.

    Args:
        func: Function to apply to each input
        inputs: List of inputs to process
        workers: Number of worker processes
        verbose: Whether to show progress bar
        show_traceback: Whether to show full traceback on errors
        desc: Description for the progress bar

    Returns:
        List of results in the same order as inputs
    """
    from multiprocessing import Process, Queue
    from queue import Empty

    def worker(input_queue, output_queue, func):
        while True:
            try:
                idx, item = input_queue.get_nowait()
                try:
                    result = func(item)
                    output_queue.put((idx, result))
                except Exception as e:
                    if show_traceback:
                        import traceback

                        logger.error(traceback.format_exc())
                    else:
                        logger.error(f"Error with input {item}: {e}")
                    output_queue.put((idx, None))
            except Empty:
                break

    # Initialize queues
    input_queue = Queue()
    output_queue = Queue()

    # Fill input queue
    for i, item in enumerate(inputs):
        input_queue.put((i, item))

    # Start workers
    processes = []
    for _ in range(min(workers, len(inputs))):
        p = Process(target=worker, args=(input_queue, output_queue, func))
        p.start()
        processes.append(p)

    # Collect results
    results = {}
    total_inputs = len(inputs)

    with tqdm(total=total_inputs, disable=not verbose, desc=desc) as pbar:
        while len(results) < total_inputs:
            idx, result = output_queue.get()
            results[idx] = result
            pbar.update(1)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Clean up
    input_queue.close()
    output_queue.close()
    gc.collect()

    # Return results in original order
    return [results[i] for i in range(total_inputs)]
