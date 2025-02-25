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
    workers=4,
    verbose=True,
    show_traceback=True,
    desc="",
):
    # Function wrapper to handle errors
    def func_wrapper(idx, item):
        try:
            return idx, func(item)
        except Exception as e:
            if show_traceback:
                import traceback
                logger.error(traceback.format_exc())
            else:
                logger.error(f"Error with input {item}: {e}")
            return idx, None
    
    # Process inputs with joblib in batches to update tqdm
    results = {}
    total_inputs = len(inputs)
    
    # Determine a reasonable batch size for progress updates
    batch_size = max(1, min(100, total_inputs // max(1, workers * 2)))
    
    with tqdm(total=total_inputs, disable=not verbose, desc=desc) as pbar:
        for i in range(0, total_inputs, batch_size):
            batch_end = min(i + batch_size, total_inputs)
            batch_inputs = [(j, inputs[j]) for j in range(i, batch_end)]
            
            # Process batch in parallel using joblib
            batch_results = joblib.Parallel(n_jobs=workers, verbose=0)(
                joblib.delayed(func_wrapper)(idx, item) for idx, item in batch_inputs
            )
            
            # Store results and update progress
            for idx, result in batch_results:
                results[idx] = result
            pbar.update(len(batch_inputs))
    
    gc.collect()
    return [results[i] for i in range(total_inputs)]

