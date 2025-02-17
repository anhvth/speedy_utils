from speedy_utils.common.utils_misc import is_notebook
from fastcore.all import threaded
import psutil
import gc
import os
from tqdm import tqdm
from loguru import logger
from multiprocessing import Manager
from threading import Semaphore


def _multi_thread_in_sub_process(
    func,
    orig_inputs,
    shared_result_list,  # Already a parameter
    workers=4,
    desc=None,
    verbose=False,
):
    desc = desc or "Processing"

    semaphore = Semaphore(workers)  # Limit concurrent threads

    @threaded(process=False)
    def _f(i):
        with semaphore:  # Ensure only 'workers' threads run at a time
            try:
                return func(i)
            except Exception as e:
                return e

    threads = [_f(i) for i in orig_inputs]  # Start threads
    idx = 0
    with tqdm(total=len(orig_inputs), desc=desc, ncols=88, leave=verbose) as pbar:
        for t in threads:
            t.join()  # Wait for thread completion
            shared_result_list[idx] = getattr(t, "result")
            idx += 1
            pbar.update(1)

    return shared_result_list


def multi_thread(
    func, orig_inputs, workers=10, desc=None, verbose=False, do_memoize=False, **kwargs
):
    if do_memoize:
        from speedy_utils import memoize

        func = memoize(func)
    if is_notebook():
        manager = Manager()
        data = [None] * len(orig_inputs)
        shared_result_list = manager.list(data)
        _func = threaded(process=True)(_multi_thread_in_sub_process)
        process = _func(
            func,
            orig_inputs,
            shared_result_list,  # Pass as positional argument
            workers=workers,
            desc=desc,
            verbose=verbose,
        )
        try:
            process.join()
        except KeyboardInterrupt:
            logger.error("Interrupted by user")
            process.terminate()
        finally:
            ntrash = gc.collect()
            logger.info(f"Garbage collected: {ntrash}")

        results = list(shared_result_list)
    else:
        return _multi_thread_in_sub_process(
            func,
            orig_inputs,
            [None] * len(orig_inputs),
            workers=workers,
            desc=desc,
            verbose=verbose,
        )
    return results
