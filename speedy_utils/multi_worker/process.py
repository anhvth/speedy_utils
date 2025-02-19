import gc
import random
import time
from multiprocessing import Manager, Process
from threading import Thread
from typing import List

from fastcore.all import threaded
from loguru import logger
from tqdm import tqdm

from speedy_utils.common.utils_print import setup_logger


def multi_process(
    func: callable,
    inputs: List[any],
    workers=4,
    verbose=True,
    process=False,
    show_traceback=True,
    desc="",
):

    manager = Manager()
    errors = manager.list()
    process = False
    shared_results = manager.dict()
    share_count = manager.Value("i", 0)
    process_lock = manager.Lock()
    results = {}

    @threaded(process=process)
    def f_wrapper_process(i_id, item):
        try:
            result = func(item)
        except Exception as e:
            errors.append(e)
            result = None
            if show_traceback:
                import traceback

                logger.error(traceback.format_exc())
            else:
                logger.error(f"Error with input {item}: {e}")

        with process_lock:
            share_count.value += 1
            if process:
                shared_results[i_id] = result
            else:
                results[i_id] = result

    running_f: List[Thread | Process] = []
    pbar = tqdm(total=len(inputs), disable=not verbose, desc=desc)
    inputs = [(i, inputs[i]) for i in range(len(inputs))]
    total = len(inputs)
    while share_count.value < total:
        logger.debug(f"Share count: {share_count.value}/{total}")
        submited = 0
        while len(running_f) < workers and len(running_f) < len(inputs):
            i_id, item = inputs.pop(0)
            submited += 1
            running_f.append(f_wrapper_process(i_id, item))
        with process_lock:
            to_pop = []
            for i, p in enumerate(running_f):
                if not p.is_alive():
                    pbar.update(1)
                    to_pop.append(i)
            running_f = [running_f[i] for i in range(len(running_f)) if i not in to_pop]
    pbar.close()

    for p in running_f:
        logger.warning(f"Joining {p}")
        p.join()
    if not results:
        results = [shared_results[i] for i in range(len(inputs))]
    gc.collect()
    return [results[i] for i in range(len(results))]


if __name__ == "__main__":

    def f(x):
        time.sleep(0.1)
        return x + 1

    class Aclass:
        def f(self, x, y):
            time.sleep(0.1)
            return x

    obj = Aclass()
    inputs = [(i, i + 1) for i in range(3000)]

    def f2(x):
        return obj.f(x[0], x[1])

    f3 = lambda x: obj.f(x[0], x[1])
    from speedy_utils.common.clock import Clock

    clock = Clock()
    results = multi_process(f3, inputs, workers=100, verbose=True, process=False)
    logger.success(f"Results: {results}\nTime: {clock.time_since_last_checkpoint()}")
    assert results == [i for i in range(3000)]
    proc_per_sec = len(inputs) / clock.time_since_last_checkpoint()
    logger.success(
        f"Results: {results}\nTime: {clock.time_since_last_checkpoint()}\nProcesses per second: {proc_per_sec}"
    )
