from speedy_utils.all import is_notebook
from fastcore.all import threaded
import psutil
import gc
import os
from tqdm import tqdm
from loguru import logger
from multiprocessing import Manager
from threading import Semaphore


# def get_kernel_memory_usage():
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     msg = f"Memory usage: {mem_info.rss / (1024 ** 2):.2f} MB"
#     # logger.debug(msg)


# print(get_kernel_memory_usage())



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


def multi_thread(func, orig_inputs, workers=10, desc=None, verbose=False, **kwargs):
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


# Example usage
if __name__ == "__main__":
    import time
    def f(x):
        logger.info(f"{x=}")
        if x == 5:
            1 / 0
        time.sleep(1)  # Simulate work
        return x

    inputs = list(range(3))  # Example inputs
    results = multi_thread(f, inputs, workers=3, desc="Running tasks", verbose=True)
    print(results)
