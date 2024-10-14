import asyncio
import inspect
import os
import signal
import threading
from concurrent.futures import (ThreadPoolExecutor, as_completed)
from multiprocessing import Pool
from typing import Any, Callable, List

from loguru import logger
from tqdm import tqdm

def multi_thread(
    func: Callable,
    inputs: List[Any],
    workers: int = 4,
    verbose: bool = True,
    desc: str | None = None,
) -> List[Any]:
    if desc is None:
        fn_name = func.__name__
        try:
            source_file = inspect.getsourcefile(func) or "<string>"
            source_line = inspect.getsourcelines(func)[1]
            file_line = f"{source_file}:{source_line}"
        except (TypeError, OSError):
            file_line = "Unknown location"
        desc = f"{fn_name} at {file_line}"

    stop_event = threading.Event()
    results = [None] * len(inputs)  # Placeholder for results in order of inputs

    def wrapped_func(idx, *args, **kwargs):
        if stop_event.is_set():
            return None
        return idx, func(*args, **kwargs)  # Return both index and result

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(wrapped_func, i, inp) for i, inp in enumerate(inputs)]

            if verbose:
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    if stop_event.is_set():
                        break
                    try:
                        idx, result = future.result()
                        results[idx] = result  # Store result at the correct index
                    except Exception as e:
                        print(f"Error occurred in one of the threads: {e}")
            else:
                for future in as_completed(futures):
                    if stop_event.is_set():
                        break
                    try:
                        idx, result = future.result()
                        results[idx] = result  # Store result at the correct index
                    except Exception as e:
                        print(f"Error occurred in one of the threads: {e}")

    except KeyboardInterrupt:
        print("\nExecution manually interrupted. Cleaning up...")
        stop_event.set()

        for future in futures:
            if not future.done():
                future.cancel()
    except Exception as e:
        print(f"An error occurred during execution: {e}")
    finally:
        print("Cleaning up any remaining threads or resources...")
        executor.shutdown(wait=False)

    return results



# Set up logging


# Global variable for function initialization in pool workers
def _init_pool_processes(func):
    global _func
    _func = func

# Wrapper to unpack arguments and execute the function
def _pool_process_executor(args):
    if isinstance(args, tuple):
        return _func(*args)
    else:
        return _func(args)

def multi_process(
    func: Callable,
    inputs: List[Any],
    workers: int = 16,
    verbose: bool = True,
    desc: str = "",
) -> List[Any]:
    if not desc:
        fn_name = func.__name__
        try:
            source_file = inspect.getsourcefile(func) or "<string>"
            source_line = inspect.getsourcelines(func)[1]
            file_line = f"{source_file}:{source_line}"
        except (TypeError, OSError):
            file_line = "Unknown location"
        desc = f"Multi-process running: {fn_name} at {file_line}"

    # Debugging flag check to reduce workers to 1
    if os.environ.get("DEBUG", "0") == "1":
        logger.opt(depth=2).info("DEBUGGING set num workers to 1")
        workers = 1

    logger.info(f"Multi-processing {desc} | Num samples: {len(inputs)}")

    # Results list to store processed outputs
    results = []

    # Define a custom signal handler to catch interruptions (KeyboardInterrupt)
    def signal_handler(signum, frame):
        print("\nExecution interrupted! Terminating the pool...")
        pool.terminate()  # Gracefully terminate the pool
        pool.join()       # Ensure all workers are cleaned up
        print("Pool terminated.")
        raise KeyboardInterrupt  # Raise the interrupt for higher-level handling

    # Register the custom signal handler
    signal.signal(signal.SIGINT, signal_handler)

    with Pool(processes=workers, initializer=_init_pool_processes, initargs=(func,)) as pool:
        try:
            if verbose:
                # Use tqdm to show progress in verbose mode
                for result in tqdm(pool.imap(_pool_process_executor, inputs), total=len(inputs), desc=desc):
                    results.append(result)
            else:
                # Execute without tqdm in non-verbose mode
                results = pool.map(_pool_process_executor, inputs)
        except KeyboardInterrupt:
            # Handle manual stop gracefully
            logger.warning("Execution manually interrupted. Terminating workers.")
            pool.terminate()  # Terminate any remaining workers
            pool.join()       # Ensure pool shutdown completes
        except Exception as e:
            # Catch any other exceptions and log the error
            logger.error(f"[multiprocess] Error: {e}")
            pool.terminate()  # Ensure the pool is terminated in case of error
            pool.join()       # Ensure resources are cleaned up
        finally:
            # Ensure the pool is always closed and joined to prevent resource leakage
            pool.close()  # Prevents new tasks from being submitted
            pool.join()   # Wait for all processes to finish or be terminated
            logger.info("Pool closed and cleaned up.")

    return results


import asyncio

from tqdm import tqdm


async def async_multi_thread(f, inputs, desc="", user_tqdm=True, max_workers=10):
    """
    Usage:
        inputs = list(range(10))
        def function(i):
            time.sleep(1)
            return 1/i
        results = await async_multi_thread(function, inputs)
    
    Params:
        f: The function to execute asynchronously.
        inputs: The list of inputs to the function.
        desc: Description for the progress bar.
        user_tqdm: boolean to enable/disable tqdm progress bar.
        max_workers: Maximum number of concurrent tasks.
    """

    def ensure_output_idx(idx_i):
        idx, i = idx_i
        return idx, f(i)

    semaphore = asyncio.Semaphore(max_workers)

    async def sem_task(task_coro):
        async with semaphore:
            return await task_coro

    tasks = [sem_task(asyncio.to_thread(ensure_output_idx, i)) for i in enumerate(inputs)]
    if not desc:
        desc = f"{f.__name__}"

    pbar = tqdm(total=len(inputs), desc=desc, disable=not user_tqdm)
    results = [None] * len(inputs)
    for task in asyncio.as_completed(tasks):
        idx, result = await task
        results[idx] = result
        pbar.update(1)
    return results


__all__ = ["multi_thread", "multi_process", "async_multi_thread"]
