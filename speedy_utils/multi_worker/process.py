import gc
import multiprocessing
import time
from multiprocessing import Pool
from typing import Any, Callable, Dict, List, Union

import pandas as pd
from loguru import logger
from tqdm import tqdm

from speedy_utils.multi_worker._handle_inputs import handle_inputs


def task_wrapper(func: Callable, index: int, input_kwargs: Dict[str, Any], results: List[Any]) -> None:
    """
    Wraps the function execution to capture the result along with its index.
    """
    try:
        result = func(**input_kwargs)
        results[index] = result
    except Exception as e:
        logger.error(f"Error in task {index}: {e}")
        results[index] = None  # Handle exception and return None for failed tasks


__gf__: callable = None


def multi_process(func: Callable, inputs: List[Dict[str, Any]], workers: int = 4, verbose: bool = True) -> List[Any]:
    """
    Executes a function concurrently across multiple processes with a list of dictionary inputs.
    Returns partial results on KeyboardInterrupt.
    """
    inputs = handle_inputs(func, inputs)
    
    global __gf__
    __gf__ = func
    manager = multiprocessing.Manager()
    results = manager.list([None] * len(inputs))  # Shared list for results
    # Create a pool of worker processes
    with Pool(processes=workers) as pool:
        tasks = []

        try:
            # Prepare and submit tasks to the pool
            for idx, inp in enumerate(inputs):
                task = pool.apply_async(task_wrapper, args=(func, idx, inp, results))
                tasks.append(task)

            # Display a progress bar if verbose is True
            with tqdm(total=len(tasks), desc="Processing", disable=not verbose) as pbar:
                for task in tasks:
                    task.wait()  # Wait for each task to complete
                    pbar.update(1)  # Update progress bar by 1 for each completed task

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt detected, returning partial results...")
            pool.terminate()  # Stop remaining processes
        finally:
            logger.debug("Closing and joining the pool...")
            pool.close()  # Ensure all processes are properly closed
            pool.join()  # Wait for all processes to complete
            gc.collect()  # Collect garbage to free up resources
    # Check results if more than 5 percent is None, then raise an error
    none_rate= sum(1 for r in results if r is None) / len(results)
    if none_rate > 0.05:
        logger.warning(f"{none_rate*100:0.2f} % of tasks failed. Consider increasing workers or checking input data.")
    __gf__ = None
    return list(results)


def f2(input):
    logger.debug(f"x={input}")
    return input


class Test:
    def f_simple(self, **kwargs):
        """
        A simple function to simulate a task.
        """
        y = kwargs["y"]
        x = kwargs["x"]
        logger.info(f"Starting task with x={x}, y={y}, sleep {x}")
        time.sleep(x)  # Simulate a delay
        result = x / y
        print(f"Done with x={x}, y={y}, result={result}")
        return result


if __name__ == "__main__":

    o = Test()
    # Inputs for testing
    inputs = [
        {"x": 1, "y": 2},
        {"x": 4, "y": 10},
        {"x": 3, "y": 0},
        {"x": 1, "y": 100},
        {"x": 5, "y": 6},
        {"x": 3, "y": 12},
    ]

    def f1(**kwargs):
        return o.f_simple(**kwargs)

    # f2 = lambda **x: o.f_simple(**x)
    results = multi_process(f2, inputs, workers=3, verbose=True)
    print(results)
