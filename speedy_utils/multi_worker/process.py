import gc
import multiprocessing
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from loguru import logger
import time


def task_wrapper(func, index, input_kwargs, results):
    """
    Wraps the function execution to capture the result along with its index.
    """
    try:
        result = func(**input_kwargs)
        results[index] = result
    except Exception as e:
        logger.error(f"Error processing input at index {index}: {e}")
        results[index] = None  # Handle exception and return None for failed tasks


def handle_inputs(inputs):
    if isinstance(inputs, pd.DataFrame):
        logger.info("Converting DataFrame to list of dictionaries...")
        inputs = inputs.to_dict('records')
    return inputs

def multi_process(func, inputs, workers=4, verbose=True):
    """
    Executes a function concurrently across multiple processes with a list of dictionary inputs.
    Returns partial results on KeyboardInterrupt.
    """
    inputs = handle_inputs(inputs)
    manager = multiprocessing.Manager()
    results = manager.list([None] * len(inputs))  # Shared list for results

    pool = Pool(processes=workers)
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
        # Terminate remaining processes and return the results collected so far
        pool.terminate()  # Stop remaining processes
        pool.join()  # Clean up the worker pool
        return list(results)  # Return the results so far

    finally:
        pool.close()  # Ensure all processes are properly closed
        pool.join()  # Wait for all processes to complete
        gc.collect()  # Collect garbage to free up resources

    return list(results)


if __name__ == "__main__":

    def f_simple(x, y):
        """
        A simple function to simulate a task.
        """
        print(f"Starting task with x={x}, y={y}, sleep {x}")
        time.sleep(x)  # Simulate a delay
        result = x + y
        print(f"Done with x={x}, y={y}, result={result}")
        return result

    # Inputs for testing
    inputs = [
        {"x": 1, "y": 2},
        {"x": 9, "y": 10},
        {"x": 3, "y": 4},
        {"x": 1, "y": 100},
        {"x": 5, "y": 6},
        {"x": 11, "y": 12},
    ]

    try:
        results = multi_process(f_simple, inputs, workers=3, verbose=True)
        print("Results:", results)
    except KeyboardInterrupt:
        print("Process was interrupted.")
