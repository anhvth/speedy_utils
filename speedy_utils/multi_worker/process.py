import gc
import multiprocessing
import time
from multiprocessing import Pool
from typing import Any, Callable, Dict, List

from loguru import logger
from tqdm import tqdm
import dill  # Add this import

from speedy_utils.multi_worker._handle_inputs import handle_inputs


def task_wrapper(func: Callable, index: int, input_kwargs: Dict[str, Any]) -> tuple:
    """
    Wraps the function execution to capture the result along with its index.
    """
    try:
        result = func(**input_kwargs)
        return index, result
    except Exception as e:
        logger.error(f"Error in task {index}: {e}")
        return index, None  # Handle exception and return None for failed tasks


def multi_process(
    func: Callable, inputs: List[Dict[str, Any]], workers: int = 4, verbose: bool = True
) -> List[Any]:
    """
    Executes a function concurrently across multiple processes with a list of dictionary inputs.
    Ensures ordered output by preserving task indices.
    """
    inputs = handle_inputs(func, inputs)

    with multiprocessing.Manager() as manager:
        # Create a pool of worker processes
        with Pool(processes=workers, initializer=dill.load, initargs=(dill.dumps(func),)) as pool:
            tasks = []
            try:
                # Prepare and submit tasks to the pool
                for idx, inp in enumerate(inputs):
                    task = pool.apply_async(task_wrapper, args=(func, idx, inp))
                    tasks.append(task)

                # Display a progress bar if verbose is True
                results = []
                with tqdm(total=len(tasks), desc="Processing", disable=not verbose) as pbar:
                    for task in tasks:
                        idx, result = task.get()  # Wait for each task to complete and retrieve its result
                        results.append((idx, result))
                        pbar.update(1)  # Update progress bar by 1 for each completed task

            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt detected, returning partial results...")
                pool.terminate()  # Stop remaining processes
            finally:
                logger.debug("Closing and joining the pool...")
                pool.close()  # Ensure all processes are properly closed
                pool.join()  # Wait for all processes to complete
                gc.collect()  # Collect garbage to free up resources

        # Sort results by task index to ensure ordered output
        results.sort(key=lambda x: x[0])
        ordered_results = [result for _, result in results]

        # Check if more than 5 percent of tasks failed, then raise a warning
        none_rate = sum(1 for r in ordered_results if r is None) / len(ordered_results)
        if none_rate > 0.05:
            logger.warning(
                f"{none_rate*100:0.2f} % of tasks failed. Consider increasing workers or checking input data."
            )

    return ordered_results


# Example usage
if __name__ == "__main__":
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
            logger.info(f"Done with x={x}, y={y}, result={result}")
            return result

    o = Test()
    inputs = [
        {"x": 1, "y": 2},
        {"x": 4, "y": 10},
        {"x": 3, "y": 1},
        {"x": 1, "y": 100},
        {"x": 5, "y": 6},
        {"x": 3, "y": 12},
    ]

    def f1(**kwargs):
        return o.f_simple(**kwargs)

    results = multi_process(f1, inputs, workers=3, verbose=True)
    print(results)