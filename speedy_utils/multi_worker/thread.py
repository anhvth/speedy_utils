import inspect
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Tuple

from loguru import logger
from speedy_utils.common.utils_print import fprint
from speedy_utils.multi_worker._handle_inputs import handle_inputs
from tqdm import tqdm


def _get_function_info(func: Callable) -> str:
    """Get function name and location information."""
    fn_name = func.__name__
    try:
        source_file = inspect.getsourcefile(func) or "<string>"
        source_line = inspect.getsourcelines(func)[1]
        file_line = f"{source_file}:{source_line}"
    except (TypeError, OSError):
        file_line = "Unknown location"
    return f"Function: `{fn_name}"


class RunErr(Exception):
    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        self.message = message

    def to_dict(self) -> dict:
        return {"error_type": self.error_type, "message": self.message}

    def __str__(self) -> str:
        formated = f"{self.error_type}: {self.message}"
        return formated


def _process_single_task(
    func: Callable, idx: int, inp: Any, stop_event: threading.Event
) -> Tuple[int, Any]:
    """Process a single task and handle exceptions."""
    if stop_event.is_set():
        return idx, None

    try:
        ret = func(inp) if not isinstance(inp, dict) else func(**inp)
        return idx, ret
    except Exception as e:
        import traceback

        return idx, RunErr(type(e).__name__, traceback.format_exc())


def _handle_results(
    future: Any,
    results: List[Any],
    stop_event: threading.Event,
    stop_on_error: bool,
    result_counter: defaultdict,
    pbar: tqdm,
) -> None:
    """Handle the results from completed futures."""
    idx, result_or_error = future.result()
    results[idx] = result_or_error

    if isinstance(result_or_error, RunErr):
        error_key = f"Error_{result_or_error.error_type}"
        result_counter[error_key] += 1
        if result_counter[error_key] == 1:
            logger.error(
                f"First error of type {result_or_error.error_type}: {result_or_error.message}"
            )
        if stop_on_error:
            stop_event.set()
    else:
        result_counter["SUCCESS"] += 1


def multi_thread(
    func: Callable,
    orig_inputs: List[Any],
    workers: int = 4,
    verbose: bool = None,
    desc: str = None,
    stop_on_error: bool = False,
    filter_none: bool = False,
    do_memoize: bool = False,
) -> List[Any]:
    """
    Execute the given function `func` on a list of inputs `orig_inputs` in parallel
    using multiple threads. Manages progress tracking, handling of errors, and optional
    filtering of None results.

    Parameters
    ----------
    func : Callable
        The function to be executed for each input.
    orig_inputs : List[Any]
        The list of inputs to process.
    workers : int, optional
        Number of worker threads to use. Defaults to 4. If set to 1, the execution is
        effectively single-threaded.
    verbose : bool, optional
        If True, shows a tqdm progress bar. If None, it defaults to True if `desc` is
        provided, otherwise False.
    desc : str, optional
        The description for the progress bar. Providing this automatically sets
        verbose to True if verbose is None.
    stop_on_error : bool, optional
        If True, stops execution upon the first exception encountered in any thread.
    filter_none : bool, optional
        If True, filters out None results from the final list.

    Returns
    -------
    List[Any]
        The list of results from applying `func` to each element in `orig_inputs`,
        in the original order. May exclude None values if `filter_none` is True.

    Notes
    -----
    - If the environment variable SPEEDY_DEBUG is set to "1", the function forces
      `workers` to 1 and runs in single-threaded mode, aiding debug.
    - Uses a ThreadPoolExecutor with as_completed to manage parallel tasks.
    - Displays progress via tqdm if verbose is True.
    """
    if do_memoize:
        # apply memoize on the function, need to make sure the method
        from speedy_utils import memoize

        func = memoize(func)
    if verbose is None:
        # Default verbosity based on whether a desc is provided
        verbose = bool(desc)

    if bool(int(os.getenv("SPEEDY_DEBUG", "0"))):
        logger.debug("Running in debug mode, setting workers to 1")
        workers = 1

    # If workers <= 1, no parallelization
    if workers <= 1:
        return [func(inp) for inp in tqdm(orig_inputs, desc="Single thread")]

    inputs = handle_inputs(func, orig_inputs)
    stop_event = threading.Event()
    results = [None] * len(inputs)
    result_counter = defaultdict(int)

    log_fn = logger.info if verbose else (lambda x: None)

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_process_single_task, func, i, inp, stop_event)
                for i, inp in enumerate(inputs)
            ]

            last_update_time = time.time()
            completed_since_last_update = 0

            with tqdm(
                total=len(futures), desc=desc, disable=not verbose, ncols=120
            ) as pbar:
                for future in as_completed(futures):
                    if stop_event.is_set():
                        logger.info("Stopping execution due to an error.")
                        break

                    _handle_results(
                        future, results, stop_event, stop_on_error, result_counter, pbar
                    )

                    completed_since_last_update += 1
                    current_time = time.time()

                    # Update progress bar and postfix periodically
                    if current_time - last_update_time > 0.1:
                        pbar.set_postfix(dict(result_counter))
                        pbar.update(completed_since_last_update)
                        completed_since_last_update = 0
                        last_update_time = current_time

                # Final update for any remaining tasks
                pbar.set_postfix(dict(result_counter))
                pbar.update(completed_since_last_update)

    except KeyboardInterrupt:
        logger.warning("Execution manually interrupted.")
        stop_event.set()

    except Exception as e:
        import traceback

        log_fn(f"An error occurred during execution: {traceback.format_exc()}")

    finally:
        if verbose:
            print("Multi-thread results:")
            fprint(result_counter, "Result counter", is_notebook=False)

    if filter_none:
        results = [r for r in results if r is not None]

    return results
