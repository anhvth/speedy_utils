import os
from concurrent.futures import TimeoutError as FuturesTimeoutError
import gc
import time
import threading
import traceback
from collections import defaultdict
from multiprocessing import Manager
from concurrent.futures import (
    ThreadPoolExecutor,
    wait,
    TimeoutError as FuturesTimeoutError,  # Currently unused, but kept for reference
)
from typing import Any, Callable, List, Optional, Tuple

from fastcore.all import threaded
from loguru import logger
from tqdm import tqdm

from speedy_utils.multi_worker._handle_inputs import handle_inputs


class RunErr(Exception):
    """Custom exception to wrap and capture details of runtime errors."""

    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        self.message = message
        self.traceback = traceback.format_exc()

    def to_dict(self) -> dict:
        return {"error_type": self.error_type, "message": self.message}

    def __str__(self) -> str:
        return f"{self.error_type}: {self.message}"

    def __repr__(self) -> str:
        return f"RunErr({self.error_type})"


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
        return idx, RunErr(type(e).__name__, str(e))


class Result(List[Any]):
    """Custom list class that logs errors and sets a flag to stop on error."""

    should_stop = False

    def __setitem__(self, index: int, value: Any) -> None:
        super().__setitem__(index, value)
        if isinstance(value, RunErr):
            logger.error(f"Error at index {index}: {value}")
            self.should_stop = True
        else:
            logger.debug(f"Result at index {index} set to {value}")


def multi_thread(
    func: Callable[..., Any],
    orig_inputs: List[Any],
    workers: int = 4,
    verbose: Optional[bool] = None,
    desc: Optional[str] = None,
    stop_on_error: bool = False,
    filter_none: bool = False,
    do_memoize: bool = False,
    share_across_processes: bool = True,
    **kwargs: Any,
) -> List[Any]:
    """
    Execute `func` on items in `orig_inputs` in parallel using multiple threads.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to be executed for each input.
    orig_inputs : List[Any]
        Inputs to process.
    workers : int, optional
        Number of worker threads. Defaults to 4. If 1, runs single-threaded.
    verbose : Optional[bool], optional
        If True, shows a tqdm progress bar. If None, auto-set based on `desc`.
    desc : Optional[str], optional
        Description for the progress bar. If provided, `verbose` defaults to True.
    stop_on_error : bool, optional
        If True, stop all threads at the first exception.
    filter_none : bool, optional
        If True, None results are omitted from final output.
    do_memoize : bool, optional
        If True, memoizes `func`.
    share_across_processes : bool, optional
        If True, use a multiprocessing Manager for the results.
    **kwargs : Any
        Additional arguments (e.g., legacy `n_threads`).

    Returns
    -------
    List[Any]
        List of results in the original order. May exclude None if `filter_none=True`.
    """
    if "n_threads" in kwargs:
        logger.warning(
            "The 'n_threads' argument is deprecated. Please use 'workers' instead."
        )
        workers = kwargs["n_threads"]

    if do_memoize:
        from speedy_utils import memoize  # type: ignore

        func = memoize(func)

    if verbose is None:
        verbose = bool(desc)

    # If debugging, run single-threaded
    if bool(int(os.getenv("SPEEDY_DEBUG", "0"))):
        logger.debug("Running in debug mode; setting workers to 1.")
        workers = 1

    # Run single-threaded if workers <= 1
    if workers <= 1:
        return [
            func(inp)
            for inp in tqdm(orig_inputs, desc="Single thread", disable=not verbose)
        ]

    inputs = handle_inputs(func, orig_inputs)
    stop_event = threading.Event()

    # Create results list
    if share_across_processes:
        manager = Manager()
        shared_result_list = manager.list([None] * len(orig_inputs))
    else:
        shared_result_list: Result = Result([None] * len(inputs))

    result_counter = defaultdict(int)

    process = _execute_tasks_in_parallel(
        func,
        workers,
        verbose,
        desc,
        stop_on_error,
        inputs,
        stop_event,
        shared_result_list,
        result_counter,
    )

    # Wait for the background process to finish
    while process.is_alive():
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt detected. Stopping execution.")
            stop_event.set()
            break

    # Optionally filter out None results
    final_results = (
        [r for r in shared_result_list if r is not None]
        if filter_none
        else list(shared_result_list)
    )

    # Explicitly remove references to the process and manager
    process = None
    if share_across_processes:
        manager.shutdown()
    gc.collect()  # Force a garbage collection sweep

    return final_results


@threaded(process=True)
def _execute_tasks_in_parallel(
    func: Callable[..., Any],
    workers: int,
    verbose: bool,
    desc: Optional[str],
    stop_on_error: bool,
    inputs: List[Any],
    stop_event: threading.Event,
    results: List[Any],
    result_counter: defaultdict,
) -> List[Any]:
    """Spawn threads to execute tasks in parallel, updating a progress bar if requested."""
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {}
            futures = []
            for i, inp in enumerate(inputs):
                fut = executor.submit(_process_single_task, func, i, inp, stop_event)
                futures.append(fut)
                future_to_index[fut] = i

            pending = set(futures)
            with tqdm(
                total=len(futures),
                desc=desc,
                disable=not verbose,
                ncols=120,
                leave=verbose,
            ) as pbar:
                while pending and not stop_event.is_set():
                    done, pending = wait(
                        pending, timeout=0.1, return_when="FIRST_EXCEPTION"
                    )
                    completed_count = 0

                    for future in done:
                        idx = future_to_index[future]
                        _, result_or_error = future.result()
                        is_error = isinstance(result_or_error, RunErr)

                        results[idx] = result_or_error
                        completed_count += 1
                        result_counter["SUCCESS"] += 1

                        if is_error and stop_on_error:
                            error_msg = f"Error at index {idx}: {result_or_error}"
                            logger.error(f"{error_msg}")
                            stop_event.set()
                            for fut in pending:
                                fut.cancel()
                            break

                    if completed_count > 0:
                        pbar.set_postfix(dict(result_counter))
                        pbar.update(completed_count)
        logger.debug(f"All tasks completed. Results: {str(results)[:100]} ...")
    except FuturesTimeoutError:
        logger.error("Timeout error occurred.")
        stop_event.set()
    finally:
        # Clear references to help garbage collection
        futures.clear()
        future_to_index.clear()
        gc.collect()
