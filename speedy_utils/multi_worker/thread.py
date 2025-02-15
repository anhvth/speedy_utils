import functools
import inspect
import os
import signal
import threading
import time
import traceback
from collections import defaultdict
from concurrent.futures import (
    FIRST_COMPLETED,
    FIRST_EXCEPTION,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from fastcore.all import threaded
from typing import Any, Callable, List, Optional, Tuple

from loguru import logger
from tqdm import tqdm

from speedy_utils.multi_worker._handle_inputs import handle_inputs


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


def multi_thread(
    func: Callable[..., Any],
    orig_inputs: List[Any],
    workers: int = 4,
    verbose: Optional[bool] = None,
    desc: Optional[str] = None,
    stop_on_error: bool = False,
    filter_none: bool = False,
    do_memoize: bool = False,
    per_run_timeout: Optional[int] = None,
    **kwargs: Any,
) -> List[Any]:
    """
    Execute the given function `func` on a list of inputs `orig_inputs` in parallel
    using multiple threads. Manages progress tracking, handling of errors, and optional
    filtering of None results.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to be executed for each input.
    orig_inputs : List[Any]
        The list of inputs to process.
    workers : int, optional
        Number of worker threads to use. Defaults to 4. If set to 1, the execution is
        effectively single-threaded.
    verbose : Optional[bool], optional
        If True, shows a tqdm progress bar. If None, it defaults to True if `desc` is
        provided, otherwise False.
    desc : Optional[str], optional
        The description for the progress bar. Providing this automatically sets
        verbose to True if verbose is None.
    stop_on_error : bool, optional
        If True, stops execution upon the first exception encountered in any thread.
    filter_none : bool, optional
        If True, filters out None results from the final list.
    do_memoize : bool, optional
        If True, memoizes the function using speedy_utils.memoize.

    Returns
    -------
    List[Any]
        The list of results from applying `func` to each element in `orig_inputs`,
        in the original order. May exclude None values if `filter_none` is True.
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
        # Default verbosity based on whether a description is provided
        verbose = bool(desc)

    if bool(int(os.getenv("SPEEDY_DEBUG", "0"))):
        logger.debug("Running in debug mode, setting workers to 1")
        workers = 1

    # If workers <= 1, run in single-threaded mode
    if workers <= 1:
        return [func(inp) for inp in tqdm(orig_inputs, desc="Single thread")]

    inputs = handle_inputs(func, orig_inputs)
    stop_event = threading.Event()
    results: List[Any] = [None] * len(inputs)
    result_counter = defaultdict(int)

    log_fn: Callable[[str], None] = logger.info if verbose else (lambda x: None)

    main_thread = __execute_tasks_in_parallel(
        func,
        workers,
        verbose,
        desc,
        stop_on_error,
        inputs,
        stop_event,
        results,
        result_counter,
        log_fn,
        per_run_timeout,
    )

    while main_thread.is_alive():
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, stopping execution.")
            stop_event.set()
            break

    if filter_none:
        results = [r for r in results if r is not None]

    return results


from concurrent.futures import (
    ThreadPoolExecutor,
    wait,
    TimeoutError as FuturesTimeoutError,
)


@threaded
def __execute_tasks_in_parallel(
    func,
    workers,
    verbose,
    desc,
    stop_on_error,
    inputs,
    stop_event,
    results,
    result_counter,
    log_fn,
    per_run_timeout,
):
    import traceback
    import gc

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Keep track of which future corresponds to which index
            future_to_index = {}
            futures = []
            for i, inp in enumerate(inputs):
                fut = executor.submit(_process_single_task, func, i, inp, stop_event)
                futures.append(fut)
                future_to_index[fut] = i

            last_progress_time = time.time()
            completed_since_last_update = 0
            pending = set(futures)

            # Use tqdm for progress
            with tqdm(
                total=len(futures),
                desc=desc,
                disable=not verbose,
                ncols=120,
                leave=not verbose,
            ) as pbar:
                while pending and not stop_event.is_set():
                    # try:
                    done, pending = wait(
                        pending, timeout=0.1, return_when="FIRST_COMPLETED"
                    )
                    # except KeyboardInterrupt:
                    #     log_fn("Received keyboard interrupt, stopping execution.")
                    #     currently_peending = [f for f in pending]
                    #     logger.info(
                    #         f"Pending: {len(currently_peending)}, Done: {len(done)}"
                    #     )

                    #     stop_event.set()
                    #     break

                    for future in done:
                        idx = future_to_index[future]
                        try:
                            # If per_run_timeout is set, try retrieving the result with a timeout
                            timeout = per_run_timeout if per_run_timeout else None
                            idx, result_or_error = future.result(timeout=timeout)

                            # Normal successful result
                            results[idx] = result_or_error
                            completed_since_last_update += 1
                            result_counter["SUCCESS"] += 1

                        except FuturesTimeoutError:
                            # Mark result as a timeout
                            logger.error(
                                f"Timeout error at idx {idx}, took longer than {per_run_timeout} seconds."
                            )
                            results[idx] = RunErr(
                                "TimeoutError",
                                f"Function took longer than {per_run_timeout} seconds.",
                            )
                            result_counter["Error_TimeoutError"] += 1

                            if stop_on_error:
                                logger.info("Stopping execution due to timeout error.")
                                stop_event.set()

                        except Exception as e:
                            # Catch any other errors
                            exc = RunErr(type(e).__name__, traceback.format_exc())
                            results[idx] = exc
                            result_counter[f"Error_{exc.error_type}"] += 1

                            logger.error(
                                f"Error of type {exc.error_type} at idx {idx}: {exc.message}"
                            )
                            if stop_on_error:
                                logger.info("Stopping execution due to error.")
                                stop_event.set()

                    current_time = time.time()
                    if completed_since_last_update > 0:
                        pbar.set_postfix(dict(result_counter))
                        pbar.update(completed_since_last_update)
                        completed_since_last_update = 0

                    # Debug log if no progress for > 5 seconds
                    if (current_time - last_progress_time) > 5:
                        logger.debug(
                            f"Stuck waiting for tasks to complete. Pending tasks: {len(pending)}"
                        )
                        last_progress_time = current_time

                # Final update for any un-updated completions
                if completed_since_last_update > 0:
                    pbar.set_postfix(dict(result_counter))
                    pbar.update(completed_since_last_update)

            if verbose:
                pbar.close()

    except Exception:
        log_fn(f"An error occurred during execution: {traceback.format_exc()}")
    finally:
        ret = gc.collect()
        logger.debug(f"Garbage collector: collected {ret} objects")
