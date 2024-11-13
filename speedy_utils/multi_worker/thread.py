import inspect
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Any, Callable, List, Tuple

from loguru import logger
from tqdm import tqdm

from speedy_utils.common.utils_print import fprint
from speedy_utils.multi_worker._handle_inputs import handle_inputs


def _get_function_info(func: Callable) -> str:
    """Get function name and location information."""
    fn_name = func.__name__
    try:
        source_file = inspect.getsourcefile(func) or "<string>"
        source_line = inspect.getsourcelines(func)[1]
        file_line = f"{source_file}:{source_line}"
    except (TypeError, OSError):
        file_line = "Unknown location"
    return f"Function: `{fn_name}` at {file_line}"


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
) -> List[Any]:
    """Execute tasks in parallel using multiple threads."""
    inputs = handle_inputs(func, orig_inputs)
    verbose = len(inputs) > 1000 if verbose is None else verbose
    desc = "Multi-thread, " + _get_function_info(func) if desc is None else desc
    desc = desc[:30] + ".."

    stop_event = threading.Event()
    results = [None] * len(inputs)
    result_counter = defaultdict(int)
    _log = logger.info if verbose else lambda x: None

    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_process_single_task, func, i, inp, stop_event)
                for i, inp in enumerate(inputs)
            ]
            t = time.time()
            counter = 0
            with tqdm(
                total=len(futures), desc=desc, disable=not verbose, ncols=120
            ) as pbar:
                for future in as_completed(futures):
                    if stop_event.is_set():
                        logger.info("Stopping due to error")
                        break
                    _handle_results(
                        future, results, stop_event, stop_on_error, result_counter, pbar
                    )
                    last_t = time.time()
                    counter += 1
                    if last_t - t > 0.1:
                        pbar.set_postfix(dict(result_counter))
                        t = last_t
                        pbar.update(counter)
                        counter = 0
                pbar.set_postfix(dict(result_counter))
                pbar.update(counter)

    except KeyboardInterrupt:
        logger.warning("Execution manually interrupted")
        stop_event.set()

    except Exception as e:
        import traceback

        _log(f"An error occurred during execution: {traceback.format_exc()}")

    finally:
        if verbose:
            print("Multi thread results:")
            fprint(result_counter, "Result counter", is_notebook=False)

    return results
