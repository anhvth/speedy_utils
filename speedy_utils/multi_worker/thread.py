import inspect
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from typing import Any, Callable, List

from loguru import logger
from tqdm import tqdm

from speedy_utils.multi_worker._handle_inputs import handle_inputs


def multi_thread(
    func: Callable,
    inputs: List[Any],
    workers: int = 4,
    verbose: bool = None,
    desc: str | None = None,
) -> List[Any]:

    inputs = handle_inputs(func, inputs)
    if workers == 1:
        return [func(inp) for inp in tqdm(inputs)]
    if verbose is None:
        if len(inputs) > 1000:
            verbose = True
        else:
            verbose = False
    if desc is None:
        fn_name = func.__name__
        try:
            source_file = inspect.getsourcefile(func) or "<string>"
            source_line = inspect.getsourcelines(func)[1]
            file_line = f"{source_file}:{source_line}"
        except (TypeError, OSError):
            file_line = "Unknown location"
        desc = f"{fn_name} at {file_line}"

    if isinstance(inputs[0], dict):
        orig_f = func
        func = lambda x: orig_f(**x)

    stop_event = threading.Event()
    results = [None] * len(inputs)  # Placeholder for results in order of inputs
    _log = logger.info if verbose else lambda x: None

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
                        _log(f"Error occurred in one of the threads: {str(e)[:1000]}")
            else:
                for future in as_completed(futures):
                    if stop_event.is_set():
                        break
                    try:
                        idx, result = future.result()
                        results[idx] = result  # Store result at the correct index
                    except Exception as e:
                        _log(f"An error occurred in one of the threads: {str(e)[:1000]}")

    except KeyboardInterrupt:
        _log("\nExecution manually interrupted. Cleaning up...")
        stop_event.set()

        for future in futures:
            if not future.done():
                future.cancel()
    except Exception as e:
        import traceback
        trace_back_str = traceback.format_exc()
        _log(f"An error occurred during execution: {trace_back_str}")
    finally:
        _log("Cleaning up any remaining threads or resources...")
        executor.shutdown(wait=False)
        
    none_rate= sum(1 for r in results if r is None) / len(results)
    if none_rate > 0.:
        logger.warning(f"{none_rate*100:0.2f} % of tasks failed. Consider increasing workers or checking input data.")
    return results
