import inspect
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from typing import Any, Callable, List

from loguru import logger
from tqdm import tqdm

from speedy_utils.multi_worker._handle_inputs import handle_inputs


# @handle_inputs
def multi_thread(
    func: Callable,
    inputs: List[Any],
    workers: int = 4,
    verbose: bool = True,
    desc: str | None = None,
) -> List[Any]:
    if workers == 1:
        return [func(inp) for inp in tqdm(inputs)]
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




