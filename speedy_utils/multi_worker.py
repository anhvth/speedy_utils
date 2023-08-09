import inspect
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from typing import Any, Callable, List
import asyncio
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

    with ThreadPoolExecutor(max_workers=workers) as executor:
        # Use executor.map to apply func to inputs in order
        map_func = executor.map(func, inputs)
        if verbose:
            results = list(tqdm(map_func, total=len(inputs), desc=desc))
        else:
            results = list(map_func)
    return results


def _init_pool_processes(func):
    global _func
    _func = func


def _pool_process_executor(args):
    # Unpack arguments if necessary
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
        desc = f"{fn_name} at {file_line}"

    if os.environ.get("DEBUG", "0") == "1":
        logger.info("DEBUGGING set num workers to 1")
        workers = 1

    logger.info("Multi-processing {} | Num samples: {}", desc, len(inputs))

    results = []
    with Pool(
        processes=workers, initializer=_init_pool_processes, initargs=(func,)
    ) as pool:
        try:
            if verbose:
                for result in tqdm(
                    pool.imap(_pool_process_executor, inputs),
                    total=len(inputs),
                    desc=desc,
                ):
                    results.append(result)
            else:
                results = pool.map(_pool_process_executor, inputs)
        except Exception as e:
            logger.error(f"[multiprocess] Error {e}")

    return results


async def async_multi_thread(f, inputs, desc="", user_tqdm=True):
    """
    Uasge:
        inputs = list(range(10))
        def function(i):
            time.sleep(1)
            return 1/i
        results = await amulti_thread(function, inputs)
    """

    def ensure_output_idx(idx_i):
        idx, i = idx_i
        return idx, f(i)

    tasks = [asyncio.to_thread(ensure_output_idx, i) for i in enumerate(inputs)]
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
