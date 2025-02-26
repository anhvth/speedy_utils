import gc
import inspect
import random
import time
import traceback
from multiprocessing import Manager, Process
from threading import Thread
from typing import List, Literal

from fastcore.all import threaded
from loguru import logger
from tqdm import tqdm

from speedy_utils.common.clock import Clock
from speedy_utils.common.report_manager import ReportManager
from speedy_utils.common.utils_print import setup_logger


def _convert_to_dict_input(func, args, input_type):
    """Convert positional arguments to dictionary using function parameter names"""
    if input_type == "dict":
        return args if isinstance(args, dict) else {}
    elif input_type == "tuple":
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        return dict(zip(params, args)) if isinstance(args, (list, tuple)) else {}
    elif input_type == "single":
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        return {params[0]: args} if len(params) == 1 else {}
    return {}


def _clean_traceback(tb_text: str) -> str:
    """Remove unnecessary lines from traceback"""
    lines = tb_text.split("\n")
    filtered_lines = []
    skip_next = False
    for line in lines:
        if "Traceback (most recent call last):" in line:
            continue
        if "speedy_utils/multi_worker/thread.py" in line:
            skip_next = True
            continue
        if skip_next:
            skip_next = False
            continue
        filtered_lines.append(line)
    return "\n".join(line for line in filtered_lines if line.strip())


def multi_thread(
    func: callable,
    inputs: List[any],
    workers=64,
    verbose=True,
    process=False,
    desc="",
    report=True,
    input_type: Literal["single", "tuple", "dict", "df"] = "single",
    reducer: Literal["flatten_list", "None"] = "None",
    filter_none=False,
):
    if input_type == "df":
        inputs = inputs.to_dict(orient="records")
        input_type = "dict"
    if workers <= 1:

        return [func(i) for i in tqdm(inputs, desc=desc)]
    clock = Clock()
    manager = Manager()
    errors = manager.list()
    process = False
    shared_results = manager.dict()
    completed_task_count = manager.Value("i", 0)
    process_lock = manager.Lock()
    results = {}

    @threaded(process=process)
    def f_wrapper_process(i_id, item):
        try:
            dict_input = _convert_to_dict_input(func, item, input_type)
            if input_type == "dict":
                result = func(**dict_input)
            elif input_type == "tuple":
                result = func(*item)
            else:  # input_type == "single"
                result = func(item)
        except Exception as e:

            errors.append(
                {
                    "index": i_id,
                    "error": e,
                    "input": str(dict_input),
                    "traceback": _clean_traceback(traceback.format_exc()),
                }
            )
            result = None

        with process_lock:
            completed_task_count.value += 1
            if process:
                shared_results[i_id] = result
            else:
                results[i_id] = result

    running_f: List[Thread | Process] = []
    pbar = tqdm(
        total=len(inputs),
        disable=not verbose,
        desc=desc,
        smoothing=0.1,
        colour="green",
        position=0,
        mininterval=2,
        ncols=80,
        leave=True,
    )
    inputs = [(i, inputs[i]) for i in range(len(inputs))]
    total = len(inputs)
    # clock = Clock()
    while completed_task_count.value < total:
        num_running = len(running_f)
        while num_running < workers and len(inputs) > 0:
            i_id, item = inputs.pop(0)
            process_or_thread = f_wrapper_process(i_id, item)
            running_f.append(process_or_thread)
            num_running = len(running_f)

        with process_lock:
            to_pop = []
            for i, p in enumerate(running_f):
                if not p.is_alive():
                    to_pop.append(i)
                    pbar.update(1)

            running_f = [running_f[i] for i in range(len(running_f)) if i not in to_pop]
    pbar.update(total - pbar.n)
    pbar.close()

    for p in running_f:
        p.join()
    if not results:
        results = [shared_results[i] for i in range(len(inputs))]
    gc.collect()
    final_results = [results[i] for i in range(len(results))]

    if filter_none:
        logger.debug("Filtering None values from results")
        final_results = [item for item in final_results if item is not None]
    if reducer == "flatten_list":
        logger.debug("Flattening list")
        final_results = [item for sublist in final_results for item in sublist]

    if report:
        try:
            metadata = {
                "workers": workers,
                "mode": "process" if process else "thread",
                "total_inputs": len(inputs),
                "execution_mode": (
                    "multi_process"
                    if process
                    else "multi_thread" if workers > 1 else "sequential"
                ),
                "max_workers": workers,
                "description": desc
                or func.__name__,  # Use description if provided, otherwise function name
                "function_name": func.__name__,
            }
            ReportManager().save_report(
                errors=errors,
                results=final_results,
                execution_time=clock.time_since_last_checkpoint(),
                metadata=metadata,
            )

        except Exception as e:
            logger.debug(f"Error saving report: {e}")

    return final_results


if __name__ == '__main__':
    def f(x):
        time.sleep(random.random())
        return x * x
    inputs = list(range(100))
    results = multi_thread(f, inputs, workers=4, verbose=True)