import gc
import inspect
import random
import time
import traceback
import dill
from multiprocessing import Manager, Process
from threading import Thread
from typing import List, Literal

from fastcore.all import threaded, parallel
from loguru import logger
from tqdm import tqdm

from speedy_utils.common.clock import Clock
from speedy_utils.common.report_manager import ReportManager

THREADS = []  # Add a global variable to store threads


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
    report=True,
    input_type: Literal["single", "tuple", "dict", "df"] = "single",
    stop_on_error=True,
    **kwargs,
):

    if input_type == "df":
        inputs = inputs.to_dict(orient="records")
        input_type = "dict"
    clock = Clock()
    errors = []
    results = []

    def f_wrapper(item):
        try:
            dict_input = _convert_to_dict_input(func, item, input_type)
            if input_type == "dict":
                return func(**dict_input, **kwargs)
            elif input_type == "tuple":
                return func(*item,**kwargs)
            else:
                return func(item,**kwargs)
        except Exception as e:
            errors.append(
                {
                    "error": e,
                    "input": str(dict_input),
                    "traceback": _clean_traceback(traceback.format_exc()),
                }
            )
            if stop_on_error:
                raise e

    results = parallel(
        f_wrapper,
        inputs,
        n_workers=workers,
        progress=verbose,
        threadpool=True,
    )

    if report:
        try:
            metadata = {
                "mode": "multi_thread",
                "workers": workers,
                "total_inputs": len(inputs),
                "execution_mode": "multi_thread",
                "max_workers": workers,
                "function_name": func.__name__,
            }
            ReportManager().save_report(
                errors=errors,
                results=results,
                execution_time=clock.time_since_last_checkpoint(),
                metadata=metadata,
            )

        except Exception as e:
            logger.debug(f"Error saving report: {e}")

    return results


if __name__ == "__main__":

    def f(x):
        time.sleep(random.random())
        return x * x

    inputs = list(range(100))
    results = multi_thread(f, inputs, workers=4, verbose=True)
