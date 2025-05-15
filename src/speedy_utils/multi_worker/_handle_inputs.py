import functools
import inspect
from collections.abc import Callable, Iterable
from typing import Any, Dict, List, Union

import pandas as pd

# Example object


def _get_original_func(func):
    """
    Recursively unwrap a decorated function to find the actual
    original function object.
    """
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


def handle_inputs(
    f: Callable, inputs: list[dict[str, Any]] | list[Any] | pd.DataFrame
) -> list[dict[str, Any]]:
    # 1. Unwrap in case f is decorated (e.g., by @memoize).
    real_func = _get_original_func(f)

    # 2. Count parameters with inspect.signature.
    #    This handles normal or annotated arguments, etc.
    sig = inspect.signature(real_func)
    num_params = len(sig.parameters)

    # Convert certain input types to list to unify processing
    if isinstance(inputs, (range, list, tuple)):
        inputs = list(inputs)

    # 3. If exactly 1 parameter, we do the single-arg logic:
    if num_params == 1:
        # If the user passed a dataframe, break it into rows
        if isinstance(inputs, pd.DataFrame):
            inputs = [r for _, r in inputs.iterrows()]

        # For a single-arg function, turn each item into a dict: {arg_name: item}
        # so we can later call func(**inp)
        arg_name = next(iter(sig.parameters))  # name of the single parameter
        inputs = [{arg_name: input_} for input_ in inputs]
        return f, inputs

    else:

        return lambda x: f(x), inputs
