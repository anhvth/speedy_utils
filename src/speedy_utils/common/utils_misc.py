# utils/utils_misc.py

import inspect
import os
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


def mkdir_or_exist(dir_name: str) -> None:
    """Create a directory if it doesn't exist."""
    os.makedirs(dir_name, exist_ok=True)


def flatten_list(list_of_lists: list[list[Any]]) -> list[Any]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]


def get_arg_names(func: Callable) -> list[str]:
    """Retrieve argument names of a function."""
    return inspect.getfullargspec(func).args


def is_notebook() -> bool:
    try:
        if "get_ipython" in globals().keys():
            get_ipython = globals()["get_ipython"]
            shell = get_ipython().__class__.__name__
            if shell == "ZMQInteractiveShell":
                return True  # Jupyter notebook or qtconsole
        return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def convert_to_builtin_python(input_data: Any) -> Any:
    """Convert input data to built-in Python types."""
    if isinstance(input_data, dict):
        return {k: convert_to_builtin_python(v) for k, v in input_data.items()}
    elif isinstance(input_data, list):
        return [convert_to_builtin_python(v) for v in input_data]
    elif isinstance(input_data, (int, float, str, bool, type(None))):
        return input_data
    elif isinstance(input_data, BaseModel):
        data = input_data.model_dump_json()
        return convert_to_builtin_python(data)
    else:
        raise ValueError(f"Unsupported type {type(input_data)}")


__all__ = [
    "mkdir_or_exist",
    "flatten_list",
    "get_arg_names",
    "is_notebook",
    "convert_to_builtin_python",
]
