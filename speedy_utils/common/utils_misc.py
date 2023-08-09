# utils/utils_misc.py

import inspect
import os
import sys
from typing import Any, Callable, List
from IPython import get_ipython
from openai import BaseModel


def mkdir_or_exist(dir_name: str) -> None:
    """Create a directory if it doesn't exist."""
    os.makedirs(dir_name, exist_ok=True)

def flatten_list(list_of_lists: List[List[Any]]) -> List[Any]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]

def get_arg_names(func: Callable) -> List[str]:
    """Retrieve argument names of a function."""
    return inspect.getfullargspec(func).args



def is_interactive() -> bool:
    """Check if the environment is interactive (e.g., Jupyter notebook)."""
    try:
        get_ipython()
        return True
    except NameError:
        return len(sys.argv) == 1


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
