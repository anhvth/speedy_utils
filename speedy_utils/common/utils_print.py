# utils/utils_print.py

import copy
import json
import pprint
import textwrap
from typing import Any, Dict, List, Optional

from IPython.display import HTML, display
from openai import BaseModel
from pandas import get_option
from tabulate import tabulate

from .utils_misc import is_interactive


def display_pretty_table_html(data: Dict) -> None:
    """
    Display a pretty HTML table in Jupyter notebooks.
    """
    table = "<table>"
    for key, value in data.items():
        table += f"<tr><td>{key}</td><td>{value}</td></tr>"
    table += "</table>"
    display(HTML(table))


# Flattening the dictionary using "." notation for keys
def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def fprint(
    input_data: Any,
    key_ignore: Optional[List[str]] = None,
    key_keep: Optional[List[str]] = None,
    max_width: int = 100,
    indent: int = 2,
    depth: Optional[int] = None,
    table_format: str = "grid",
    str_wrap_width: int = 80,
    grep=None,
) -> None:
    """
    Pretty print structured data.
    """
    if hasattr(input_data, "model_dump"):
        input_data = input_data.model_dump()
    if not isinstance(input_data, (dict, str)):
        raise ValueError("Input data must be a dictionary or string")

    if isinstance(input_data, dict):
        input_data = flatten_dict(input_data)
    
    if grep is not None:
        input_data = {k: v for k, v in input_data.items() if grep in str(k)}

    def remove_keys(d: Dict, keys: List[str]) -> Dict:
        """Remove specified keys from a dictionary."""
        for key in keys:
            parts = key.split(".")
            sub_dict = d
            for part in parts[:-1]:
                sub_dict = sub_dict.get(part, {})
            sub_dict.pop(parts[-1], None)
        return d

    def keep_keys(d: Dict, keys: List[str]) -> Dict:
        """Keep only specified keys in a dictionary."""
        result = {}
        for key in keys:
            parts = key.split(".")
            sub_source = d
            sub_result = result
            for part in parts[:-1]:
                if part not in sub_source:
                    break
                sub_result = sub_result.setdefault(part, {})
                sub_source = sub_source[part]
            else:
                sub_result[parts[-1]] = copy.deepcopy(sub_source.get(parts[-1]))
        return result

    if hasattr(input_data, "to_dict"):
        input_data = input_data.to_dict()

    processed_data = copy.deepcopy(input_data)

    if isinstance(processed_data, dict):
        if key_keep is not None:
            processed_data = keep_keys(processed_data, key_keep)
        elif key_ignore is not None:
            processed_data = remove_keys(processed_data, key_ignore)

        if is_interactive():
            display_pretty_table_html(processed_data)
            return

    if isinstance(processed_data, dict):
        table = [[k, v] for k, v in processed_data.items()]
        print(
            tabulate(
                table,
                headers=["Key", "Value"],
                tablefmt=table_format,
                maxcolwidths=[None, max_width],
            )
        )
    elif isinstance(processed_data, str):
        wrapped_text = textwrap.fill(processed_data, width=str_wrap_width)
        print(wrapped_text)
    else:
        printer = pprint.PrettyPrinter(width=max_width, indent=indent, depth=depth)
        printer.pprint(processed_data)


def print_table(data: Any) -> None:
    """
    Print data as a table.
    """

    def __get_table(data: Any) -> str:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError("String input could not be decoded as JSON") from exc

        if isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                headers = list(data[0].keys())
                rows = [list(item.values()) for item in data]
                return tabulate(rows, headers=headers)
            else:
                raise ValueError("List must contain dictionaries")

        if isinstance(data, dict):
            headers = ["Key", "Value"]
            rows = list(data.items())
            return tabulate(rows, headers=headers)

        raise TypeError("Input data must be a list of dictionaries, a dictionary, or a JSON string")

    table = __get_table(data)
    print(table)
