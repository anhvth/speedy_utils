# utils/utils_print.py

import copy
import inspect
import json
import pprint
import re
import sys
import textwrap
import time
from collections import OrderedDict
from typing import Annotated, Any, Dict, List, Literal, Optional

from IPython.display import HTML, display
from loguru import logger
from tabulate import tabulate

from .utils_misc import is_notebook


def display_pretty_table_html(data: dict) -> None:
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
    key_ignore: list[str] | None = None,
    key_keep: list[str] | None = None,
    max_width: int = 100,
    indent: int = 2,
    depth: int | None = None,
    table_format: str = "grid",
    str_wrap_width: int = 80,
    grep=None,
    is_notebook=None,
    f=print,
) -> None | str:
    """
    Pretty print structured data.
    """
    if isinstance(input_data, list):
        for i, item in enumerate(input_data):
            fprint(
                item,
                key_ignore,
                key_keep,
                max_width,
                indent,
                depth,
                table_format,
                str_wrap_width,
                grep,
                is_notebook,
                f,
            )
            print("\n" + "-" * 100 + "\n")

    from speedy_utils import is_notebook as is_interactive

    # is_notebook = is_notebook or is_interactive()
    if is_notebook is None:
        is_notebook = is_interactive()
    if isinstance(input_data, list):
        if all(hasattr(item, "toDict") for item in input_data):
            input_data = [item.toDict() for item in input_data]
    elif hasattr(input_data, "toDict"):
        input_data = input_data.toDict()

    if isinstance(input_data, list):
        if all(hasattr(item, "to_dict") for item in input_data):
            input_data = [item.to_dict() for item in input_data]
    elif hasattr(input_data, "to_dict"):
        input_data = input_data.to_dict()

    if isinstance(input_data, list):
        if all(hasattr(item, "model_dump") for item in input_data):
            input_data = [item.model_dump() for item in input_data]
    elif hasattr(input_data, "model_dump"):
        input_data = input_data.model_dump()
    if not isinstance(input_data, (dict, str)):
        raise ValueError("Input data must be a dictionary or string")

    if isinstance(input_data, dict):
        input_data = flatten_dict(input_data)

    if grep is not None and isinstance(input_data, dict):
        input_data = {k: v for k, v in input_data.items() if grep in str(k)}

    def remove_keys(d: dict, keys: list[str]) -> dict:
        """Remove specified keys from a dictionary."""
        for key in keys:
            parts = key.split(".")
            sub_dict = d
            for part in parts[:-1]:
                sub_dict = sub_dict.get(part, {})
            sub_dict.pop(parts[-1], None)
        return d

    def keep_keys(d: dict, keys: list[str]) -> dict:
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

    if hasattr(input_data, "to_dict") and not isinstance(input_data, str):
        input_data = input_data.to_dict()

    processed_data = copy.deepcopy(input_data)

    if isinstance(processed_data, dict) and is_notebook:
        if key_keep is not None:
            processed_data = keep_keys(processed_data, key_keep)
        elif key_ignore is not None:
            processed_data = remove_keys(processed_data, key_ignore)

        if is_notebook:
            display_pretty_table_html(processed_data)
            return

    if isinstance(processed_data, dict):
        table = [[k, v] for k, v in processed_data.items()]
        f(
            tabulate(
                table,
                headers=["Key", "Value"],
                tablefmt=table_format,
                maxcolwidths=[None, max_width],
            )
        )
    elif isinstance(processed_data, str):
        wrapped_text = textwrap.fill(processed_data, width=str_wrap_width)
        f(wrapped_text)
    elif isinstance(processed_data, list):
        f(tabulate(processed_data, tablefmt=table_format))
    else:
        printer = pprint.PrettyPrinter(width=max_width, indent=indent, depth=depth)
        printer.pprint(processed_data)


def print_table(data: Any, use_html: bool = True) -> None:
    """
    Print data as a table. If use_html is True, display using IPython HTML.
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
                return tabulate(
                    rows, headers=headers, tablefmt="html" if use_html else "grid"
                )
            else:
                raise ValueError("List must contain dictionaries")

        if isinstance(data, dict):
            headers = ["Key", "Value"]
            rows = list(data.items())
            return tabulate(
                rows, headers=headers, tablefmt="html" if use_html else "grid"
            )

        raise TypeError(
            "Input data must be a list of dictionaries, a dictionary, or a JSON string"
        )

    table = __get_table(data)
    if use_html:
        display(HTML(table))
    else:
        print(table)


__all__ = [
    "display_pretty_table_html",
    "flatten_dict",
    "fprint",
    "print_table",
    # "setup_logger",
    # "log",
]
