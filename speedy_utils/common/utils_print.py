# utils/utils_print.py

import copy
import json
import pprint
import re
import sys
import textwrap
from typing import Annotated, Any, Dict, List, Literal, Optional

from IPython.display import HTML, display
from loguru import logger
from tabulate import tabulate

from .utils_misc import is_notebook


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
    is_notebook=None,
    f=print,
) -> None | str:
    """
    Pretty print structured data.
    """
    if isinstance(input_data, list):
        for i, item in enumerate(input_data):
            fprint(item, key_ignore, key_keep, max_width, indent, depth, table_format, str_wrap_width, grep, is_notebook, f)
            print("\n" + "-" * 100 + "\n")

    from speedy_utils import is_notebook as is_interactive

    # is_notebook = is_notebook or is_interactive()
    if is_notebook is None:
        is_notebook = is_interactive()
    if hasattr(input_data, "toDict"):
        input_data = input_data.toDict()
    if hasattr(input_data, "to_dict"):
        input_data = input_data.to_dict()

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




def setup_logger(
    level: Annotated[
        Literal[
            "Trace",
            "Debug",
            "Info",
            "Success",
            "Warning",
            "Error",
            "Critical",
            "Disable",
            "T",
            "D",
            "I",
            "S",
            "W",
            "E",
            "C"
        ],
        "The desired log level"
    ] = "Info",
    enable_grep: Annotated[str, "Comma-separated patterns for enabling logs"] = "",
    disable_grep: Annotated[str, "Comma-separated patterns for disabling logs"] = "",
) -> None:
    # """
    # Setup the logger with the specified level and control logging based on grep patterns.

    # :param level: The desired log level.
    #               Valid levels: 'T' (TRACE), 'D' (DEBUG), 'I' (INFO), 'S' (SUCCESS),
    #               'W' (WARNING), 'E' (ERROR), 'C' (CRITICAL), 'DISABLE'.
    # :param enable_grep: A comma-separated string of patterns. Only logs matching these patterns will be enabled.
    # :param disable_grep: A comma-separated string of patterns. Logs matching these patterns will be disabled.
    # """

    # Map the shorthand level to the full name
    level_mapping = {
        "T": "TRACE",
        "D": "DEBUG",
        "I": "INFO",
        "S": "SUCCESS",
        "W": "WARNING",
        "E": "ERROR",
        "C": "CRITICAL",
    }

    # Set the level based on the input (or default to INFO)
    level = level_mapping.get(level.upper(), level.upper())

    # Remove any existing handlers to avoid duplication
    logger.remove()

    # Grep pattern handling
    enable_patterns = [pattern.strip() for pattern in enable_grep.split(",") if pattern.strip()]
    disable_patterns = [pattern.strip() for pattern in disable_grep.split(",") if pattern.strip()]

    def log_filter(record):
        """
        This filter applies the logging level and the grep pattern matching.
        Only logs with a level >= the specified level and that match the enable/disable patterns will be logged.
        """
        log_message = f"{record['file']}:{record['line']} ({record['function']})"

        # Check if the log should be enabled or disabled based on the grep patterns
        if enable_patterns and not any(re.search(pattern, log_message) for pattern in enable_patterns):
            return False  # If enable_grep is provided, log only if it matches
        if disable_patterns and any(re.search(pattern, log_message) for pattern in disable_patterns):
            return False  # If disable_grep matches, don't log

        # Return True if the log level is >= the set level
        return record["level"].no >= logger.level(level).no

    # Add the handler to stdout with the log format and the custom filter
    logger.add(
        sys.stdout,
        colorize=True,
        format=(
            "<level>{level: <8}</level> | <cyan>{file}:{line} ({function})</cyan> - "
            "<level>{message}</level>"
        ),
        filter=log_filter,
    )

    # Handle the "DISABLE" level: if set, disable all logging
    if level.upper() == "DISABLE":
        logger.disable("")  # Disable all logging
        logger.info("Logging disabled")
    else:
        logger.enable("")  # Ensure logging is enabled
        logger.debug(f"Logging set to {level}")
