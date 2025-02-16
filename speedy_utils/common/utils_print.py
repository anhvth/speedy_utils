# utils/utils_print.py

import copy
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


# A subclass of OrderedDict to automatically evict the oldest item after max_size is exceeded
class _RateLimitCache(OrderedDict):
    def __init__(self, max_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size

    def __setitem__(self, key, value):
        # If the key already exists, move it to the end (so it's considered "newer")
        if key in self:
            self.move_to_end(key)
        # Use normal __setitem__
        super().__setitem__(key, value)
        # Evict the oldest if we're over capacity
        if len(self) > self.max_size:
            self.popitem(last=False)  # pop the *first* item

# Create a global rate-limit cache with, say, 2,000 distinct entries max
_last_log_times = _RateLimitCache(max_size=2000)

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
    min_interval: float = 0.1,
    max_cache_entries: int = 2000,
) -> None:
    """
    Setup the logger with a rate-limiting feature:
    - No more than 1 log from the same file:line within `min_interval` seconds.
    - Track up to `max_cache_entries` distinct file:line pairs in memory.
    """
    # Update the cache size if desired
    _last_log_times.max_size = max_cache_entries

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
    level = level_mapping.get(level.upper(), level.upper())

    # Remove any existing handlers to avoid duplication
    logger.remove()

    # Prepare grep patterns
    enable_patterns = [p.strip() for p in enable_grep.split(",") if p.strip()]
    disable_patterns = [p.strip() for p in disable_grep.split(",") if p.strip()]

    def log_filter(record):
        """
        1. Filters out messages below the specified log level.
        2. Applies 'enable'/'disable' grep filters.
        3. Rate-limits same file:line messages if they occur within `min_interval` seconds.
        4. Enforces a max size on the (file:line) dictionary.
        """
        # ---------- 1) Log-level check ----------
        if record["level"].no < logger.level(level).no:
            return False

        # ---------- 2) Grep pattern handling ----------
        log_message = f"{record['file']}:{record['line']} ({record['function']})"
        if enable_patterns and not any(re.search(p, log_message) for p in enable_patterns):
            return False
        if disable_patterns and any(re.search(p, log_message) for p in disable_patterns):
            return False

        # ---------- 3) Rate limiting by file:line ----------
        file_line_key = f"{record['file']}:{record['line']}"
        now = time.time()

        last_time = _last_log_times.get(file_line_key)
        if last_time is not None and (now - last_time < min_interval):
            return False  # Skip logging within min_interval

        # Update the cache with new time (will also handle size eviction)
        _last_log_times[file_line_key] = now
        return True

    # Add the handler
    logger.add(
        sys.stdout,
        colorize=True,
        format=(
            "<level>{level: <8}</level> | "
            "<cyan>{file}:{line} ({function})</cyan> - <level>{message}</level>"
        ),
        filter=log_filter,
    )

    # ---------- 4) Handle "DISABLE" level ----------
    if level.upper() == "DISABLE":
        logger.disable("")
        logger.info("Logging disabled")
    else:
        logger.enable("")
        logger.debug(f"Logging set to {level}")