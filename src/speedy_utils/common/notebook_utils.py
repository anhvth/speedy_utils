# jupyter notebook utilities
import json
import os
import pathlib
from typing import Any

from IPython.display import HTML, display
from tabulate import tabulate


def change_dir(target_directory: str = 'POLY') -> None:
    """Change directory to the first occurrence of x in the current path."""
    cur_dir = pathlib.Path('./')
    target_dir = str(cur_dir.absolute()).split(target_directory)[0] + target_directory
    os.chdir(target_dir)
    print(f'Current dir: {target_dir}')


def display_pretty_table_html(data: dict) -> None:
    """Display a pretty HTML table in Jupyter notebooks."""
    table = "<table>"
    for key, value in data.items():
        table += f"<tr><td>{key}</td><td>{value}</td></tr>"
    table += "</table>"
    display(HTML(table))


def print_table(data: Any, use_html: bool = True) -> None:
    """Print data as a table. If use_html is True, display using IPython HTML."""

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