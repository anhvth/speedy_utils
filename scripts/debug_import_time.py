#!/usr/bin/env python3
"""Import-time diagnosis helper.

This script serves two use cases:

1. Developer diagnosis: show the slowest imports for one or more packages.
2. Import budget enforcement: fail with actionable diagnostics when a package
   exceeds a maximum total import time.

Examples
--------
    uv run python scripts/debug_import_time.py speedy_utils --no-stdlib

    uv run python scripts/debug_import_time.py \
        speedy_utils llm_utils vision_utils \
        --max-total-sec 0.4 \
        --min-sec 0.01 \
        --top 12 \
        --no-stdlib
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODULES = ["speedy_utils", "llm_utils", "vision_utils"]
IMPORTTIME_PATTERN = re.compile(
    r"^\s*import time:\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.+)$"
)


@dataclass(frozen=True)
class ImportTimeRow:
    module_name: str
    top_level_name: str
    self_sec: float
    cumulative_sec: float


@dataclass(frozen=True)
class TopLevelSummary:
    module_name: str
    self_sec: float
    max_cumulative_sec: float


def find_repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in (start, *start.parents):
        if (parent / "pyproject.toml").is_file() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


REPO_ROOT = find_repo_root()
SRC_DIR = REPO_ROOT / "src"


def pick_python_executable() -> str:
    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


PYTHON = pick_python_executable()


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    if SRC_DIR.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{SRC_DIR}{os.pathsep}{existing}" if existing else str(SRC_DIR)
        )
    return env


def build_import_statement(module: str, star_import: bool) -> str:
    if star_import:
        return f"from {module} import *"
    return f"import {module}"


def run_subprocess(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(REPO_ROOT),
        env=build_env(),
    )


def measure_total_import_time(module: str, star_import: bool) -> tuple[float | None, str]:
    import_stmt = build_import_statement(module, star_import)
    code = (
        "import time\n"
        "start = time.perf_counter()\n"
        f"{import_stmt}\n"
        "print(time.perf_counter() - start)\n"
    )
    result = run_subprocess([PYTHON, "-c", code])
    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip()

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None, "Missing import timing output"

    try:
        return float(lines[-1]), ""
    except ValueError:
        return None, (result.stderr or result.stdout).strip()


def run_importtime_trace(module: str, star_import: bool) -> subprocess.CompletedProcess[str]:
    import_stmt = build_import_statement(module, star_import)
    return run_subprocess([PYTHON, "-X", "importtime", "-c", import_stmt])


def parse_importtime_rows(output: str) -> list[ImportTimeRow]:
    rows: list[ImportTimeRow] = []
    for line in output.splitlines():
        match = IMPORTTIME_PATTERN.match(line)
        if not match:
            continue

        self_us = int(match.group(1))
        cumulative_us = int(match.group(2))
        module_name = match.group(3).strip()
        rows.append(
            ImportTimeRow(
                module_name=module_name,
                top_level_name=module_name.split(".", 1)[0],
                self_sec=self_us / 1_000_000.0,
                cumulative_sec=cumulative_us / 1_000_000.0,
            )
        )
    return rows


def summarize_top_levels(rows: list[ImportTimeRow]) -> list[TopLevelSummary]:
    totals: dict[str, TopLevelSummary] = {}
    for row in rows:
        previous = totals.get(row.top_level_name)
        if previous is None:
            totals[row.top_level_name] = TopLevelSummary(
                module_name=row.top_level_name,
                self_sec=row.self_sec,
                max_cumulative_sec=row.cumulative_sec,
            )
            continue

        totals[row.top_level_name] = TopLevelSummary(
            module_name=row.top_level_name,
            self_sec=previous.self_sec + row.self_sec,
            max_cumulative_sec=max(previous.max_cumulative_sec, row.cumulative_sec),
        )

    return sorted(
        totals.values(),
        key=lambda item: (item.self_sec, item.max_cumulative_sec),
        reverse=True,
    )


def is_stdlib(name: str) -> bool:
    if name == "builtins":
        return True
    stdlib_names = getattr(sys, "stdlib_module_names", None)
    if stdlib_names is None:
        return False
    return name in stdlib_names


def filter_rows(
    rows: list[ImportTimeRow],
    show_stdlib: bool | None,
) -> list[ImportTimeRow]:
    if show_stdlib is None:
        return rows
    return [row for row in rows if is_stdlib(row.top_level_name) is show_stdlib]


def filter_top_levels(
    rows: list[TopLevelSummary],
    show_stdlib: bool | None,
) -> list[TopLevelSummary]:
    if show_stdlib is None:
        return rows
    return [row for row in rows if is_stdlib(row.module_name) is show_stdlib]


def pick_slowest_top_levels(
    rows: list[TopLevelSummary],
    min_sec: float,
    top: int,
) -> list[TopLevelSummary]:
    filtered = [row for row in rows if row.self_sec >= min_sec]
    if filtered:
        return filtered[:top]
    return rows[:top]


def pick_slowest_rows(
    rows: list[ImportTimeRow],
    min_sec: float,
    top: int,
) -> list[ImportTimeRow]:
    ordered = sorted(
        rows,
        key=lambda row: (row.self_sec, row.cumulative_sec),
        reverse=True,
    )
    filtered = [row for row in ordered if row.self_sec >= min_sec]
    if filtered:
        return filtered[:top]
    return ordered[:top]


def extract_error_tail(output: str, max_lines: int = 4) -> str:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return "Unknown import error"
    return "\n".join(lines[-max_lines:])


def print_top_level_summary(rows: list[TopLevelSummary]) -> None:
    if not rows:
        print("Top-level contributors: no importtime rows captured")
        return

    print("Top-level contributors:")
    print("  self     max-cum  module")
    for row in rows:
        print(
            f"  {row.self_sec:7.3f}s  {row.max_cumulative_sec:7.3f}s  {row.module_name}"
        )


def print_module_summary(rows: list[ImportTimeRow]) -> None:
    if not rows:
        print("Slowest import paths: no importtime rows captured")
        return

    print("Slowest import paths:")
    print("  self     cum      module")
    for row in rows:
        print(
            f"  {row.self_sec:7.3f}s  {row.cumulative_sec:7.3f}s  {row.module_name}"
        )


def analyze_module(
    module: str,
    *,
    min_sec: float,
    max_total_sec: float | None,
    top: int,
    show_stdlib: bool | None,
    star_import: bool,
    raw: bool,
    quiet_success: bool,
    no_x: bool,
) -> bool:
    total_sec, import_error = measure_total_import_time(module, star_import)
    passed = import_error == ""
    if passed and max_total_sec is not None and total_sec is not None:
        passed = total_sec <= max_total_sec

    if quiet_success and passed and total_sec is not None:
        if max_total_sec is None:
            print(f"{module}: {total_sec:.3f}s")
        else:
            print(f"{module}: {total_sec:.3f}s / {max_total_sec:.3f}s")
        return True

    print("=" * 72)
    print(f"Module: {module}")
    print("=" * 72)

    if import_error:
        print("Status: IMPORT ERROR")
        print(extract_error_tail(import_error))
    elif total_sec is not None:
        budget = ""
        if max_total_sec is not None:
            budget = f" (budget {max_total_sec:.3f}s)"
        status = "PASS" if passed else "FAIL"
        print(f"Status: {status}")
        print(f"Total import time: {total_sec:.3f}s{budget}")

    trace_output = ""
    trace_rows: list[ImportTimeRow] = []
    if not no_x and (raw or not quiet_success or not passed or import_error):
        trace_result = run_importtime_trace(module, star_import)
        trace_output = trace_result.stderr or trace_result.stdout
        trace_rows = parse_importtime_rows(trace_output)

    if import_error and trace_output and not trace_rows:
        print("\nNo parsed importtime rows were captured before the import failed.")

    if trace_rows:
        top_level_rows = pick_slowest_top_levels(
            filter_top_levels(summarize_top_levels(trace_rows), show_stdlib),
            min_sec=min_sec,
            top=top,
        )
        module_rows = pick_slowest_rows(
            filter_rows(trace_rows, show_stdlib),
            min_sec=min_sec,
            top=top,
        )
        print()
        print_top_level_summary(top_level_rows)
        print()
        print_module_summary(module_rows)
    elif not no_x:
        print()
        print("No importtime breakdown was captured.")

    if raw and trace_output:
        print("\nRaw -X importtime output:\n")
        print(trace_output)

    if not passed:
        print()
        print("Suggested fix: move heavy third-party imports into function scope or")
        print("lazy __getattr__ exports so package import stays below budget.")

    return passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure module import time and show the slowest top-level and "
            "full import paths when the import is slow or broken."
        )
    )
    parser.add_argument("modules", nargs="*", default=DEFAULT_MODULES)
    parser.add_argument(
        "--min-sec",
        type=float,
        default=0.2,
        help="Minimum self time to show in the breakdown before falling back to top N.",
    )
    parser.add_argument(
        "--max-total-sec",
        type=float,
        default=None,
        help="Fail if total import time exceeds this budget.",
    )
    parser.add_argument(
        "--star-import",
        action="store_true",
        help='Use "from module import *" instead of plain import.',
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Show raw -X importtime output in addition to the summary.",
    )
    parser.add_argument(
        "--quiet-success",
        action="store_true",
        help="Print a single summary line when a module stays within budget.",
    )
    parser.add_argument(
        "--no-x",
        action="store_true",
        help="Skip -X importtime breakdown collection.",
    )
    parser.add_argument("-n", "--top", type=int, default=20)
    stdlib_group = parser.add_mutually_exclusive_group()
    stdlib_group.add_argument(
        "--stdlib",
        action="store_true",
        help="Show only standard library modules in the breakdown.",
    )
    stdlib_group.add_argument(
        "--no-stdlib",
        action="store_true",
        help="Exclude standard library modules from the breakdown.",
    )
    args = parser.parse_args(argv)
    show_stdlib = True if args.stdlib else False if args.no_stdlib else None

    overall_passed = True
    for module in args.modules:
        module_passed = analyze_module(
            module,
            min_sec=args.min_sec,
            max_total_sec=args.max_total_sec,
            top=args.top,
            show_stdlib=show_stdlib,
            star_import=args.star_import,
            raw=args.raw,
            quiet_success=args.quiet_success,
            no_x=args.no_x,
        )
        overall_passed = overall_passed and module_passed

    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
