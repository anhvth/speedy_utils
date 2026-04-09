#!/usr/bin/env python3
"""Run pyright type-checking and report errors in a VS Code-compatible format.

This script mirrors what Pylance (the VS Code Python language server) shows as
red/error diagnostics.  It reads ``pyrightconfig.json`` at the repo root, adds
*extra* paths (e.g. ``tests/``) when requested, and emits coloured output to
the terminal.

Usage
-----
    # Check only what pyrightconfig.json includes (default: src/)
    uv run python tools/check_syntax.py

    # Also check tests/
    uv run python tools/check_syntax.py --include tests

    # Check arbitrary paths
    uv run python tools/check_syntax.py --include tests src/llm_utils

    # Show only errors from a specific file
    uv run python tools/check_syntax.py --file src/llm_utils/lm/llm_qwen3.py

    # JSON output (for programmatic consumption)
    uv run python tools/check_syntax.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYRIGHT_CONFIG = REPO_ROOT / "pyrightconfig.json"

# ANSI colour helpers
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _run_pyright(extra_includes: list[str] | None = None) -> dict:
    """Invoke ``pyright --outputjson`` and return the parsed result."""
    cmd = ["uv", "run", "pyright", "--outputjson"]
    if extra_includes:
        for inc in extra_includes:
            cmd.extend(["-p", str(REPO_ROOT)])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"{RED}ERROR: pyright produced invalid JSON{RESET}", file=sys.stderr)
        print(result.stdout[:2000], file=sys.stderr)
        print(result.stderr[:2000], file=sys.stderr)
        sys.exit(2)


def _format_diagnostic(diag: dict, repo_root: str) -> str:
    """Format a single diagnostic into a human-readable coloured line."""
    filepath = diag.get("file", "?")
    if filepath.startswith(repo_root):
        filepath = filepath[len(repo_root) :].lstrip("/")

    rng = diag.get("range", {})
    start_line = rng.get("start", {}).get("line", 0) + 1
    severity = diag.get("severity", "error")
    rule = diag.get("rule", "")
    message = diag.get("message", "").split("\n")[0]  # first line only

    if severity == "error":
        sev_colour = RED
        sev_label = "error"
    elif severity == "warning":
        sev_colour = YELLOW
        sev_label = "warning"
    else:
        sev_colour = DIM
        sev_label = "info"

    rule_str = f" [{rule}]" if rule else ""
    return (
        f"  {CYAN}{filepath}{RESET}:{YELLOW}{start_line}{RESET}"
        f" {sev_colour}{sev_label}{rule_str}{RESET}: {message}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run pyright and report type errors (mirrors VS Code Pylance).",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Extra directories to include beyond pyrightconfig.json",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Filter output to a specific file (relative or absolute path)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output raw pyright JSON instead of formatted text",
    )
    args = parser.parse_args()

    data = _run_pyright(extra_includes=args.include)

    if args.json_output:
        json.dump(data, sys.stdout, indent=2)
        print()
        return data.get("summary", {}).get("errorCount", 1)

    repo_root = str(REPO_ROOT)
    diagnostics = data.get("generalDiagnostics", [])

    # Filter by file if requested
    if args.file:
        target = str((REPO_ROOT / args.file).resolve())
        diagnostics = [d for d in diagnostics if d.get("file", "") == target]

    # Separate by severity
    errors = [d for d in diagnostics if d["severity"] == "error"]
    warnings = [d for d in diagnostics if d["severity"] == "warning"]
    infos = [d for d in diagnostics if d["severity"] == "information"]

    summary = data.get("summary", {})
    total_errors = summary.get("errorCount", len(errors))
    total_warnings = summary.get("warningCount", len(warnings))

    if not errors and not warnings:
        print(f"\n{BOLD}  ✓ No pyright errors or warnings{RESET}\n")
        return 0

    # Group errors by file
    by_file: dict[str, list[dict]] = {}
    for d in errors + warnings:
        f = d.get("file", "?")
        if f.startswith(repo_root):
            f = f[len(repo_root) :].lstrip("/")
        by_file.setdefault(f, []).append(d)

    print(f"\n{BOLD}pyright type check results{RESET}")
    print(f"{'─' * 60}")

    for filepath, diags in sorted(by_file.items()):
        file_errors = sum(1 for d in diags if d["severity"] == "error")
        file_warnings = sum(1 for d in diags if d["severity"] == "warning")
        counts = []
        if file_errors:
            counts.append(f"{RED}{file_errors} error{'s' if file_errors > 1 else ''}{RESET}")
        if file_warnings:
            counts.append(f"{YELLOW}{file_warnings} warning{'s' if file_warnings > 1 else ''}{RESET}")
        print(f"\n{BOLD}{filepath}{RESET}  ({', '.join(counts)})")
        for d in sorted(diags, key=lambda x: x.get("range", {}).get("start", {}).get("line", 0)):
            print(_format_diagnostic(d, repo_root))

    print(f"\n{'─' * 60}")
    parts = []
    if total_errors:
        parts.append(f"{RED}{total_errors} error{'s' if total_errors > 1 else ''}{RESET}")
    if total_warnings:
        parts.append(f"{YELLOW}{total_warnings} warning{'s' if total_warnings > 1 else ''}{RESET}")
    print(f"{BOLD}Total:{RESET} {', '.join(parts)}")
    print()

    return 1 if total_errors else 0


if __name__ == "__main__":
    sys.exit(main())
