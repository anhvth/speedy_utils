"""Concise import-time helper.

Run this script to get a short list of top-level modules that take a
non-trivial amount of time to import. The script will try CPython's
``-X importtime`` and parse it into aggregated per-top-level times.

If the interpreter doesn't support ``-X importtime``, the script
falls back to a small instrumented subprocess that wraps
``builtins.__import__`` and reports per-top-level timings.

Default threshold: 0.2 seconds. Use --min-sec to change.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from typing import Dict, List, Tuple


DEFAULT_MODULES = ['speedy_utils', 'llm_utils', 'vision_utils']


def parse_x_importtime(stderr: str) -> List[Tuple[str, float]]:
    """Parse -X importtime stderr into per-top-level module seconds.

    We use the first column (self-time) aggregated per top-level module
    as a good approximation of which third-party or heavy packages cost
    time during import.
    """

    times: Dict[str, float] = {}
    pattern = re.compile(r'^\s*import time:\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.+)$')
    for line in stderr.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        try:
            self_us = int(match.group(1))
            mod_name = match.group(3).strip()
        except Exception:
            continue

        top = mod_name.split('.', 1)[0]
        times[top] = times.get(top, 0.0) + (self_us / 1_000_000.0)

    # return sorted list (desc)
    return sorted(times.items(), key=lambda it: it[1], reverse=True)


def run_importtime(module: str) -> Tuple[bool, str]:
    exe = sys.executable
    cmd = [exe, '-X', 'importtime', '-c', f'from {module} import *']
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    ok = p.returncode == 0 and bool(p.stderr.strip())
    out = p.stderr if p.stderr else p.stdout
    return ok, out


def run_timed_import(module: str) -> Tuple[bool, str]:
    code = (
        'import builtins, time, json\n'
        'orig = builtins.__import__\n'
        'times = {}\n'
        'def timed(name, globals=None, locals=None, fromlist=(), level=0):\n'
        '    start = time.perf_counter()\n'
        '    try:\n'
        '        return orig(name, globals, locals, fromlist, level)\n'
        '    finally:\n'
        '        elapsed = time.perf_counter() - start\n'
        "        key = name.split('.',1)[0]\n"
        '        times[key] = times.get(key, 0.0) + elapsed\n'
        'builtins.__import__ = timed\n'
        f'from {module} import *\n'
        'builtins.__import__ = orig\n'
        'print(json.dumps(sorted(times.items(), key=lambda it: it[1], reverse=True)))\n'
    )

    cmd = [sys.executable, '-c', code]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        return False, (p.stderr or p.stdout).strip()
    return True, p.stdout.strip()


def pretty_print_list(items: List[Tuple[str, float]]) -> None:
    for name, sec in items:
        print(f'{sec:6.3f}s  {name}')


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('modules', nargs='*', default=DEFAULT_MODULES)
    parser.add_argument(
        '--min-sec', type=float, default=0.2, help='Minimum seconds to show'
    )
    parser.add_argument('--no-x', action='store_true', help="Don't try -X importtime")
    parser.add_argument(
        '--raw', action='store_true', help='Show raw -X output in addition'
    )
    parser.add_argument('-n', '--top', type=int, default=20)
    args = parser.parse_args(argv)

    for module in args.modules:
        print('=' * 60)
        print(f'Module: {module}')
        print('=' * 60)

        if not args.no_x:
            ok, out = run_importtime(module)
            if ok:
                parsed = parse_x_importtime(out)
                filtered = [it for it in parsed if it[1] >= args.min_sec]
                if filtered:
                    print('Top heavy imports (from -X importtime):')
                    pretty_print_list(filtered[: args.top])
                else:
                    print(
                        f'No top-level modules >= {args.min_sec:.3f}s (from -X importtime)'
                    )
                if args.raw:
                    print('\nRaw -X importtime output:\n')
                    print(out)
                continue

        # Fallback instrumentation
        ok, out = run_timed_import(module)
        if not ok:
            print('Failed to measure imports:\n', out)
            continue

        items = json.loads(out)
        filtered = [it for it in items if it[1] >= args.min_sec]
        if not filtered:
            print(f'No imports >= {args.min_sec:.3f}s (fallback)')
            continue

        print('Top heavy imports (fallback):')
        pretty_print_list(filtered[: args.top])

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
