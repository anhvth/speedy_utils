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
from typing import Dict, List, Optional, Tuple


DEFAULT_MODULES = ['speedy_utils', 'llm_utils', 'vision_utils']


def parse_x_importtime(stderr: str) -> List[Tuple[str, float, float]]:
    """Parse -X importtime stderr into per-top-level module seconds.

    We aggregate both self-time and cumulative-time per top-level module
    as a quick proxy for heavy imports.
    """

    times: Dict[str, Tuple[float, float]] = {}
    pattern = re.compile(r'^\s*import time:\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.+)$')
    for line in stderr.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        try:
            self_us = int(match.group(1))
            cum_us = int(match.group(2))
            mod_name = match.group(3).strip()
        except Exception:
            continue

        top = mod_name.split('.', 1)[0]
        self_sec = self_us / 1_000_000.0
        cum_sec = cum_us / 1_000_000.0
        prev_self, prev_cum = times.get(top, (0.0, 0.0))
        times[top] = (prev_self + self_sec, prev_cum + cum_sec)

    # return sorted list (desc)
    return sorted(
        [(name, vals[0], vals[1]) for name, vals in times.items()],
        key=lambda it: it[1],
        reverse=True,
    )


def run_importtime(module: str, star_import: bool) -> Tuple[bool, str]:
    exe = sys.executable
    if star_import:
        code = f'from {module} import *'
    else:
        code = f'import {module}'
    cmd = [exe, '-X', 'importtime', '-c', code]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    ok = p.returncode == 0 and bool(p.stderr.strip())
    out = p.stderr if p.stderr else p.stdout
    return ok, out


def run_timed_import(module: str, star_import: bool) -> Tuple[bool, str]:
    if star_import:
        import_stmt = f'from {module} import *'
    else:
        import_stmt = f'import {module}'
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
        f'{import_stmt}\n'
        'builtins.__import__ = orig\n'
        'print(json.dumps(sorted(times.items(), key=lambda it: it[1], reverse=True)))\n'
    )

    cmd = [sys.executable, '-c', code]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if p.returncode != 0:
        return False, (p.stderr or p.stdout).strip()
    return True, p.stdout.strip()


def pretty_print_list(items: List[Tuple[str, float, Optional[float]]]) -> None:
    for name, self_sec, cum_sec in items:
        if cum_sec is None:
            print(f'{self_sec:6.3f}s  {name}')
        else:
            print(f'{self_sec:6.3f}s  {cum_sec:6.3f}s  {name}')


def is_stdlib(name: str) -> bool:
    if name == 'builtins':
        return True
    stdlib_names = getattr(sys, 'stdlib_module_names', None)
    if stdlib_names is None:
        return False
    return name in stdlib_names


def filter_stdlib(
    items: List[Tuple[str, float, Optional[float]]],
    show_stdlib: Optional[bool],
) -> List[Tuple[str, float, Optional[float]]]:
    if show_stdlib is None:
        return items
    return [it for it in items if is_stdlib(it[0]) is show_stdlib]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('modules', nargs='*', default=DEFAULT_MODULES)
    parser.add_argument(
        '--min-sec', type=float, default=0.2, help='Minimum seconds to show'
    )
    parser.add_argument('--no-x', action='store_true', help="Don't try -X importtime")
    parser.add_argument(
        '--star-import',
        action='store_true',
        help='Use "from module import *" instead of plain import',
    )
    parser.add_argument(
        '--raw', action='store_true', help='Show raw -X output in addition'
    )
    parser.add_argument('-n', '--top', type=int, default=20)
    stdlib_group = parser.add_mutually_exclusive_group()
    stdlib_group.add_argument(
        '--stdlib',
        action='store_true',
        help='Show only standard library modules',
    )
    stdlib_group.add_argument(
        '--no-stdlib',
        action='store_true',
        help='Exclude standard library modules',
    )
    args = parser.parse_args(argv)
    show_stdlib = True if args.stdlib else False if args.no_stdlib else None

    for module in args.modules:
        print('=' * 60)
        print(f'Module: {module}')
        print('=' * 60)

        if not args.no_x:
            ok, out = run_importtime(module, args.star_import)
            if ok:
                parsed = parse_x_importtime(out)
                filtered = [it for it in parsed if it[1] >= args.min_sec]
                filtered = filter_stdlib(
                    [(n, s, c) for n, s, c in filtered], show_stdlib
                )
                if filtered:
                    print('Top heavy imports (from -X importtime):')
                    print(' self    cum    module')
                    pretty_print_list(filtered[: args.top])
                else:
                    print(
                        f'No top-level modules >= {args.min_sec:.3f}s (from -X importtime)'
                    )
                if args.raw:
                    print('\nRaw -X importtime output:\n')
                    print(out)
                continue
            print(
                f'Failed to measure imports for {module!r} with -X importtime. '
                'Retry with --raw for details.'
            )

        # Fallback instrumentation
        ok, out = run_timed_import(module, args.star_import)
        if not ok:
            print(f'Failed to measure imports for {module!r}:\n', out)
            continue

        items = json.loads(out)
        filtered = [it for it in items if it[1] >= args.min_sec]
        filtered = filter_stdlib([(n, s, None) for n, s in filtered], show_stdlib)
        if not filtered:
            print(f'No imports >= {args.min_sec:.3f}s (fallback)')
            continue

        print('Top heavy imports (fallback):')
        pretty_print_list(filtered[: args.top])

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
