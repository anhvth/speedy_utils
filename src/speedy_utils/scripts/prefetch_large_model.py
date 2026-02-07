#!/usr/bin/env python3
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

DEFAULT_EXTS = [".safetensors", ".json", ".model"]


def human_bytes(n: int) -> str:
    """Return a human-readable byte count."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    x = float(n)
    for unit in units:
        if x < 1024.0:
            return f"{x:.2f}{unit}"
        x /= 1024.0
    return f"{x:.2f}EB"


def iter_files(root: Path, exts: list[str]) -> Iterable[Path]:
    """Yield files under `root` whose names end with any extension in `exts`."""
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in exts):
                yield Path(dirpath) / filename


def scan(root: Path, exts: list[str]) -> tuple[list[Path], int]:
    files: list[Path] = []
    total = 0
    for path in iter_files(root, exts):
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        if st.st_size <= 0:
            continue
        files.append(path)
        total += st.st_size
    return files, total


def prefetch_file(path: Path, chunk_bytes: int) -> tuple[int, int]:
    """Read a file sequentially to populate the OS page cache.

    Returns: (bytes_read, errors)
    """
    read_bytes = 0
    try:
        buf = bytearray(chunk_bytes)
        with path.open("rb", buffering=0) as f:
            while True:
                n = f.readinto(buf)
                if not n:
                    break
                read_bytes += n
        return read_bytes, 0
    except Exception:
        return 0, 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prefetch model files into OS page cache (fast + parallel)."
    )
    ap.add_argument("model_path", help="Path to model directory")
    ap.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=8,
        help="Parallel workers (threads). Default: 8",
    )
    ap.add_argument(
        "--chunk-mb",
        type=int,
        default=16,
        help="Read chunk size in MB. Default: 16",
    )
    ap.add_argument(
        "--ext",
        action="append",
        default=[],
        help="File extension to include (repeatable). Default: .safetensors .json .model",
    )
    ap.add_argument("--no-progress", action="store_true", help="Disable progress output")
    args = ap.parse_args(argv)

    root = Path(args.model_path).expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 2

    exts = args.ext if args.ext else DEFAULT_EXTS
    chunk_bytes = max(1, args.chunk_mb) * 1024 * 1024
    jobs = max(1, args.jobs)

    print(f"\nPrefetching: {root}")
    print(f"Extensions: {', '.join(exts)}")
    print(f"Workers:    {jobs}")
    print(f"Chunk:      {args.chunk_mb} MB\n")

    files, total_bytes = scan(root, exts)
    if not files:
        print("No matching files found.")
        return 1

    print(f"Found {len(files)} files, total {human_bytes(total_bytes)}")

    start = time.time()
    last_report = start
    done_files = 0
    done_bytes = 0
    errors = 0
    error_paths: list[Path] = []

    try:
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(prefetch_file, path, chunk_bytes): path for path in files
            }
            for fut in as_completed(futures):
                path = futures[fut]
                bytes_read, err = fut.result()
                done_files += 1
                done_bytes += bytes_read
                errors += err
                if err:
                    error_paths.append(path)

                if args.no_progress:
                    continue

                now = time.time()
                if now - last_report < 0.5 and done_files != len(files):
                    continue
                last_report = now

                pct = (done_bytes / total_bytes * 100.0) if total_bytes else 100.0
                elapsed = now - start
                speed = int(done_bytes / elapsed) if elapsed > 0 else 0
                msg = (
                    f"\r[{done_files}/{len(files)}] "
                    f"{pct:5.1f}% "
                    f"{human_bytes(done_bytes)}/{human_bytes(total_bytes)} "
                    f"({human_bytes(speed)}/s) "
                    f"errors={errors}"
                )
                print(msg, end="", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    finally:
        if not args.no_progress:
            print()

    elapsed = time.time() - start
    avg_speed = int(done_bytes / elapsed) if elapsed > 0 else 0
    print(
        f"Done in {elapsed:.2f}s: read {human_bytes(done_bytes)} "
        f"({human_bytes(avg_speed)}/s), errors={errors}"
    )
    if error_paths:
        max_show = 20
        print(f"Failed files (showing up to {max_show}):", file=sys.stderr)
        for p in error_paths[:max_show]:
            print(f"  - {p}", file=sys.stderr)
        if len(error_paths) > max_show:
            print(f"  ... and {len(error_paths) - max_show} more", file=sys.stderr)
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
