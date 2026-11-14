from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._shared import (
    _run_glob_serve,
    build_common_parser,
    main_hf_dataset,
    main_jsonl,
)


_SUBCOMMANDS = {"jsonl", "hf-dataset"}


def _looks_like_hf_dataset(path: Path) -> bool:
    """Check if a directory looks like a HuggingFace dataset."""
    markers = ("dataset_info.json", "dataset_dict.json")
    return path.is_dir() and any((path / marker).exists() for marker in markers)


def _build_auto_parser() -> argparse.ArgumentParser:
    parser = build_common_parser(
        prog="pcat",
        description="Interactive JSON viewer for JSONL files and HuggingFace datasets.",
        path_help="path to a .jsonl file or HF dataset directory",
    )
    parser.add_argument(
        "--split",
        help="dataset split (HF DatasetDict only; default: train)",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="**/*.jsonl",
        help="glob pattern for folder-based --serve (default: **/*.jsonl)",
    )
    return parser


def _detect_subcommand(argv: list[str]) -> tuple[str | None, list[str]]:
    """Return (mode, remaining_args) if a subcommand is found, else (None, argv)."""
    for i, token in enumerate(argv):
        if token.startswith("-"):
            continue
        if token in _SUBCOMMANDS:
            return token, argv[:i] + argv[i + 1 :]
        return None, argv
    return None, argv


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _build_auto_parser().print_help()
        return 1

    mode, remaining = _detect_subcommand(argv)
    if mode == "jsonl":
        return main_jsonl(remaining)
    if mode == "hf-dataset":
        return main_hf_dataset(remaining)

    # Auto-detect mode (no subcommand)
    parser = _build_auto_parser()
    args = parser.parse_args(argv)

    if args.path is None:
        parser.print_help()
        return 1

    if args.serve and args.field is not None:
        parser.error("--field is supported in terminal and --plain modes only")

    path = args.path.expanduser()
    if not path.exists():
        parser.error(f"path not found: {path}")

    if _looks_like_hf_dataset(path):
        return main_hf_dataset(_forward_args(args, path, for_hf=True))

    # Folder + --serve: glob source with the file picker
    if path.is_dir() and args.serve:
        return _run_glob_serve(parser, path, args)

    return main_jsonl(_forward_args(args, path, for_hf=False))


def _forward_args(args: argparse.Namespace, path: Path, *, for_hf: bool) -> list[str]:
    """Re-emit auto-parser args as argv for main_hf_dataset/main_jsonl."""
    argv: list[str] = []
    if for_hf and args.split:
        argv += ["--split", args.split]
    if args.index is not None:
        argv += ["--index", str(args.index)]
    if args.plain:
        argv.append("--plain")
    if args.sample is not None:
        argv += ["--sample", str(args.sample)]
    if args.serve:
        argv.append("--serve")
    if args.port != 8888:
        argv += ["--port", str(args.port)]
    if args.host != "127.0.0.1":
        argv += ["--host", args.host]
    if args.mode != "auto":
        argv += ["--mode", args.mode]
    if args.field is not None:
        argv += ["--field", args.field]
    if args.tokenizer is not None:
        argv += ["--tokenizer", args.tokenizer]
    if args.no_browser:
        argv.append("--no-browser")
    if not for_hf and args.ext != "**/*.jsonl":
        argv += ["--ext", args.ext]
    argv.append(str(path))
    return argv


if __name__ == "__main__":
    raise SystemExit(main())
