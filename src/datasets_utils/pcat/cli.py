from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ._shared import JsonlGlobRowSource, main_hf_dataset, main_jsonl


_SUBCOMMANDS = {"jsonl", "hf-dataset"}


def _looks_like_hf_dataset(path: Path) -> bool:
    """Check if a directory looks like a HuggingFace dataset."""
    return path.is_dir() and (path / "dataset_info.json").exists()


def _build_subcommand_parser(mode: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pcat " + mode)
    parser.add_argument(
        "path",
        type=Path,
        help=(
            "path to a .jsonl file or a directory of .json files"
            if mode == "jsonl"
            else "path to a dataset saved via datasets.load_from_disk()"
        ),
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=None,
        help="zero-based row index (negative counts from end; default = last row)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="print pretty JSON of selected row and exit",
    )
    parser.add_argument(
        "-s",
        "--sample",
        nargs="?",
        const=1,
        type=int,
        metavar="N",
        default=None,
        help="pre-draw N random rows (default 1); press s in TUI to step through",
    )
    if mode == "hf-dataset":
        parser.add_argument(
            "--split",
            help="dataset split to open when load_from_disk() returns a DatasetDict",
        )
    return parser


def _build_auto_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pcat",
        description="Interactive JSON viewer for JSONL files and HuggingFace datasets.",
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        help="path to a .jsonl file or HF dataset directory",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=None,
        help="zero-based row index (negative counts from end; default = last row)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="print pretty JSON of selected row and exit",
    )
    parser.add_argument(
        "-s",
        "--sample",
        nargs="?",
        const=1,
        type=int,
        metavar="N",
        default=None,
        help="pre-draw N random rows (default 1); press s in TUI to step through",
    )
    parser.add_argument(
        "--split", help="dataset split (auto-detect mode, HF datasets only)"
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="start a local web server to browse rows in the browser",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="port for --serve (default: 8888)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="bind address for --serve (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        help="render mode for --serve: auto, generic, raw, sdd (default: auto)",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3.5-27B",
        help="tokenizer name or path for tokenized SDD rows",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open browser when --serve starts",
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


def _serve_glob_dir(path: Path, args: argparse.Namespace) -> int:
    """Serve a folder by globbing for JSONL files and showing a file picker."""
    from .serve import serve as _serve

    try:
        glob_source = JsonlGlobRowSource.from_path(path, pattern=args.ext)
    except ValueError as exc:
        _build_auto_parser().error(str(exc))
        return 2

    print(
        f"  Found {len(glob_source.files)} files in {path}/{args.ext}", file=sys.stderr
    )

    # Pre-select first file as the initial source (the handler shows picker anyway)
    from ._shared import JsonlRowSource

    initial = JsonlRowSource.from_path(glob_source.files[0])

    return _serve(
        initial,
        host=args.host,
        port=args.port,
        mode=args.mode,
        tokenizer_name=args.tokenizer,
        open_browser=not args.no_browser,
        glob_source=glob_source,
    )


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        _build_auto_parser().print_help()
        return 1

    mode, remaining = _detect_subcommand(argv)

    if mode:
        sub_parser = _build_subcommand_parser(mode)
        args = sub_parser.parse_args(remaining)
        path = args.path.expanduser()
        sub_argv = []
        if args.index is not None:
            sub_argv += ["--index", str(args.index)]
        if args.plain:
            sub_argv.append("--plain")
        if args.sample is not None:
            sub_argv += ["--sample", str(args.sample)]
        if mode == "hf-dataset" and args.split:
            sub_argv += ["--split", args.split]
        sub_argv.append(str(path))
        if mode == "jsonl":
            return main_jsonl(sub_argv)
        return main_hf_dataset(sub_argv)

    # Auto-detect mode (no subcommand)
    auto_parser = _build_auto_parser()
    args = auto_parser.parse_args(argv)

    if args.path is None:
        auto_parser.print_help()
        return 1

    path = args.path.expanduser()
    if not path.exists():
        auto_parser.error(f"path not found: {path}")

    if _looks_like_hf_dataset(path):
        sub_argv = ["--split", args.split] if args.split else []
        if args.index is not None:
            sub_argv += ["--index", str(args.index)]
        if args.plain:
            sub_argv.append("--plain")
        if args.sample is not None:
            sub_argv += ["--sample", str(args.sample)]
        if args.serve:
            sub_argv.append("--serve")
        if args.port != 8888:
            sub_argv += ["--port", str(args.port)]
        if args.host != "127.0.0.1":
            sub_argv += ["--host", args.host]
        if args.mode != "auto":
            sub_argv += ["--mode", args.mode]
        if args.tokenizer != "Qwen/Qwen3.5-27B":
            sub_argv += ["--tokenizer", args.tokenizer]
        if args.no_browser:
            sub_argv.append("--no-browser")
        if args.ext != "**/*.jsonl":
            sub_argv += ["--ext", args.ext]
        sub_argv.append(str(path))
        return main_hf_dataset(sub_argv)

    # Folder + --serve: use glob source with file picker
    if path.is_dir() and args.serve:
        return _serve_glob_dir(path, args)

    sub_argv = []
    if args.index is not None:
        sub_argv += ["--index", str(args.index)]
    if args.plain:
        sub_argv.append("--plain")
    if args.sample is not None:
        sub_argv += ["--sample", str(args.sample)]
    if args.serve:
        sub_argv.append("--serve")
    if args.port != 8888:
        sub_argv += ["--port", str(args.port)]
    if args.host != "127.0.0.1":
        sub_argv += ["--host", args.host]
    if args.mode != "auto":
        sub_argv += ["--mode", args.mode]
    if args.tokenizer != "Qwen/Qwen3.5-27B":
        sub_argv += ["--tokenizer", args.tokenizer]
    if args.no_browser:
        sub_argv.append("--no-browser")
    if args.ext != "**/*.jsonl":
        sub_argv += ["--ext", args.ext]
    sub_argv.append(str(path))
    return main_jsonl(sub_argv)


if __name__ == "__main__":
    raise SystemExit(main())
