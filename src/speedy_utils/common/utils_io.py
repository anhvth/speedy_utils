# utils/utils_io.py

import json
import os
import os.path as osp
import pickle
import time
from glob import glob
from pathlib import Path
from typing import Any, Union

from json_repair import loads as jloads
from pydantic import BaseModel

from .utils_misc import mkdir_or_exist


def dump_jsonl(list_dictionaries: list[dict], file_name: str = "output.jsonl") -> None:
    """
    Dumps a list of dictionaries to a file in JSON Lines format.
    """
    with open(file_name, "w", encoding="utf-8") as file:
        for dictionary in list_dictionaries:
            file.write(json.dumps(dictionary, ensure_ascii=False) + "\n")


def dump_json_or_pickle(
    obj: Any, fname: str, ensure_ascii: bool = False, indent: int = 4
) -> None:
    """
    Dump an object to a file, supporting both JSON and pickle formats.
    """
    if isinstance(fname, Path):
        fname = str(fname)
    mkdir_or_exist(osp.abspath(os.path.dirname(osp.abspath(fname))))
    if fname.endswith(".json"):
        with open(fname, "w", encoding="utf-8") as f:
            try:
                json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
            # TypeError: Object of type datetime is not JSON serializable
            except TypeError:
                print(
                    "Error: Object of type datetime is not JSON serializable",
                    str(obj)[:1000],
                )
                raise
    elif fname.endswith(".jsonl"):
        dump_jsonl(obj, fname)
    elif fname.endswith(".pkl"):
        try:
            with open(fname, "wb") as f:
                pickle.dump(obj, f)
        except Exception as e:
            if isinstance(obj, BaseModel):
                data = obj.model_dump()
                from fastcore.all import obj2dict, dict2obj
                obj2 = dict2obj(data)
                with open(fname, "wb") as f:
                    pickle.dump(obj2, f)
            else:
                raise ValueError(f"Error {e} while dumping {fname}") from e

    else:
        raise NotImplementedError(f"File type {fname} not supported")


def load_json_or_pickle(fname: str, counter=0) -> Any:
    """
    Load an object from a file, supporting both JSON and pickle formats.
    """
    if fname.endswith(".json") or fname.endswith(".jsonl"):
        with open(fname, encoding="utf-8") as f:
            return json.load(f)
    else:
        try:
            with open(fname, "rb") as f:
                return pickle.load(f)
        # EOFError: Ran out of input
        except EOFError:
            time.sleep(1)
            if counter > 5:
                print("Error: Ran out of input", fname)
                os.remove(fname)
                raise
            return load_json_or_pickle(fname, counter + 1)
        except Exception as e:
            raise ValueError(f"Error {e} while loading {fname}") from e


import os, io, json, gzip, bz2, lzma, warnings
from typing import Iterable, Union, IO, Any, Optional, cast

try:
    import orjson  # type: ignore[import-not-found]  # fastest JSON parser when available
except Exception:
    orjson = None

try:
    import zstandard as zstd  # type: ignore[import-not-found]  # optional .zst support
except Exception:
    zstd = None


def fast_load_jsonl(
    path_or_file: Union[str, os.PathLike, IO],
    *,
    progress: bool = False,
    desc: str = "Reading JSONL",
    use_orjson: bool = True,
    encoding: str = "utf-8",
    errors: str = "strict",
    on_error: str = "raise",   # 'raise' | 'warn' | 'skip'
    skip_empty: bool = True,
    max_lines: Optional[int] = None,
) -> Iterable[Any]:
    """
    Lazily iterate objects from a JSON Lines file.

    - Streams line-by-line (constant memory).
    - Optional tqdm progress over bytes (compressed size if gz/bz2/xz/zst).
    - Auto-detects compression by extension: .gz, .bz2, .xz/.lzma, .zst/.zstd.
    - Uses orjson if available (use_orjson=True), falls back to json.

    Args:
        path_or_file: Path-like or file-like object. File-like can be binary or text.
        progress: Show a tqdm progress bar (bytes). Requires `tqdm` if True.
        desc: tqdm description if progress=True.
        use_orjson: Prefer orjson for speed if installed.
        encoding, errors: Used when decoding text or when falling back to `json`.
        on_error: What to do on a malformed line: 'raise', 'warn', or 'skip'.
        skip_empty: Skip blank/whitespace-only lines.
        max_lines: Stop after reading this many lines (useful for sampling).

    Yields:
        Parsed Python objects per line.
    """
    def _open_auto(pth_or_f) -> IO[Any]:
        if hasattr(pth_or_f, "read"):
            # ensure binary buffer for consistent byte-length progress
            fobj = pth_or_f
            # If it's text, wrap it to binary via encoding; else just return
            if isinstance(fobj, io.TextIOBase):
                # TextIO -> re-encode to bytes on the fly
                return io.BufferedReader(io.BytesIO(fobj.read().encode(encoding, errors)))
            return pth_or_f  # assume binary
        s = str(pth_or_f).lower()
        if s.endswith(".gz"):
            return gzip.open(pth_or_f, "rb")
        if s.endswith(".bz2"):
            return bz2.open(pth_or_f, "rb")
        if s.endswith((".xz", ".lzma")):
            return lzma.open(pth_or_f, "rb")
        if s.endswith((".zst", ".zstd")) and zstd is not None:
            fh = open(pth_or_f, "rb")
            dctx = zstd.ZstdDecompressor()
            stream = dctx.stream_reader(fh)
            return io.BufferedReader(stream)
        # plain
        return open(pth_or_f, "rb", buffering=1024 * 1024)

    f = _open_auto(path_or_file)


    try:
        from tqdm import tqdm  # type: ignore
    except Exception as e:
        raise ImportError("tqdm is required when progress=True") from e
    total = None
    if not hasattr(path_or_file, "read"):
        try:
            path_for_size = cast(Union[str, os.PathLike], path_or_file)
            total = os.path.getsize(path_for_size)  # compressed size if any
        except Exception:
            total = None
    pbar = tqdm(total=total, unit="B", unit_scale=True, desc=desc)

    line_no = 0
    try:
        for raw_line in f:
            line_no += 1
            if pbar is not None:
                # raw_line is bytes here; if not, compute byte length
                nbytes = len(raw_line) if isinstance(raw_line, (bytes, bytearray)) else len(str(raw_line).encode(encoding, errors))
                pbar.update(nbytes)

            # Normalize to bytes -> str only if needed
            if isinstance(raw_line, (bytes, bytearray)):
                if skip_empty and not raw_line.strip():
                    if max_lines and line_no >= max_lines:
                        break
                    continue
                line_bytes = raw_line.rstrip(b"\r\n")
                # Parse
                try:
                    if use_orjson and orjson is not None:
                        obj = orjson.loads(line_bytes)
                    else:
                        obj = json.loads(line_bytes.decode(encoding, errors))
                except Exception as e:
                    if on_error == "raise":
                        raise
                    if on_error == "warn":
                        warnings.warn(f"Skipping malformed line {line_no}: {e}")
                    # 'skip' and 'warn' both skip the line
                    if max_lines and line_no >= max_lines:
                        break
                    continue
            else:
                # Text line path (unlikely)
                if skip_empty and not raw_line.strip():
                    if max_lines and line_no >= max_lines:
                        break
                    continue
                try:
                    obj = json.loads(raw_line)
                except Exception as e:
                    if on_error == "raise":
                        raise
                    if on_error == "warn":
                        warnings.warn(f"Skipping malformed line {line_no}: {e}")
                    if max_lines and line_no >= max_lines:
                        break
                    continue

            yield obj
            if max_lines and line_no >= max_lines:
                break
    finally:
        if pbar is not None:
            pbar.close()
        # Close only if we opened it (i.e., not an external stream)
        if not hasattr(path_or_file, "read"):
            try:
                f.close()
            except Exception:
                pass



def load_by_ext(fname: Union[str, list[str]], do_memoize: bool = False) -> Any:
    """
    Load data based on file extension.
    """
    if isinstance(fname, Path):
        fname = str(fname)
    from speedy_utils import multi_process

    from .utils_cache import (  # Adjust import based on your actual multi_worker module
        memoize,
    )

    try:
        if isinstance(fname, str) and "*" in fname:
            paths = glob(fname)
            paths = sorted(paths)
            return multi_process(load_by_ext, paths, workers=16)
        elif isinstance(fname, list):
            paths = fname
            return multi_process(load_by_ext, paths, workers=16)

        def load_csv(path: str, **pd_kwargs) -> Any:
            import pandas as pd

            return pd.read_csv(path, engine="pyarrow", **pd_kwargs)

        def load_txt(path: str) -> list[str]:
            with open(path, encoding="utf-8") as f:
                return f.read().splitlines()

        def load_default(path: str) -> Any:
            if path.endswith(".jsonl"):
                return list(fast_load_jsonl(path))
            elif path.endswith(".json"):
                try:
                    return load_json_or_pickle(path)
                except json.JSONDecodeError as exc:
                    raise ValueError("JSON decoding failed") from exc
            return load_json_or_pickle(path)

        handlers = {
            ".csv": load_csv,
            ".tsv": load_csv,
            ".txt": load_txt,
            ".pkl": load_default,
            ".json": load_default,
            ".jsonl": load_default,
        }

        ext = os.path.splitext(fname)[-1]
        load_fn = handlers.get(ext)

        if not load_fn:
            raise NotImplementedError(f"File type {ext} not supported")

        if do_memoize:
            load_fn = memoize(load_fn)

        return load_fn(fname)
    except Exception as e:
        raise ValueError(f"Error {e} while loading {fname}") from e


def jdumps(obj, ensure_ascii=False, indent=2, **kwargs):
    return json.dumps(obj, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


load_jsonl = lambda path: list(fast_load_jsonl(path))

__all__ = [
    "dump_json_or_pickle",
    "dump_jsonl",
    "load_by_ext",
    "load_json_or_pickle",
    "jdumps",
    "jloads",
]
