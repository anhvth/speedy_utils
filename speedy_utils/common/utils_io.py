# utils/utils_io.py

import json
import os
import os.path as osp
from pathlib import Path
import pickle
from glob import glob
from typing import Any, List, Dict, Union

from .utils_misc import mkdir_or_exist


def dump_jsonl(list_dictionaries: List[Dict], file_name: str = "output.jsonl") -> None:
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
            json.dump(obj, f, ensure_ascii=ensure_ascii, indent=indent)
    elif fname.endswith(".jsonl"):
        dump_jsonl(obj, fname)
    elif fname.endswith(".pkl"):
        with open(fname, "wb") as f:
            pickle.dump(obj, f)
    else:
        raise NotImplementedError(f"File type {fname} not supported")


def load_json_or_pickle(fname: str) -> Any:
    """
    Load an object from a file, supporting both JSON and pickle formats.
    """
    if fname.endswith(".json") or fname.endswith(".jsonl"):
        with open(fname, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        with open(fname, "rb") as f:
            return pickle.load(f)

def load_jsonl(path):
    lines = open(path, "r", encoding="utf-8").read().splitlines()
    return [json.loads(line) for line in lines]

def load_by_ext(
    fname: Union[str, List[str]], do_memoize: bool = False
) -> Any:
    """
    Load data based on file extension.
    """
    if isinstance(fname, Path):
        fname = str(fname)
    from .utils_cache import (
        memoize,
    )  # Adjust import based on your actual multi_worker module

    from speedy_utils import multi_process  # Ensure multi_worker is correctly referenced
    

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

        def load_txt(path: str) -> List[str]:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().splitlines()

        def load_default(path: str) -> Any:
            if path.endswith(".jsonl"):
                return load_jsonl(path)
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
