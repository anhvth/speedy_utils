from __future__ import annotations

import importlib
from typing import Any

# NOTE:
# Keep `import speedy_utils` fast (<0.4s) by deferring heavy imports (numpy, torch,
# ray, pandas, matplotlib, etc.) until attributes are actually accessed.
#
# This module preserves the previous "kitchen-sink" namespace behavior via:
# - explicit lazy exports for Speedy Utils APIs
# - a fallback to `speedy_utils.__imports` for convenience stdlib/typing/3rd-party names

__all__ = [
    # Standard library (from speedy_utils.__imports)
    "random",
    "copy",
    "functools",
    "gc",
    "inspect",
    "json",
    "multiprocessing",
    "os",
    "osp",
    "pickle",
    "pprint",
    "re",
    "sys",
    "textwrap",
    "threading",
    "time",
    "traceback",
    "uuid",
    "Counter",
    "ThreadPoolExecutor",
    "as_completed",
    "glob",
    "Pool",
    "Path",
    "Lock",
    "defaultdict",
    # Typing (from speedy_utils.__imports)
    "Any",
    "Awaitable",
    "Callable",
    "TypingCallable",
    "Dict",
    "Generic",
    "Iterable",
    "List",
    "Literal",
    "Mapping",
    "Optional",
    "Sequence",
    "Set",
    "Tuple",
    "Type",
    "TypeVar",
    "Union",
    # Third-party (from speedy_utils.__imports)
    "pd",
    "xxhash",
    "get_ipython",
    "HTML",
    "display",
    "logger",
    "BaseModel",
    "tabulate",
    "tqdm",
    "np",
    "matplotlib",
    "plt",
    # Clock module
    "Clock",
    "speedy_timer",
    "timef",
    # Function decorators
    "retry_runtime",
    # Cache utilities
    "memoize",
    "imemoize",
    "identify",
    "identify_uuid",
    # IO utilities
    "dump_json_or_pickle",
    "dump_jsonl",
    "load_by_ext",
    "load_json_or_pickle",
    "load_jsonl",
    "jdumps",
    "jloads",
    # Misc utilities
    "mkdir_or_exist",
    "flatten_list",
    "get_arg_names",
    "is_notebook",
    "convert_to_builtin_python",
    "dedup",
    # Print utilities
    "display_pretty_table_html",
    "flatten_dict",
    "fprint",
    "print_table",
    "setup_logger",
    "log",
    # Error handling utilities
    "clean_traceback",
    "handle_exceptions_with_clean_traceback",
    # Multi-worker processing
    "multi_process",
    "multi_thread",
    "kill_all_thread",
    "multi_process_dataset_ray",
    "multi_process_dataset",
    "WorkerResources",
    "report_progress",
    # Notebook utilities
    "change_dir",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # Clock
    "Clock": ("speedy_utils.common.clock", "Clock"),
    "speedy_timer": ("speedy_utils.common.clock", "speedy_timer"),
    "timef": ("speedy_utils.common.clock", "timef"),
    # Decorators / logging
    "retry_runtime": ("speedy_utils.common.function_decorator", "retry_runtime"),
    "log": ("speedy_utils.common.logger", "log"),
    "setup_logger": ("speedy_utils.common.logger", "setup_logger"),
    # Notebook helpers
    "change_dir": ("speedy_utils.common.notebook_utils", "change_dir"),
    "display_pretty_table_html": (
        "speedy_utils.common.notebook_utils",
        "display_pretty_table_html",
    ),
    "print_table": ("speedy_utils.common.notebook_utils", "print_table"),
    # Cache utils
    "identify": ("speedy_utils.common.utils_cache", "identify"),
    "identify_uuid": ("speedy_utils.common.utils_cache", "identify_uuid"),
    "imemoize": ("speedy_utils.common.utils_cache", "imemoize"),
    "memoize": ("speedy_utils.common.utils_cache", "memoize"),
    # IO utils
    "dump_json_or_pickle": ("speedy_utils.common.utils_io", "dump_json_or_pickle"),
    "dump_jsonl": ("speedy_utils.common.utils_io", "dump_jsonl"),
    "jdumps": ("speedy_utils.common.utils_io", "jdumps"),
    "jloads": ("speedy_utils.common.utils_io", "jloads"),
    "load_by_ext": ("speedy_utils.common.utils_io", "load_by_ext"),
    "load_json_or_pickle": ("speedy_utils.common.utils_io", "load_json_or_pickle"),
    "load_jsonl": ("speedy_utils.common.utils_io", "load_jsonl"),
    # Misc utils
    "convert_to_builtin_python": (
        "speedy_utils.common.utils_misc",
        "convert_to_builtin_python",
    ),
    "dedup": ("speedy_utils.common.utils_misc", "dedup"),
    "flatten_list": ("speedy_utils.common.utils_misc", "flatten_list"),
    "get_arg_names": ("speedy_utils.common.utils_misc", "get_arg_names"),
    "is_notebook": ("speedy_utils.common.utils_misc", "is_notebook"),
    "mkdir_or_exist": ("speedy_utils.common.utils_misc", "mkdir_or_exist"),
    # Print utils
    "flatten_dict": ("speedy_utils.common.utils_print", "flatten_dict"),
    "fprint": ("speedy_utils.common.utils_print", "fprint"),
    # Error utils
    "clean_traceback": ("speedy_utils.common.utils_error", "clean_traceback"),
    "handle_exceptions_with_clean_traceback": (
        "speedy_utils.common.utils_error",
        "handle_exceptions_with_clean_traceback",
    ),
    # Multi-worker
    "multi_process": ("speedy_utils.multi_worker.process", "multi_process"),
    "kill_all_thread": ("speedy_utils.multi_worker.thread", "kill_all_thread"),
    "multi_thread": ("speedy_utils.multi_worker.thread", "multi_thread"),
    "WorkerResources": ("speedy_utils.multi_worker.dataset_ray", "WorkerResources"),
    "multi_process_dataset_ray": (
        "speedy_utils.multi_worker.dataset_ray",
        "multi_process_dataset_ray",
    ),
    "multi_process_dataset": (
        "speedy_utils.multi_worker.dataset_sharding",
        "multi_process_dataset",
    ),
    "report_progress": ("speedy_utils.multi_worker.progress", "report_progress"),
}

_IMPORTS_MODULE = None


def _imports():
    global _IMPORTS_MODULE
    if _IMPORTS_MODULE is None:
        _IMPORTS_MODULE = importlib.import_module("speedy_utils.__imports")
    return _IMPORTS_MODULE


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is not None:
        module_name, attr_name = target
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value

    # Backward-compat convenience namespace:
    # defer to speedy_utils.__imports for stdlib/typing/third-party symbols.
    imports_mod = _imports()
    if hasattr(imports_mod, name):
        value = getattr(imports_mod, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Avoid importing speedy_utils.__imports on dir(); keep it cheap.
    return sorted({*globals().keys(), *_LAZY_ATTRS.keys(), *__all__})

