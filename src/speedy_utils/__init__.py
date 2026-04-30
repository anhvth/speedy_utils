# Third-party (fast)
import xxhash
from loguru import logger
from tqdm import tqdm


tabulate = __import__("tabulate").tabulate


# Clock
from speedy_utils.common.clock import Clock, speedy_timer, timef

# Error handling utilities
from speedy_utils.common.exceptions import (
    SpeedyExecutionError,
    SpeedySerializationError,
    SpeedyWorkerError,
)

# Function decorators / logging
from speedy_utils.common.function_decorator import retry_runtime
from speedy_utils.common.logger import log, setup_logger

# Notebook helpers
from speedy_utils.common.notebook_utils import (
    change_dir,
    display_pretty_table_html,
    print_table,
)

# Cache utilities
from speedy_utils.common.utils_cache import identify, identify_uuid, imemoize, memoize
from speedy_utils.common.utils_error import (
    clean_traceback,
    handle_exceptions_with_clean_traceback,
)

# IO utilities
from speedy_utils.common.utils_io import (
    dump_json_or_pickle,
    dump_jsonl,
    fast_load_jsonl,
    jdumps,
    jloads,
    load_by_ext,
    load_json_or_pickle,
    load_jsonl,
)

# Misc utilities
from speedy_utils.common.utils_misc import (
    convert_to_builtin_python,
    dedup,
    flatten_list,
    get_arg_names,
    is_notebook,
    mkdir_or_exist,
)

# Print utilities
from speedy_utils.common.utils_print import flatten_dict, fprint

# Multi-worker processing
from speedy_utils.multi_worker.dataset_sharding import multi_process_dataset
from speedy_utils.multi_worker.process import multi_process
from speedy_utils.multi_worker.thread import kill_all_thread, multi_thread


__all__ = [
    # Third-party
    "xxhash",
    "logger",
    "tabulate",
    "tqdm",
    # Clock
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
    "fast_load_jsonl",
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
    "SpeedyExecutionError",
    "SpeedySerializationError",
    "SpeedyWorkerError",
    "clean_traceback",
    "handle_exceptions_with_clean_traceback",
    # Multi-worker processing
    "multi_process",
    "multi_thread",
    "kill_all_thread",
    "multi_process_dataset",
    # Notebook utilities
    "change_dir",
]
