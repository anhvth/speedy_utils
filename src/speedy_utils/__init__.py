# Third-party (fast)
from loguru import logger
from tqdm import tqdm


tabulate = __import__("tabulate").tabulate


# Clock
from speedy_utils.common.clock import Clock, timef

# Error handling utilities
from speedy_utils.common.exceptions import (
    SpeedyExecutionError,
    SpeedySerializationError,
    SpeedyWorkerError,
)

# Logging
from speedy_utils.common.logger import log, setup_logger
from speedy_utils.common.mirror_path import MirrowPath

# Notebook helpers
from speedy_utils.common.notebook_utils import (
    display_pretty_table_html,
    print_table,
)

# Cache utilities
from speedy_utils.common.utils_cache import identify, imemoize, memoize
from speedy_utils.common.utils_error import clean_traceback

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
    is_notebook,
    mkdir_or_exist,
)

# Print utilities
from speedy_utils.common.utils_print import flatten_dict, fprint

# Multi-worker processing
from speedy_utils.multi_worker.dataset_sharding import multi_process_dataset
from speedy_utils.multi_worker.process import multi_process, multi_process_ray
from speedy_utils.multi_worker.thread import kill_all_thread, multi_thread
from speedy_utils.parallel_llm_job import (
    JobSummary,
    ParallelLLMJob,
    TargetNotReachedError,
)


__all__ = [
    # Third-party
    "logger",
    "tabulate",
    "tqdm",
    # Clock
    "Clock",
    "timef",
    # Cache utilities
    "memoize",
    "imemoize",
    "identify",
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
    # Multi-worker processing
    "multi_process",
    "multi_process_ray",
    "multi_thread",
    "kill_all_thread",
    "multi_process_dataset",
    # Notebook utilities
    "MirrowPath",
    "JobSummary",
    "ParallelLLMJob",
    "TargetNotReachedError",
]
