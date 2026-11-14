# Deprecations and removals

The rule for this codebase: public API is deprecated, never silently removed.
This page is the authoritative list of the public names the library-cleanup
refactor touched, and what to call instead. `docs/DELETED.md` has the internal
"why" for each deletion; this page is the migration view.

Everything below was checked against `HEAD` (package `__init__.py` files and
every module `__all__`) and against the current tree, then verified at runtime.

## Removed without a deprecation period

These went out without a deprecation period: each was undocumented, internal-only,
or had no caller outside its own module (checked across src, tests, docs, README,
notebooks and console scripts). Importing a removed module now raises
`ModuleNotFoundError`.

| Name | Module | What it was | Call instead |
|------|--------|-------------|--------------|
| `retry_runtime` | `speedy_utils` (was `common/function_decorator.py`) | Decorator retrying a call with exponential backoff on `RuntimeError` | Your own retry loop, or `ParallelLLMJob` |
| `xxhash` | `speedy_utils` | Re-export of the `xxhash` module | `hashlib` (stdlib) or `pip install xxhash` yourself |
| `speedy_timer` | `speedy_utils.clock` | Module-level `Clock(start_now=False)` shared by every importer | `Clock(start_now=False)` where you need it |
| `identify_uuid` | `speedy_utils.common.utils_cache` | `uuid.UUID` built from `xxh128` of the serialized object | `identify(obj)` — a 16-hex content digest, not a UUID |
| `get_arg_names` | `speedy_utils.common.utils_misc` | `inspect.getfullargspec(func).args` | `inspect.getfullargspec(func).args` |
| `handle_exceptions_with_clean_traceback` | `speedy_utils.common.utils_error` | Alias of `clean_traceback` | `clean_traceback` |
| `change_dir` | `speedy_utils.common.notebook_utils` | `os.chdir` to the first match of a path fragment | `os.chdir(path)` |
| `strip_assistant_end` | `llm_utils.lm.llm_qwen3` | Removed a trailing `<\|im_end\|>` from generated text | `split_assistant_parts` / `is_content_done` (still exported) |
| `CleanTracebackError` | `llm_utils._traceback` | llm_utils' own copy of the exception class | `speedy_utils.common.utils_error.CleanTracebackError` |
| `multi_thread_standard` | `speedy_utils.multi_worker.thread` | Older thread-pool runner kept for a test comparison | `multi_thread` |
| `visualize_tensor` | `vision_utils.plot` | Matplotlib grid of one tensor (`hwc`/`chw`) | `plot_images_notebook([array])` |
| `np`, `Image`, `parallel` | `speedy_utils.__imports` (star-import surface) | Re-exports of numpy, `PIL.Image`, `fastcore.parallel` | `import numpy as np`, `from PIL import Image`, `concurrent.futures` |
| `--format`, `--split` | `viz_chat` CLI | `--format` forced the chat format (`auto`/`chatml`/`sharegpt`); `--split` was accepted and never read | Drop the flag — `viz_chat FILE` auto-detects the format, and a `DatasetDict` directory already lists every split |

Deleted modules (nothing in any package `__init__`, no references anywhere):
`speedy_utils.common.patcher` (`patch_method`),
`speedy_utils.common.report_manager` (`ReportManager`),
`llm_utils.lm.base_prompt_builder` (`BasePromptBuilder`),
`llm_utils.system_prompted_assistant` (`WrongStudentConversationToAnswer`).

## Kept, behaviour changed

| Name | Before | Now |
|------|--------|-----|
| `CleanTracebackError` | Subclass of `Exception` | Subclass of `SpeedyExecutionError`; `except CleanTracebackError` still works, `except SpeedyExecutionError` catches it too. `multi_thread`'s `UserFunctionError` is now a `CleanTracebackError` as well |
| `clean_traceback` | Wrapped every exception it caught | `pydantic.ValidationError` and `openai.LengthFinishReasonError` propagate unchanged, so callers can catch them by type |
| `memoize(cache_type="disk")` and `"both"` | Any failure — including a failed cache write — logged a warning (only with `verbose=True`) and re-ran the function uncached | Only cache-key computation falls back to an uncached call. A cache load/write failure now raises, e.g. `ValueError: Error cannot pickle 'generator' object while dumping ...`. Use `cache_type="memory"` for results that cannot be serialized |
| `identify(obj)` | `identify(obj, depth=0, max_depth=2)`, digest `xxhash.xxh64_hexdigest` | `identify(obj)` only; digest is `hashlib.blake2b(..., digest_size=8).hexdigest()`. **Digest values differ from earlier releases** |
| Disk cache root | `/tmp/speedy_cache` | `~/.cache/speedy_cache` (`SPEED_CACHE_DIR`) |
| `memoize` keys for decorated functions | `get_source()` hashed the wrapper's source | `get_source()` calls `inspect.unwrap()` first, so the wrapped function's source is hashed |
| `multi_thread(workers=N)` | Importing set `SPEEDY_UTILS_MAX_WORKERS=2048`, so the limit was always 2048 | Limit is `min(2048, max(64, 16 * cpu_count))`, so the same `N` can now raise `ValueError: workers=N exceeds safe limit` on small machines. Override with `SPEEDY_UTILS_MAX_WORKERS` (`none`/`0` disables) |
| `load_json_or_pickle("x.jsonl")` | Tried `json.load` on the whole file and raised `JSONDecodeError` | Returns a `list` of records via `fast_load_jsonl` |
| `fast_load_jsonl` | Read every line into memory before chunking | Streams chunks from the open file; same results, lower memory |

Cold cache after upgrading: the cache root moved, `identify` changed digest, and
`get_source` now unwraps. Every disk entry written by an earlier release is
unreachable and is never cleaned up — delete `/tmp/speedy_cache` by hand.

## Still exported, not yet removed

Public names kept for compatibility even though nothing in this repo calls them.
They emit no `DeprecationWarning`; removal is not scheduled.

| Name | Exported from | Note |
|------|---------------|------|
| `convert_to_builtin_python` | `speedy_utils` | Listed in the README import example; only called by itself (recursion) |
| `LMBase` | `llm_utils` | Used inside `lm_base.py` to build a raw client; no other caller |
| `convert_to_arrow` | `datasets_utils` | Lazy export; only its own `__main__` block calls it |

All nine console scripts in `pyproject.toml` (`mpython`, `kill-mpython`,
`openapi_client_codegen`, `spu-prefetch-large-model`, `sp_chat`, `viz_chat`,
`pcat`, `pcat-jsonl`, `pcat-hf-dataset`) still resolve to existing modules.
