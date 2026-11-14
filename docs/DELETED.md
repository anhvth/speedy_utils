# What this refactor deleted

Things that were removed on purpose. Do not re-add them; the replacement is in
the right-hand column.

## Code

| Removed | Why | What to use |
|---------|-----|-------------|
| Traceback renderer copies in `src/llm_utils/_traceback.py` and `src/speedy_utils/multi_worker/thread.py` (`CleanTracebackError.format_rich`, `_get_code_context_rich`, `_filter_traceback_frames`) | Near-identical renderer copied into three modules; only the `utils_error.py` copy was maintained. | `speedy_utils/common/utils_error.py` owns it. `_traceback.clean_traceback` still exists and delegates lazily on first call (eager import would blow the import-time budget); `UserFunctionError` now subclasses `CleanTracebackError`. |
| `src/llm_utils/_traceback.py`'s renderer | See the row above. | Same file keeps only the lazy `clean_traceback` shim. |
| `src/speedy_utils/common/report_manager.py` | `ReportManager` wrote a markdown run report to `~/.cache/speedy_utils`; nothing imported it. | `multi_thread` reports through its hooks, `ErrorStats`, and `JobSummary`. |
| `src/speedy_utils/common/patcher.py` | `patch_method` rewrote a method's source at runtime. No caller. | Fix the source. |
| `src/speedy_utils/common/function_decorator.py` | `retry_runtime` was exported from `speedy_utils` but retrying belongs to the job layer. | Retry in the caller (`ParallelLLMJob` or your own loop). |
| `src/llm_utils/lm/base_prompt_builder.py` | 344-line `BasePromptBuilder` ABC that nothing imported; `LLMSignature` is the prompt API. | `LLMSignature`. |
| `src/llm_utils/system_prompted_assistant/` | One concrete `LLM` subclass wrapping a system prompt, imported by nothing. | `LLMSignature` or a plain system message. |
| `vision_utils.plot.visualize_tensor` | Second image viewer next to the one that is tested. | `plot_images_notebook`. |
| `multi_worker.thread._chunk_items_for_processes` | Helper for the old process path; chunking is inline in `multi_thread`'s process branch. | Nothing; chunking happens inside the worker dispatch. |
| `multi_worker.common._track_multiprocessing_processes` | Guessed child processes from a "started in the last 5s" heuristic and was never called. | `_track_processes` / `_prune_dead_processes` with the pids you actually spawned. |
| `multi_worker.thread.multi_thread_standard` | Older copy of the thread pool kept for a test comparison. | `multi_thread`. |
| `tests/import_time_report.py` | Ad-hoc import timing script that duplicated the maintained tool. | `scripts/debug_import_time.py`; the 0.4s budget is guarded by `tests/test_import_time.py` and `.githooks/pre-commit`. |
| `utils_cache._QUICK_ID_MAP` and the `_try_get_quick_id` / `_try_store_quick_id` path | The weakref cache keyed on object identity, so a mutated object kept returning the digest computed before the mutation. | `identify()` recomputes the content-based digest each call. |
| `utils_cache.identify_uuid` | Second public identifier entry point with no callers. | `identify`. |
| `utils_error.handle_exceptions_with_clean_traceback` | Alias for `clean_traceback`, exported publicly. | `clean_traceback`. |
| `clock.speedy_timer`, `utils_misc.get_arg_names`, `notebook_utils.change_dir` | Small public helpers with no callers in the repo or the docs. | Standard library or your own two lines. |

## Dependencies

| Removed from `dependencies` | Why | What to use |
|-----------------------------|-----|-------------|
| `jupyter`, `jupyterlab`, `notebook` | Not imported anywhere in `src/`; the whole IPython/Jupyter stack came with every install. | Install them yourself for notebook work. |
| `fastcore` | Only used for `dict2obj` (replaced with `types.SimpleNamespace`) and an eager `parallel` import. | `types.SimpleNamespace`; `concurrent.futures`. |
| `xxhash` | Hashing needs one 64-bit digest. | `hashlib.blake2b(..., digest_size=8).hexdigest()` in `utils_cache`. |
| `matplotlib`, `pandas`, `datasets`, `chainlit` | Moved to optional extras, not deleted. | `pip install 'speedy-utils[plot]'`, `[pandas]`, `[datasets]`, `[chat]`. |
