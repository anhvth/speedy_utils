---
name: "speedy-utils-core"
description: "Use when a simple for loop should be sped up with speedy_utils multi_thread or multi_process, or when the native OpenAI client should be wrapped with llm_utils.LLM or Qwen3LLM for a cleaner API. Focus only on these repo-specific surfaces."
---

# Speedy Utils Core Skill

Use this skill only for changes involving:

- `speedy_utils.multi_worker.thread.multi_thread`
- `speedy_utils.multi_worker._multi_process.multi_process`
- `llm_utils.lm.llm.LLM`
- `llm_utils.lm.llm_qwen3.Qwen3LLM`

Reach for this skill in two common cases:

- a simple `for` loop should run faster with `multi_thread(...)` or `multi_process(...)`
- raw OpenAI client usage should be replaced with `LLM(...)` or `Qwen3LLM(...)` to make call sites cleaner and more consistent

Do not use this skill for caching, vision, dataset tooling, Ray, or generic repo cleanup outside those surfaces.

## Working rules

- Prefer public APIs and current behavior over historical helper patterns.
- Keep changes small, direct, and easy to read.
- Follow Red -> Green -> Refactor for bug fixes.
- Add or update focused regression tests for behavior changes.
- Keep imports lazy when touching LLM or Qwen3 code paths.
- Do not introduce heavy top-level imports in package `__init__.py` files.

## Repository-specific guidance

### `multi_thread`

- Primary file: `src/speedy_utils/multi_worker/thread.py`
- Use `multi_thread` for managed thread-pool execution over an input iterable.
- Preserve support for batching, ordered results, progress reporting, timeouts, and error handling.
- `error_handler` is the active control surface. Treat `stop_on_error` as compatibility behavior only.
- `n_proc > 1` is a fan-out path layered on top of `multi_thread`; avoid duplicating process orchestration logic elsewhere.

### `multi_process`

- Primary file: `src/speedy_utils/multi_worker/_multi_process.py`
- Public API returns a `list[Any]`.
- Respect the current normalization behavior around `num_procs`; do not assume auto-scaling unless the implementation explicitly does it.
- Keep fallback behavior across sequential, thread, and multiprocess execution paths coherent.
- Error handling and progress behavior should stay aligned with the thread utilities where practical.

### `LLM`

- Primary file: `src/llm_utils/lm/llm.py`
- `LLM()` defaults to the chat path.
- `LLM.generate()` expects a string prompt and supports only `n=1`.
- `LLM.pydantic_parse()` is the structured-output path.
- `LLM(..., return_dict=True)` behavior must stay compatible with the normalized dict contract used in this repo.
- Avoid constructor regressions: legacy constructor keys are intentionally rejected.

### `Qwen3LLM`

- Primary file: `src/llm_utils/lm/llm_qwen3.py`
- `Qwen3LLM.chat_completion()` returns an OpenAI-style chat message object.
- `complete_until()` returns continuation state, not a chat message.
- `complete_reasoning()` and `complete_content()` are staged helpers built on prefix completion.
- Keep tokenizer loading lazy and cached.
- Preserve reasoning/content split behavior and the assistant-prefix normalization flow.

## Clean-code bar

- Prefer one obvious implementation path.
- Remove dead branches only when you can prove they are obsolete.
- Keep naming aligned with existing public APIs.
- Add short comments only where the control flow is genuinely non-obvious.
- Avoid broad refactors unless they directly reduce risk in the touched logic.

## Validation

Run the smallest relevant checks after changes:

- `./tools/uv_test.sh <targeted_test>`
- `uv run python tools/check_syntax.py`
- `uv run ruff check .`

If import-time-sensitive files are touched, also run:

- `uv run python scripts/debug_import_time.py speedy_utils llm_utils vision_utils --max-total-sec 0.4 --top 12 --min-sec 0.01 --no-stdlib`

## Examples of when to use this skill

- Fix a regression in `multi_thread` batching or error handling.
- Adjust `multi_process` backend behavior without changing unrelated utilities.
- Update `LLM` request/response behavior while preserving current public contracts.
- Modify `Qwen3LLM` staged reasoning/content generation without breaking lazy tokenizer loading.
