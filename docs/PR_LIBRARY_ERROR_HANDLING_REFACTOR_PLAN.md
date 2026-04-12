# PR Plan: Library-Friendly Error Handling (Replace Core `SystemExit`)

## PR Title
Refactor core library error paths to raise exceptions instead of terminating process

## Why this PR
Current behavior mixes CLI-style termination (`sys.exit` / `SystemExit`) into library/core execution paths (`multi_process`, `multi_thread`, traceback decorators).  
That makes composition, retries, and testing harder for downstream callers.

Goal: keep CLI behavior for scripts, but make library APIs raise catchable exceptions.

---

## Scope

### In Scope
- `speedy_utils.multi_worker` core runtime error paths.
- Shared traceback decorators in:
  - `src/speedy_utils/common/utils_error.py`
  - `src/llm_utils/_traceback.py`
- Tests that currently assert `SystemExit` for library API calls.

### Out of Scope
- CLI script entrypoints that should still return non-zero exit codes (e.g. files under `src/*/scripts/` and `if __name__ == "__main__"` wrappers).
- Functional behavior changes to parallel scheduling, batching, or import-time performance.

---

## Desired End State
- Library calls (`multi_process`, `multi_thread`, decorated LLM methods) raise structured exceptions, not `SystemExit`.
- Human-readable rich traceback formatting is preserved.
- CLI wrappers remain responsible for converting exceptions to process exit codes.

---

## Architecture Decision

Introduce explicit library exceptions:
- `SpeedyExecutionError` (base for worker/runtime failures)
- `SpeedyWorkerError` (worker function failure with backend context)
- `SpeedySerializationError` (unpicklable input/output/result queue payloads)
- `CleanTracebackError` remains used by traceback decorators

Behavior rule:
- Core/lib layer: `raise`.
- CLI layer: `except Exception -> print -> sys.exit(code)`.

---

## Agent Work Split (Parallel-Friendly)

### Agent 1 - Multi-process core path
**Ownership**
- `src/speedy_utils/multi_worker/common.py`
- `src/speedy_utils/multi_worker/_mp_backends.py`
- `src/speedy_utils/multi_worker/_multi_process.py` (if glue changes needed)

**Tasks**
1. Replace `_display_formatted_error_and_exit` with non-terminating equivalent:
   - keep formatting side-effect
   - return/raise structured exception instead of `sys.exit(1)`
2. Update fatal event handling in `_mp_backends.py`:
   - convert `"fatal"` / `"spawn_error"` into raised exception
   - preserve rich traceback text and backend metadata
3. Ensure unpicklable result path raises `SpeedySerializationError`.

**Acceptance**
- No `sys.exit` remains in these library files.
- Existing error context (function, backend, traceback hints) still shown.

---

### Agent 2 - Multi-thread core path
**Ownership**
- `src/speedy_utils/multi_worker/thread.py`

**Tasks**
1. In `error_handler == "raise"` flow, replace `sys.exit(1)` with raised library exception.
2. Keep cancelation semantics (`_cancel_futures`, pruning) unchanged.
3. Keep `kill_all_thread` control utility behavior unless directly required for consistency.

**Acceptance**
- `multi_thread(..., error_handler="raise")` raises exception to caller.
- No regression in timeout/keyboard interrupt behavior.

---

### Agent 3 - Traceback decorator stack
**Ownership**
- `src/speedy_utils/common/utils_error.py`
- `src/llm_utils/_traceback.py`
- optionally small touchpoints in `llm_utils/lm/*` tests only

**Tasks**
1. In decorator main-thread path, replace `sys.exit(1)` with raised `CleanTracebackError`.
2. Keep rich formatting output available (print then raise, or raise with preserved context).
3. Ensure behavior is consistent between speedy_utils and llm_utils traceback modules.

**Acceptance**
- Decorated library methods no longer terminate interpreter.
- Tests can assert catchable exceptions.

---

### Agent 4 - Test migration + CLI safety checks
**Ownership**
- `tests/test_process.py`
- `tests/test_thread.py`
- `tests/llm_utils/test_llm_call_contract.py`
- targeted other tests asserting `SystemExit` for library APIs

**Tasks**
1. Replace library-path `pytest.raises(SystemExit)` with new exception assertions.
2. Keep CLI tests that intentionally assert `SystemExit` unchanged (or make explicit they are CLI-only).
3. Add regression for unpicklable result path:
   - assert new exception type and clear message.

**Acceptance**
- Test intent remains same, but exception contract is library-friendly.

---

## Integration Order
1. Merge Agent 1 first (defines core exception contract for process backend).
2. Merge Agent 2 and Agent 3 in parallel after contract is stable.
3. Merge Agent 4 last to align tests with final behavior.
4. Run full validation gate.

---

## Validation Gate (Required)
- `uv run ruff check .`
- `uv run python tools/check_syntax.py`
- `./tools/uv_test.sh -n 32`
- Focused checks:
  - `uv run pytest tests/test_process.py -k "error_handler_raise or unpicklable" -q`
  - `uv run pytest tests/test_thread.py -k "raise or interrupt" -q`
  - `uv run pytest tests/llm_utils/test_llm_call_contract.py -k "pydantic_parse" -q`
- Import-time budget:
  - `uv run python scripts/debug_import_time.py speedy_utils llm_utils vision_utils --max-total-sec 0.4 --top 12 --min-sec 0.01 --no-stdlib`

---

## Risks and Mitigations
- Risk: hidden callers rely on `SystemExit`.
  - Mitigation: changelog note + migration hint (`catch Exception` instead of process exit).
- Risk: noisy duplicate traceback output when both formatting and raising occur.
  - Mitigation: standardize one formatting point in core helper.
- Risk: behavior drift between `speedy_utils` and `llm_utils` decorators.
  - Mitigation: keep both modules in lockstep via mirrored tests.

---

## Migration Notes for Users
- Before: some library calls could terminate process via `SystemExit`.
- After: same failures raise catchable exceptions.
- CLI commands still exit with status codes as before.

---

## Definition of Done
- No `sys.exit`/`SystemExit` in library execution paths of `multi_worker` and traceback decorators.
- CLI script behavior unchanged.
- Tests updated and green.
- Import-time and type/syntax gates pass.
