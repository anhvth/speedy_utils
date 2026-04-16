# TDD and Regression Testing Playbook

Use this playbook for bug fixes and behavior changes.

## Purpose

A regression test protects a bug fix from coming back. The test should fail on
old behavior and pass only when the fix is correct.

## Core Properties

1. Reproduces a real bug
   A good regression test comes from an issue that actually happened.
2. Deterministic
   No flaky dependencies on random values, wall-clock timing, or external
   network services.
3. Public-API focused
   Test observable behavior through supported APIs instead of private state.
4. Single failure reason
   Keep one behavior per test so failures point to one breakage.
5. Fast
   Keep regression tests small so they run often in local development.

## TDD Flow (Red -> Green -> Refactor)

1. Red: write the smallest test that reproduces the bug.
2. Verify the test fails for the right reason.
3. Green: implement the minimum fix.
4. Refactor while keeping tests green.
5. Keep the regression test permanently.

## Repo Conventions

- Prefer pytest tests under `tests/` near the affected module.
- Name tests after behavior/bug, not generic labels.
  - Good: `test_multi_process_notebook_style_with_ssl_context_global`
  - Bad: `test_logic_3`
- Use real bug-triggering inputs when available.
- Avoid live network calls for `llm_utils` tests; use local fakes/mocks for
  transport boundaries.
- For spawn-related multiprocessing regressions, reproduce with a real script
  file (`python script.py`) when needed.

## Do / Do Not

| Do | Do Not |
|---|---|
| Test via public APIs | Assert private internals (`__dict__`, hidden state) |
| Use real bug-triggering input | Mock everything end-to-end |
| Assert specific output/error | Assert only "did not crash" |
| Keep tests isolated | Share mutable cross-test state |
| Keep one behavior per test | Bundle many behaviors in one test |

## Regression Test Checklist

- [ ] The test fails before the fix.
- [ ] Failure is deterministic and local.
- [ ] The assertion checks an explicit behavior or output.
- [ ] The test name documents the bug/behavior.
- [ ] The test is narrow, fast, and independent.

## Quick Examples

```python
def test_empty_input_regression():
    # Arrange: smallest input that previously triggered the bug
    input_data = ""
    result = public_api(input_data)
    assert result == expected_output
```

```python
def test_invalid_arg_regression():
    with pytest.raises(ValueError, match="batch must be >= 1"):
        public_api(batch=0)
```
