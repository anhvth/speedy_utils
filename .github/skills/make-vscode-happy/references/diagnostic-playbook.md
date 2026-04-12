# Diagnostic Playbook

This reference supports the `make-vscode-happy` skill for the `speedy_utils`
repository.

## Repo Rules That Matter

- Use `uv run python tools/check_syntax.py` as the source of truth for Python
  diagnostics.
- Fix root causes instead of masking errors with broad ignores.
- Keep public package import time under `0.4s`.
- Do not introduce heavy top-level imports such as `torch`, `pandas`,
  `matplotlib`, or `IPython` in `__init__.py` files.

## Fast Triage

### If the prompt names a specific file

Run:

```bash
uv run python tools/check_syntax.py --file path/to/file.py
```

Then fix the reported issue and re-run the same command.

### If the prompt is vague or broad

Run:

```bash
uv run python tools/check_syntax.py
```

If the change touches tests or test helpers, consider:

```bash
uv run python tools/check_syntax.py --include tests
```

## Common Fix Patterns In This Repo

### Overload ordering

Put the `Awaitable[...]` overload before the generic non-awaitable overload to
avoid overlapping-overload diagnostics.

### Dynamic Pydantic-style attributes

When the code intentionally sets attributes not declared in the model schema,
use a narrow ignore such as:

```python
# type: ignore[attr-defined]
```

Do not apply it broadly if a better annotation is possible.

### `tqdm` annotations

Use the string literal form in annotations:

```python
"tqdm"
```

### Lazy exports

For names loaded through `__getattr__`, follow the existing lazy-export pattern
instead of forcing eager imports for type convenience.

### `__all__` with heavy lazy names

Do not add heavy lazy names to `__all__`, because star imports will defeat lazy
loading and hurt import time.

## When a Typing Fix Touches Imports

If you changed imports in a public package, especially `speedy_utils`,
`llm_utils`, or `vision_utils`, run:

```bash
uv run python scripts/debug_import_time.py speedy_utils llm_utils vision_utils --max-total-sec 0.4 --top 12 --min-sec 0.01 --no-stdlib
```

Use that output to move heavy imports behind lazy boundaries if needed.

## Definition of Done

- Targeted diagnostics are gone.
- Shared changes were validated with a broader repo check.
- The fix preserves existing lazy-import and typing conventions.
- VS Code should agree with command-line output because the checker mirrors
  Pylance behavior for this repository.