---
name: 'make-vscode-happy'
description: 'Reduce VS Code Python red squiggles to zero in this repository. Use when Pylance or pyright reports errors, when tools/check_syntax.py fails, or when a change should leave VS Code diagnostics clean.'
argument-hint: 'Optional file path, module name, or failing area to fix'
---

# Make VS Code Happy

Use this skill when the goal is to make VS Code and Pylance happy in this
repository by driving Python diagnostics to zero with the repo's existing
checker: `tools/check_syntax.py`.

## When to Use

Use this skill when you need to:
- Fix Python red squiggles in VS Code.
- Reconcile Pylance diagnostics with command-line validation.
- Clean up pyright issues before committing.
- Verify that a targeted change did not leave new diagnostics behind.

## Primary Checker

The authoritative checker for this repo is:

```bash
uv run python tools/check_syntax.py
```

Useful variants:

```bash
uv run python tools/check_syntax.py --file src/llm_utils/lm/llm_qwen3.py
uv run python tools/check_syntax.py --include tests
uv run python tools/check_syntax.py --json
```

This script mirrors the diagnostics VS Code Pylance shows for the repository's
`pyrightconfig.json`.

## Procedure

1. Pick scope first.
   - If the request names a file, start with `--file` for that file.
   - If the issue sounds cross-cutting or unclear, run the full check first.
2. Read diagnostics and group them by root cause instead of fixing them one by
   one in isolation.
3. Fix the real source of the type error.
   - Prefer correcting annotations, overloads, return types, and control flow.
   - Use repo-approved ignores only when the code is intentionally dynamic.
4. Re-run the smallest relevant check after each edit.
5. Before concluding, run the broader validation needed for the scope.
   - If you touched one file, re-run that file and then a full check if the
     change affects shared APIs.
   - If you changed common utilities, package exports, or typing helpers, run
     the full check.
6. If the fix changes imports in public packages, protect the import-time
   budget as well.

## Decision Points

- Single-file symptom vs shared typing pattern:
  Start narrow for local regressions; go broad if the same diagnostic appears in
  multiple files or stems from a shared helper.
- True bug vs accepted dynamic pattern:
  Prefer a real type fix first. Only use a targeted ignore when the repository
  already treats that pattern as intentionally dynamic.
- Typing fix vs performance regression:
  Do not fix types by introducing heavy top-level imports in `__init__.py` or
  other import-sensitive modules.

## Completion Criteria

- `tools/check_syntax.py` reports zero relevant errors for the changed scope.
- Touched files do not introduce new warnings unless the user accepted them.
- The fix matches repository typing conventions.
- No eager heavy imports were added just to satisfy typing.

## References

- [Diagnostic playbook](./references/diagnostic-playbook.md)
