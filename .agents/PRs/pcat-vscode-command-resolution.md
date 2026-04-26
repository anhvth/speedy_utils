# Problem Report: PCAT VS Code Command Resolution Fails Outside Interactive Shells

## Summary

The PCAT VS Code extension opens `.jsonl` files and can launch `pcat` in the
integrated terminal, but preview actions can fail with:

```text
PCAT preview failed: /bin/sh: 1: pcat: not found
```

This is a shipped behavior bug in the extension, not a user configuration
mistake.

## User-visible Symptom

- Opening a `.jsonl` file can show the PCAT custom editor as expected.
- Using `PCAT: Preview Plain Row` fails with `pcat: not found`.
- Interactive terminal launches may appear to work in the same workspace.

## Reproduction

1. Install the current `pcat-vscode` extension build.
2. Leave `pcat.command` at its default value of `pcat`.
3. Use a shell where `pcat` is defined only through interactive shell setup,
   such as an alias in `.zshrc`.
4. Open a `.jsonl` file and run `PCAT: Preview Plain Row`.
5. Observe the failure notification.

## Confirmed Evidence

Interactive shell resolution succeeds:

```text
alias pcat='uvx --from /home/anhvth8/projects/speedy_utils pcat'
```

The non-interactive shell used by the extension cannot resolve it:

```text
/bin/sh: 32: source: not found
__PCAT_MISSING_IN_SH__
```

## Root Cause

The extension currently has two different execution paths:

- Interactive open uses the VS Code integrated terminal.
- Plain preview uses `child_process.exec(...)`.

The preview path inherits Node's default shell execution behavior, which uses a
non-interactive `/bin/sh` environment. That environment does not load the same
interactive shell aliases, functions, or PATH mutations that make `pcat`
available in the user's zsh terminal.

As a result, the extension is inconsistent:

- `pcat` can appear available in the integrated terminal.
- the same command string fails in preview mode.

## Why This Is a Product Bug

- The default extension setting is `pcat.command = "pcat"`.
- In this environment, `pcat` is not a real executable on PATH for `/bin/sh`.
- The extension depends on shell-specific behavior without resolving or
  validating the command.
- The extension does not fail early with actionable guidance.

## Impact

- `.jsonl` custom-editor flows are unreliable.
- preview is broken in common remote and shell-customized setups.
- the extension currently depends on undocumented shell state.

## Expected Behavior

The extension should either:

- resolve `pcat` in a shell-agnostic way before use, or
- require an explicit executable command and validate it clearly, or
- reuse one consistent execution strategy for both terminal and preview flows.

At minimum, preview and open actions must succeed or fail with the same command
resolution behavior.

## Acceptance Criteria

- `PCAT: Preview Plain Row` works when `pcat.command` is configured to a valid
  executable command.
- Default behavior does not silently rely on zsh aliases.
- When command resolution fails, the user sees an actionable error describing
  how to set `pcat.command`.
- Interactive open and preview use the same command-resolution rules.