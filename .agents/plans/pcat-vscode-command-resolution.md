# Plan: Fix and Validate PCAT VS Code Command Resolution

Related problem report:

- `.agents/PRs/pcat-vscode-command-resolution.md`

## Goal

Make the PCAT VS Code extension reliable in non-interactive environments,
especially for preview mode, and validate the fix against the problem report.

## Working Hypothesis

The bug exists because preview uses `child_process.exec(...)` in a non-
interactive `/bin/sh` environment while interactive open uses the integrated
terminal. The fix should remove that mismatch and stop relying on shell aliases.

## Fix Strategy

1. Introduce explicit command resolution.

   - Read `pcat.command` as a user-provided command string.
   - Validate it before execution.
   - If it is the default `pcat`, resolve whether a real executable exists.
   - If not resolvable, show a clear error telling the user to set
     `pcat.command` to something explicit such as:
     - `uv run pcat`
     - `uvx --from /home/anhvth8/projects/speedy_utils pcat`
     - an absolute executable path

2. Unify execution semantics.

   - Ensure interactive open and plain preview use the same resolved command
     model.
   - Avoid depending on shell aliases, shell functions, or interactive rc files.
   - Prefer spawning a command with explicit args over concatenating a shell
     command string when practical.

3. Improve failure reporting.

   - Detect missing command resolution before invoking preview.
   - Report the exact setting key to change: `pcat.command`.
   - Include one working example command in the error.

4. Keep the `.jsonl` custom editor behavior intact.

   - Do not remove the current default custom-editor association.
   - Do not regress `Open As Text` or integrated-terminal launch behavior.

## Implementation Notes

- Replace shell-string execution for preview with a spawn-style path where the
  command and args are separated.
- If `pcat.command` remains a free-form string, parse it once and reuse the
  parsed command for both preview and integrated terminal launch.
- Consider adding a helper that returns a normalized `{ command, args }`
  structure plus a user-facing diagnostic on failure.

## Test Plan

### Reproduction Test Against The Problem Report

1. Install the fixed extension build.
2. Configure no alias-dependent workaround in the extension itself.
3. Use the same environment described in the problem report where `pcat` is
   available only through interactive shell setup.
4. Attempt preview.
5. Verify one of these outcomes:
   - preview succeeds with an explicit configured command, or
   - the extension gives a clear configuration error instead of `/bin/sh: ... pcat: not found`.

### Manual Behavior Tests

1. Open a `.jsonl` file.
2. Confirm the PCAT custom editor still opens by default.
3. Confirm automatic terminal launch still works when enabled.
4. Run `PCAT: Preview Plain Row`.
5. Confirm the preview opens JSON content in a text editor tab.
6. Use `PCAT: Reopen As Text Editor` and confirm raw text opens normally.

### Negative Tests

1. Set `pcat.command` to an invalid executable name.
2. Run preview.
3. Verify the error message references `pcat.command` and suggests a concrete
   valid command.

### Regression Tests

1. Verify row index prompting still works for preview.
2. Verify HF dataset split prompting still works where applicable.
3. Verify the package still builds:
   - `npm run compile`
   - `npm run package`

## Definition of Done

- The failure from the problem report is either fixed or converted into a clear,
  actionable configuration error.
- Preview no longer depends on interactive shell aliases.
- Interactive open and preview follow the same command-resolution policy.
- The updated extension compiles, packages, and passes the manual checks above.