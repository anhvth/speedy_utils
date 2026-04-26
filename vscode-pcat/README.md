# PCAT VS Code Extension

This extension adds VS Code commands for the existing `pcat` terminal UI from
`speedy_utils`.

## What it does

- opens the current file or selected Explorer path in `pcat`
- opens `.jsonl` files with the PCAT custom editor by default
- supports `.json`, `.jsonl`, directories of `.json` files, and HF dataset directories
- previews a selected row with `pcat --plain` in a regular VS Code editor tab
- keeps the original `pcat` keyboard workflow by running the interactive app in the integrated terminal

When a `.jsonl` file is opened, the extension now claims that editor type and
automatically launches `pcat` in the integrated terminal. The editor tab shows
PCAT-specific actions such as reopening at a row index or opening the raw text.

## Commands

- `PCAT: Open`
- `PCAT: Open With Row Index`
- `PCAT: Preview Plain Row`

## Configuration

The extension only shells out to `pcat`, so you need a working command.

Recommended settings:

```json
{
  "pcat.command": "uv run pcat"
}
```

If you want to use the published package without a local checkout:

```json
{
  "pcat.command": "uvx --from git+https://github.com/anhvth/speedy_utils pcat"
}
```

Available settings:

- `pcat.command`: shell command prefix used to start `pcat`
- `pcat.extraArgs`: extra arguments passed before the path
- `pcat.defaultSplit`: default split for HF datasets saved with `load_from_disk()`
- `pcat.terminalName`: terminal label for interactive sessions
- `pcat.reuseTerminal`: reuse the same integrated terminal across launches
- `pcat.autoLaunchOnOpen`: auto-run `pcat` whenever a `.jsonl` file opens in the PCAT custom editor

## Build

```bash
cd vscode-pcat
npm install
npm run compile
npm run package
```

The `package` script emits a `.vsix` file that can be installed in VS Code.