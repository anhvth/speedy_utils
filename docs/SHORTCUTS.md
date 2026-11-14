# Deliberate shortcuts

Places where the code takes a simpler path on purpose and says so with a
`shortcut:` comment. Each one is a known limit, not a bug.

Regenerate this list with `rg -n "shortcut:" src` — line numbers drift as the
files change.

| Where | The limit it accepts | Upgrade when |
|-------|----------------------|--------------|
| `src/speedy_utils/common/utils_io.py:69` | Dumping a pydantic model whose `model_dump` fails falls back to `types.SimpleNamespace(**data)`, so nested dicts stay plain dicts. | A caller needs attribute access on nested values after unpickling. |
| `src/speedy_utils/multi_worker/thread.py:1128` | On Ctrl-C, `kill_all_thread(SystemExit)` kills every tracked pool in the process, not only this `multi_thread` call. | A background pool must survive Ctrl-C; then track threads per call. |
| `src/vision_utils/io_utils.py:357` | A cache rebuild truncates the shared `.dat` in place, so a process still mapping an older cache would fault. | A rebuild can run while another process reads the cache; write one cache file per hash version instead. |
