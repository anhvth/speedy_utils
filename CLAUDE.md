# CLAUDE.md

This file provides guidance for working in this repository.

## Development Commands

```bash
# Install dependencies
uv sync

# Run tests
./tools/uv_test.sh -n 32

# Run a single test file
./tools/uv_test.sh tests/test_thread.py

# Run with verbose output
./tools/uv_test.sh -v

# Check import time for the exported packages
uv run python scripts/debug_import_time.py speedy_utils llm_utils vision_utils \
    --max-total-sec 0.4 --top 12 --min-sec 0.01 --no-stdlib

# Lint and format
uv run ruff check .
uv run ruff format .

# Pyright / Pylance parity check
uv run python tools/check_syntax.py

# Bump version (runs tests, then commits and pushes)
./bumpversion.sh patch  # or minor / major

# Deploy to PyPI (requires PYPI_API_TOKEN)
./scripts/deploy.sh
```

## Performance Requirement: Import Time < 0.4s

The repository hook budget is `0.4s` for the exported packages checked by
`.githooks/pre-commit`:

- `speedy_utils`
- `llm_utils`
- `vision_utils`

Use the shared helper when import time regresses:

```bash
uv run python scripts/debug_import_time.py speedy_utils llm_utils vision_utils \
    --max-total-sec 0.4 --top 12 --min-sec 0.01 --no-stdlib
```

Current code structure keeps the budget by avoiding heavy external imports at
module import time.

### What to keep lazy

These imports are intentionally kept behind helpers or function scope because
they are expensive enough to matter for import time:

- `torch`
- `pandas`
- `matplotlib`
- `IPython`
- tokenizer-loading code in `Qwen3LLM`

### How the current code does it

- `speedy_utils.__init__` imports lightweight internal modules directly.
- heavy external helpers live in `speedy_utils.__imports`.
- `llm_utils.__init__` exports selected public classes and helpers directly.
- `vision_utils` keeps plotting and image backends lazy inside function bodies.

Do not add new top-level imports for heavy third-party libraries in public
package `__init__.py` files unless you have rechecked import-time impact.

### Hook vs test budget

The pre-commit hook is stricter than the regression test:

- `.githooks/pre-commit` enforces `0.4s`
- `tests/test_import_time.py` currently guards `speedy_utils` at `1.0s`

Treat the hook budget as the real policy.

## Architecture

### Package Structure

The wheel currently ships four packages from `src/`:

- `speedy_utils`: caching, file I/O, formatting, timing, and parallel helpers
- `llm_utils`: OpenAI-compatible LLM wrappers, memoized clients, chat-format helpers
- `vision_utils`: image loading, notebook plotting, and mmap-backed image datasets
- `datasets_utils`: dataset inspection helpers such as `viz_chat`

### Important Modules

- `src/speedy_utils/common/utils_cache.py`
  - `memoize`, `imemoize`, `identify`
  - default disk cache root: `~/.cache/speedy_cache`
- `src/speedy_utils/common/utils_io.py`
  - `load_jsonl`, `fast_load_jsonl`, `load_json_or_pickle`, `load_by_ext`
- `src/speedy_utils/multi_worker/thread.py`
  - `multi_thread`
- `src/speedy_utils/multi_worker/_multi_process.py`
  - `multi_process`
- `src/llm_utils/lm/llm.py`
  - `LLM.chat_completion`, `LLM.generate`, `LLM.pydantic_parse`, `LLM.inspect_history`
- `src/llm_utils/lm/llm_signature.py`
  - `LLMSignature`
- `src/llm_utils/lm/llm_qwen3.py`
  - `Qwen3LLM`, `complete_until`, `complete_reasoning`, `complete_content`
- `src/vision_utils/io_utils.py`
  - `read_images`, `read_images_cpu`, `read_images_gpu`, `ImageMmap`, `ImageMmapDynamic`
- `src/vision_utils/plot.py`
  - `plot_images_notebook`
- `src/datasets_utils/viz_chat.py`
  - dataset and conversation inspection CLI

## Public API Notes

### `speedy_utils`

- `memoize` supports sync and async functions.
- `load_json_or_pickle` is for `.json` and pickle-like files.
- use `load_jsonl` or `fast_load_jsonl` for JSONL.
- `multi_process(num_procs=None)` currently normalizes to `1`; it does not auto-pick a process count.

### `llm_utils`

- `LLM()` defaults to the chat path.
- `LLM.generate()` uses the completions API and returns a `CompletionChoice`-like object.
- `LLM.generate()` expects a string prompt and only supports `n=1`.
- `LLM.pydantic_parse()` is the structured-output API.
- `LLM(..., return_dict=True)` returns a normalized dict with `completion`, `message`, `messages`, and `parsed`.
- `LLMSignature` defaults structured outputs to the signature's output model.
- `Qwen3LLM.chat_completion()` returns an OpenAI-style `ChatCompletionMessage` with optional dynamic attrs such as `reasoning_content`, `usage`, and `call_count`.
- `Qwen3LLM.complete_until()` returns a continuation-state object, not a `ChatCompletionMessage`.

### `vision_utils`

- `read_images*()` returns `dict[path, ndarray | None]`.
- `plot_images_notebook()` expects arrays/tensors or lists/tuples of arrays.
- if you loaded images with `read_images*()`, pass `list(images.values())` to the plotter.
- `ImageMmap` and `ImageMmapDynamic` take image-path sequences and build cache files as needed.

## CLI Tools

Registered console scripts in `pyproject.toml`:

- `mpython`
- `kill-mpython`
- `sp_chat`
- `spu-prefetch-large-model`
- `viz_chat`
- `openapi_client_codegen`

## Code Style

- Line length: `88`
- Quote style: double quotes
- Ruff is the formatter and linter
- Avoid unnecessary wrappers and ceremony; see `CODING_STYLE.md`
- Keep public docs/examples aligned with tests and current implementation, not with historical helper scripts

## Type Checking with Pyright

Use:

```bash
uv run python tools/check_syntax.py
```

This mirrors what VS Code Pylance reports for the configured project files.

### pyright config (`pyrightconfig.json`)

Current notable suppressions:

| Rule | Why suppressed |
|------|----------------|
| `reportMissingImports` | Optional GPU and backend dependencies are not always installed |
| `reportAttributeAccessIssue` | OpenAI response models and runtime message objects gain dynamic attrs such as `usage`, `reasoning_content`, and `call_count` |
| `reportUnsupportedDunderAll` | Some modules rely on broad export patterns that are noisy for pyright |

### Common annotation patterns in this repo

- use string literal `"tqdm"` when needed in type hints
- put `Awaitable[R]` overloads before generic `R` overloads
- use `# type: ignore[attr-defined]` for dynamic OpenAI message attrs where necessary
- keep dataset `__getitem__` overrides narrow and explicit

## Version Management

Version lives in `pyproject.toml`. `./bumpversion.sh` currently:

1. runs tests
2. bumps the package version
3. commits and pushes
