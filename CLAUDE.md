# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest -n 32

# Run a single test file
uv run pytest tests/test_thread.py

# Run with verbose output
uv run pytest -v

# Check import time (must be < 0.4s)
uv run python -c "import time; start=time.perf_counter(); import speedy_utils; print(f'{time.perf_counter()-start:.3f}s')"

# Detailed import analysis
uv run python scripts/debug_import_time.py speedy_utils --min-sec 0.05 --no-stdlib

# Lint with ruff
uv run ruff check .

# Format with ruff
uv run ruff format .

# Bump version (runs tests first, then commits and pushes)
./bumpversion.sh patch  # or minor, major

# Deploy to PyPI (requires PYPI_API_TOKEN)
./scripts/deploy.sh
```

## Performance Requirement: Import Time < 0.4s

**CRITICAL:** All modules must import in under 0.4 seconds. This is enforced by a git pre-commit hook.

### Lazy Import Strategy

Import heavy modules inside functions, NOT at module level:

```python
# BAD - slow module-level import
import torch
def some_function():
    return torch.tensor([1, 2, 3])

# GOOD - lazy import
def some_function():
    import torch  # Only imports when called
    return torch.tensor([1, 2, 3])
```

### Heavy Modules to Lazy-Load

| Module | Import Time | Strategy |
|--------|-------------|----------|
| `torch` | ~5s | Lazy import inside functions |
| `matplotlib` | ~1.3s cumulative | Lazy via `__getattr__` in `__init__.py` |
| `pandas` | ~1.9s cumulative | Lazy via `__getattr__` in `__init__.py` |
| `IPython` | ~2.1s cumulative | Lazy via `__getattr__` in `__init__.py` |

### Rule: Never top-level import heavy modules in `__init__.py`

Only a short `__getattr__` block (keyed on module name) is allowed for heavy
modules. All internal `speedy_utils.*` imports must remain direct top-level
imports. External heavy modules (`pandas`, `matplotlib`, `IPython`, `torch`)
must **never** appear as top-level imports in any `__init__.py`.

```python
# BAD — in __init__.py
import pandas as pd          # adds ~1.3 s

# GOOD — in __init__.py
# pd is declared in _HEAVY dict, resolved by __getattr__ on first access

# GOOD — inside a function anywhere in the package
def load_dataframe(path):
    import pandas as pd      # only pays cost when called
    return pd.read_csv(path)
```

### Rule: Heavy lazy names must NOT be in `__all__`

`from speedy_utils import *` resolves every name in `__all__`, which triggers
`__getattr__` for each heavy name and defeats the lazy-load strategy.  Only
include names that are directly importable at module load time in `__all__`.
Heavy names (`np`, `pd`, `matplotlib`, `plt`, `get_ipython`, `HTML`, `display`,
`BaseModel`) must be kept out of `__all__`; they remain accessible via explicit
attribute access (`speedy_utils.pd`) or explicit import
(`from speedy_utils import pd`).

## Architecture

### Package Structure

Three main packages under `src/`:

- **`speedy_utils`**: Core utilities (caching, IO, parallel processing, timing)
  - `common/`: Cache (`memoize`, `imemoize`), IO (`load_json_or_pickle`), logging, error handling
  - `multi_worker/`: `multi_thread`, `multi_process`, dataset sharding helpers
  - `__imports.py`: Lazy-loaded heavy dependencies (torch, pandas, matplotlib)

- **`llm_utils`**: LLM integration layer
  - `lm/llm.py`: Main `LLM` class with OpenAI-compatible client support, structured outputs, caching
  - `lm/openai_memoize.py`: `MOpenAI` - memoized OpenAI client
  - `chat_format/`: Transform between ChatML, ShareGPT, text formats

- **`vision_utils`**: Image processing utilities
  - `io_utils.py`: Image loading, video frame extraction
  - `plot.py`: Matplotlib wrappers for visualization

### Key Patterns

**Lazy Exports via `__getattr__`**: Both `speedy_utils` and `llm_utils` use lazy attribute access to keep import time fast:

```python
# In __init__.py
_LAZY_ATTRS = {"LLM": ("llm_utils.lm", "LLM")}

def __getattr__(name):
    module_name, attr_name = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)
```

**Memoization**: `@memoize` caches function results to disk; `@imemoize` caches in memory:

```python
from speedy_utils import memoize, imemoize

@memoize  # Persists to disk at ~/.cache/speedy_utils/
def expensive_call(x):
    return x ** 2

@imemoize  # In-memory only
def fast_cached_call(x):
    return x * 2
```

**Parallel Processing with Error Handling**:

```python
from speedy_utils import multi_thread, multi_process

# Three error handling modes
results = multi_thread(func, items, error_handler='raise')  # Stop on error (default)
results = multi_thread(func, items, error_handler='ignore')  # Continue, return None for errors
results = multi_thread(func, items, error_handler='log')    # Log errors, continue
```

### CLI Tools

The package provides these CLI commands:

- `mpython`: Run Python scripts across multiple tmux windows with GPU/CPU allocation
- `kill-mpython`: Kill all mpython sessions
- `sp_chat`: Interactive chat CLI
- `spu-prefetch-large-model`: Prefetch models to disk cache

## Code Style

- Line length: 88 characters (Black/Ruff default)
- Ruff handles linting and formatting (config in `ruff.toml` and `pyproject.toml`)
- Ignore rules `E402`, `F401`, `F403` in `__init__.py` files for lazy import patterns
- Quote style: double quotes
- Avoid hacky workarounds like sys.path.insert


## Version Management

Version is in `pyproject.toml`. Use `./bumpversion.sh` to bump version, which:
1. Runs pytest (requires 90% pass rate)
2. Bumps version using `uv version`
3. Commits and pushes
