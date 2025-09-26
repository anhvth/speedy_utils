## Coding Standards

- **Naming**
  - snake_case for functions & variables
  - PascalCase for classes
- **Formatting**
  - 4-space indentation
  - Single quotes for strings
  - f-strings for interpolation
- **Typing & Docs**
  - Add type hints to all public functions & methods
  - Keep docstrings minimal; comment only non-obvious logic
- **Comparisons**
  - Use `is` / `is not` when comparing with `None`

## Tooling Assumptions

- Editor: **VS Code**
- Pylance setting: `"python.analysis.typeCheckingMode": "basic"`
  - Code must satisfy **basic** type checking (no strict-mode warnings).
  - Prefer clear, straightforward typing over complex generics.

## Preferred Libraries

- Logging: `loguru`
- Database: `Peewee` + **PostgreSQL** (or **SQLite** locally)
- Web API: `FastAPI`
- Pydantic: `FastAPI`
- Testing: `pytest`
- Async: `asyncio` + `aiohttp`
- PydanticV2: `pydantic` (v2.x)

## Project Architecture

- **speedy_utils**: General utility library for caching, parallel processing, file I/O, data manipulation. See `src/speedy_utils/__init__.py` for public API.
- **llm_utils**: LLM-specific utilities including chat formatting, language model tasks, vector caching, and VLLM server management. See `src/llm_utils/__init__.py`.
- Data flows: Function caching with disk/memory backends, parallel execution via threads/processes, auto-detect file serialization (JSON/JSONL/Pickle).

## Developer Workflows

- **Build/Install**: `uv pip install .` or `pip install -e .` for development.
- **Test**: `python test.py` (runs unittest suite) or `pytest` for individual tests.
- **Lint/Format**: `ruff check .` and `ruff format .` (configured in `ruff.toml`).
- **Type Check**: `mypy` (strict settings in `pyproject.toml`).
- **Debug**: Use `ipdb` for interactive debugging, `debugpy` for VS Code.
- **Scripts**: `mpython` for multi-process Python execution, `svllm` for VLLM server, `svllm-lb` for load balancer.

## Integration Points

- **External APIs**: OpenAI API via `llm_utils.lm.openai_memoize`, VLLM servers for local inference.
- **Distributed**: Ray backend for parallel processing (install with `[ray]` extra).
- **Async**: `aiohttp` for async HTTP, `asyncio` for concurrency.
- **Dependencies**: Managed via `pyproject.toml`, install with `uv` preferred.

## When user provide a list of problems:

Directly go over the list and edit the code to fix the problems.

## Common problems when generated code to avoid:

_Stick to these choices unless explicitly instructed otherwise._

---

## Rules for Code Generation

- Each line should less than 88 characters.
