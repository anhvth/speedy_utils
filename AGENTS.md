# Repository Guidelines

## Project Structure & Module Organization

- `src/` contains `speedy_utils`, `llm_utils`, and `vision_utils` packages.
- `tests/` holds automated tests; `examples/` and `notebooks/` are usage references.
- `scripts/` and `experiments/` are for tooling and experiments; keep changes scoped.
- `docs/` contains documentation assets.
- `pyproject.toml`, `ruff.toml`, and `bumpversion.sh` define tooling and release helpers.

## Build, Test, and Development Commands

- `pip install -e .` installs the package in editable mode.
- `uv pip install -e .` is a drop-in alternative if you use uv.
- `python -m pytest` or `pytest tests` runs the test suite.
- `ruff check .` runs lint rules; `ruff format .` formats code.

## Coding Style & Naming Conventions

- Formatting is aligned with Black-style settings (88 char lines) and Ruff rules in `ruff.toml`.
- Use `snake_case` for Python modules and functions; class names follow `CamelCase`.
- Keep public APIs exported from `src/*/__init__.py` small and intentional.

## Testing Guidelines

- Tests live in `tests/` and should be named `test_*.py`.
- Prefer pytest-style assertions and keep fixtures near the tests that use them.

## Commit & Pull Request Guidelines

- Recent history includes informal messages; prefer concise, descriptive imperatives (e.g., `add cache backend`).
- PRs should include test results and note any new dependencies or optional extras.
