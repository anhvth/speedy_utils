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
## When user provide a list of problems:
Directly go over the list and edit the code to fix the problems.

## Common problems when generated code to avoid:
*Stick to these choices unless explicitly instructed otherwise.*

---

## Rules for Code Generation

- Each line should less than 88 characters.