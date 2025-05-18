## Coding Standards
- **Naming**
  - `snake_case` for functions & variables
  - `PascalCase` for classes
- **Formatting**
  - 4‑space indentation
  - Single quotes for strings
  - f‑strings for interpolation
- **Typing & Docs**
  - Add **PEP‑484** type hints to *all* public functions & methods
  - Docstrings stay minimal; comment only the non‑obvious
  - Prefer **`Sequence[...]` / `Mapping[...]`** for input parameters  
    (avoids Pylance “invariant list” complaints)
  - Guard optionals before use:  
    `assert foo is not None`, or early `if foo is None: return`
  - Use `typing.cast` or pattern‑matching to narrow types explicitly
  - Provide `@overload` signatures when return type varies by argument
- **Comparisons**
  - Use `is` / `is not` for `None`, `True`, `False`
  - Use `==` / `!=` for value comparison
- **Imports**
  - Always `from __future__ import annotations` in new files (faster & no forward‑ref strings)

## Tooling Assumptions
- **Editor** VS Code
- **Pylance setting**  
  ```jsonc
  "python.analysis.typeCheckingMode": "basic"
````

* Code must compile *clean* (no yellow squiggles) under **basic** mode
* Treat any Pylance error as blocking
* **Pylance‑friendly patterns**

  * Never pass a `T | None` where `T` is required
  * Default str parameters with `""` rather than `None`
  * When a third‑party API expects a `list[SomeProtocol]`, build that exact list; don’t pass `dict`s
  * Keep `List[T]` for *outputs* or truly mutable collections; otherwise prefer `Sequence[T]`

\## Preferred Libraries

* Logging  `loguru`
* Database `peewee` + PostgreSQL (or SQLite locally)
* Web API  `FastAPI`
* Data model `pydantic v2`
* Testing  `pytest`
* Async     `asyncio` + `aiohttp`

\## When the user provides a **list of problems**

1. Reproduce every message locally (copy/paste into VS Code).
2. Fix the code **in order**, committing once the list is empty.
3. In the reply, show only the final file(s) needed to eliminate the warnings.

\## Common problems to avoid
*Stick to these choices unless explicitly instructed otherwise.*

* Invariant‑list errors → take `Sequence[...]`, not `List[...]`
* Missing attribute on `type[str]` → check that the variable is really a `BaseModel` subclass *before* calling model methods
* `str | None` passed where `str` required → coerce with `or ""` or explicit check
* Forgetting `list()` around an `Iterable` when calling OpenAI → Pylance will flag mismatched type
* Returning `Any` when a precise type is knowable → add overloads or casts
* Ignoring `async` context managers in `aiohttp` → always `async with`
