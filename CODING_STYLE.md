# CODING_STYLE.md

## Purpose

This is the coding style for this agent.

The goal is not generic "clean code". The goal is **code that is fast to read, easy to reason about, and easy to modify without scrolling through ceremony**.

Optimize for:

* **scanability**
* **local clarity**
* **low vertical sprawl**
* **honest abstractions**
* **small cognitive load**

Do **not** optimize for:

* maximum explicitness everywhere
* boilerplate for its own sake
* style-guide cargo culting
* abstraction count
* defensive verbosity

---

## Core philosophy

### 1. Brevity facilitates reasoning

Shorter code is better **when it helps the reader see the whole idea at once**.

Do not golf code. Do not compress code until it becomes cryptic. But do remove ceremony aggressively.

A good test:

> Can a strong reader understand the main idea without scrolling or jumping between fake helpers?

### 2. One visible thought per chunk

A function, block, or short run of lines should usually express one coherent idea.

Prefer:

* one small real function
* one obvious transformation
* one direct branch

Avoid:

* sprawling setup before the real work starts
* helpers that only forward arguments
* wrappers that rename nothing and clarify nothing

### 3. Names must earn their existence

A new function name, variable name, or class name is not free.

Only introduce a name when it:

* captures a real concept
* removes real clutter
* creates a meaningful boundary
* is reused enough to matter

Do not create helpers that are just a composed one-liner with no new concept.

### 4. Prefer honest code over fake cleanliness

Do not make `main()` or a top-level function look clean by pushing noise into pointless helpers.

If the real work is simple, let it be visible.

### 5. Short scope, short name

Use shorter names for shorter-lived variables.

Examples:

* `i`, `j`, `k`, `v`, `x`, `y`, `n` are fine in tight local scope
* `cfg`, `msg`, `tmp`, `idx`, `out`, `src` are fine when locally obvious
* keep explicit names for domain concepts that carry real meaning

Do not use long names everywhere just to look respectable.

---

## Non-negotiable rules

### Keep the main flow visible

The top-level flow of a script or function should be readable in one pass.

Good:

```python
def main() -> None:
    args = cli()
    ref_pool = build_reference_pool(load_by_ext(str(args.reference_items)))
    ds = load_ds(args.input)
    rows = pick_rows(ds, row_idx=args.row_index, limit=args.limit)
    out = rewrite_all(rows, ref_pool=ref_pool, cfg=args.client_config)
    save_jsonl(out, args.output)
```

Bad:

```python
def main() -> None:
    args = parse_args()
    reference_pool = load_reference_examples(args.reference_items)
    dataset = load_input_dataset(args.input)
    items = select_items(dataset, row_index=args.row_index, limit=args.limit)
    results = process_items(items, reference_pool=reference_pool, client_config_path=args.client_config)
    write_results(results, args.output)
```

The second version is not terrible, but it hides too much of the actual work behind weak helper names.

### Delete fake helpers

Inline helpers that:

* are used once
* only call 1-2 functions
* do not create a meaningful abstraction
* do not improve readability at the call site

Bad:

```python
def load_reference_examples(path: Path) -> list[dict[str, Any]]:
    return build_reference_pool(load_by_ext(str(path)))
```

Good:

```python
ref_pool = build_reference_pool(load_by_ext(str(args.reference_items)))
```

### Do not use wrappers that only unpack and forward

Bad:

```python
def process_one(item, **kwargs):
    idx, record = item
    return _process_one_record(
        source_index=idx,
        item=record,
        temperature=kwargs["temperature"],
        top_p=kwargs["top_p"],
    )
```

If the worker is the real thing, make the worker the real thing.

### One-line code is allowed when it truly holds one idea

Allowed:

```python
if not rows: return []
if path.suffix == ".jsonl": return load_by_ext(str(path))
print(out[0] if len(out) == 1 else f"completed={len(out)}")
```

Not allowed when the line hides complexity, side effects, or multiple mental steps.

### Prefer flat control flow

Use early returns and direct branches to avoid pyramids.

Good:

```python
def load_ds(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        if path.suffix == ".jsonl": return load_by_ext(str(path))
        raise ValueError(f"Unsupported input format: {path}")

    if (path / "dataset_info.json").exists() and (path / "state.json").exists():
        return load_from_disk(str(path))

    raise ValueError(f"Unsupported input directory format: {path}")
```

### Local density, global clarity

Compact local code is good.

But module structure, function boundaries, and data flow must stay obvious.

Dense is good when it reveals pattern.
Dense is bad when it hides intent.

---

## Naming

### General naming rules

* Use standard Python casing
* Follow local project conventions
* Prefer domain terms over explanatory prose
* Prefer familiar abbreviations in-context
* Do not invent obscure abbreviations

### Good short names

Use these freely in local scope when obvious:

* `n` = count / length
* `i` = index
* `k`, `v` = dict key/value
* `x`, `y` = math / tensor / generic transform values
* `cfg` = config
* `msg` = message
* `idx` = index
* `tmp` = temporary
* `out` = output
* `src` = source
* `ds` = dataset
* `ref_pool` = reference pool

### Bad names

Avoid:

* verbose names that merely restate type
* names that carry no concept
* names chosen only to satisfy a generic style aesthetic

Bad:

* `input_dataset_path_value`
* `processed_result_item`
* `helper_function_for_processing`

### Keep important domain names explicit

Do not over-shorten important domain concepts.

Keep names like:

* `reference_pool`
* `quality_flags`
* `enable_thinking`

unless the local context makes a shorter form plainly better.

---

## Functions

### A function must earn its boundary

Create a function when it:

* names a real concept
* hides ugly but necessary detail
* is reused
* makes testing easier
* isolates a risky or changing part

Do not create a function just to reduce indentation or make a caller look cleaner.

### Function length

Prefer short functions.

But do **not** split a 12-line coherent function into 4 weak helpers.

A 20-line function with one clear idea is often better than 5 tiny fake abstractions.

### Function arguments

Prefer explicit parameters over shapeless `**kwargs` bags unless the API genuinely requires it.

Bad:

```python
def process_one(item, **worker_kwargs):
    ...
```

Good:

```python
def rewrite_one(row, *, ref_pool, cfg, temp, top_p, thinking):
    ...
```

### Nested functions

Avoid nested functions unless they are clearly the best tool.

Usually bad when used only to:

* attach a decorator
* hide a cache wrapper
* smuggle state around

---

## Abstractions

### Prefer real abstractions, not ceremonial ones

A real abstraction removes complexity.
A fake abstraction relocates it.

Before extracting a helper, ask:

1. Does this introduce a real concept?
2. Is the call site clearer after extraction?
3. Would inlining make the file easier to read?

If inlining is better, inline it.

### Signs of fake abstraction

* used once
* only forwards arguments
* only renames a one-liner
* exists because a decorator or callback shape forced it
* exists to make another function look clean

---

## Layout

### Keep code narrow enough to read comfortably

Do not explode a simple thought into 12 lines just to satisfy a formatter.

### Use vertical space sparingly

Vertical space is not free.

Use blank lines to separate **ideas**, not every tiny operation.

### Keep related lines adjacent

Group lines that belong to the same thought.

Bad:

```python
quality = detect_quality(res.think_efficient, res.answer)

msgs = [dict(m) for m in item.get("messages", [])[:-1]]

msgs.append({"role": "assistant", "content": content})
```

Better:

```python
quality = detect_quality(res.think_efficient, res.answer)
msgs = [dict(m) for m in item.get("messages", [])[:-1]]
msgs.append({"role": "assistant", "content": content})
```

### Align repeated structure when it helps comparison

Use alignment sparingly and only when it genuinely improves visual comparison.

---

## Comments and docstrings

### Comments explain why, not what

Do not narrate obvious code.

Bad:

```python
# Increment i
i += 1
```

Good:

```python
# Keep original ordering stable for downstream merge logic
rows.sort(key=lambda o: o["source_index"])
```

### Avoid comment debris

Do not leave dead decorators, old approaches, or stale commented-out code unless there is a strong reason.

Bad:

```python
# @memoize(keys=['item'], cache_dir='.cache/opus_cot_rewrites')
```

### Docstrings should be short and useful

Add docstrings when the boundary is meaningful and the function is non-obvious.

Do not add docstrings that merely restate the function name.

---

## Data and state

### Be honest about dependencies

If output depends on config, temperature, version, or external state, do not pretend it depends on one input.

This matters especially for caching.

Bad:

```python
@memoize
def f(item):
    return rewrite(item, temperature=temp, top_p=top_p, cfg=cfg)
```

If the cache key only sees `item`, the cache is dishonest.

### Prefer explicit state boundaries

If caching or global worker state exists, make the boundary obvious.

Good:

```python
_worker_llm: Any | None = None
_worker_cfg: str | None = None
```

This is acceptable when it matches the runtime model and is easy to reason about.

---

## Refactoring policy for the agent

When asked to refactor code, follow this order:

1. Delete fake helpers
2. Flatten control flow
3. Tighten names in local scope
4. Make the main path visible
5. Split only where a real concept appears
6. Remove dead comments and stale scaffolding
7. Keep behavior unchanged unless asked otherwise

Do **not** start by:

* adding classes
* adding layers of indirection
* converting everything into helpers
* introducing patterns for their own sake
* making code longer to look safer

---

## Agent behavior rules

### When writing new code

* write the shortest code that stays easy for a strong reader to scan
* keep the semantic unit visible
* prefer a direct implementation first
* only extract helpers after the code proves it needs them

### When editing existing code

* respect local naming and structure if it is already coherent
* do not apply style mechanically
* improve readability without erasing the file’s rhythm

### When reviewing code

Flag these as likely problems:

* pass-through wrappers
* `**kwargs` used to hide the real contract
* nested functions with no concept
* fake separation of trivial steps
* ceremony-heavy parsing/setup code
* comments that restate the obvious
* stale commented-out code
* caches with dishonest keys

---

## Quick checklist

Before finishing code, check:

* Can the main flow be read top to bottom without jumping around?
* Did every helper earn its name?
* Are any wrappers just forwarding arguments?
* Did I keep important domain terms explicit?
* Did I use short names only in short scope?
* Did I remove boilerplate that adds no concept?
* Did I avoid fake abstraction?
* Did I keep behavior unchanged?

---

## Default bias

When unsure:

* inline first
* shorten second
* extract later

Prefer **honest, compact, readable code** over generic “clean code” theater.
