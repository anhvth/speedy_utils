---
name: 'edit-llm-inference-style'
description: 'Guide for adjusting Qwen3LLM inference style in speedy_utils, including staged complete_until flows, <think> prefixes, stop sequences, and boxed-answer handling.'
---

# Edit LLM Inference Style (Qwen3LLM)

Use this skill when changing how Qwen3-style models generate reasoning and final answers in speedy_utils. The current API is centered on `Qwen3LLM`, especially `complete_until()`, `complete_reasoning()`, `complete_content()`, and `chat_completion()`.

## When to Use This Skill

Use this skill when you need to:
- Insert or enforce a reasoning prefix such as `<think>\n`.
- Split generation into multiple prefix-conditioned steps.
- Stop reasoning on `</think>` and stop final content on `<|im_end|>`.
- Force boxed-answer style completions for MCQ or math evaluation.
- Change how reasoning truncation is handled when the thinking budget is exhausted.

## Prerequisites

- A `Qwen3LLM` instance pointed at a compatible backend.
- Familiarity with assistant-body prefixes: `complete_until()` works on assistant text, not on raw prompt strings.

## Core Capabilities

### 1) Use `Qwen3LLM` as the Primary API

Prefer `Qwen3LLM` over low-level `LLM.generate()` when editing Qwen3 reasoning flows:

```python
from llm_utils import Qwen3LLM

llm = Qwen3LLM(
    client="http://localhost:8001/v1",
    thinking_max_tokens=4096,
    content_max_tokens=512,
    temperature=0.8,
)
```

### 2) Stage Generation with `complete_until()`

`complete_until()` is the current low-level primitive for prefix-conditioned continuation.

It accepts:
- `messages`: the chat messages
- `assistant_prompt_prefix`: the current assistant-body prefix or a previous continuation state
- `stop`: one or more stop sequences
- `max_tokens`: the budget for that step

Example: generate reasoning until `</think>`:

```python
from llm_utils.lm.llm_qwen3 import THINK_END

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt_text},
]

reasoning_state = llm.complete_until(
    messages,
    assistant_prompt_prefix="<think>\n",
    stop=THINK_END,
    max_tokens=llm.default_thinking_max_tokens,
)
```

### 3) Artificially Close Reasoning When Budget Is Exhausted

If reasoning stops because the budget is exhausted rather than because `</think>` was emitted, inject the closing tag yourself before continuing.

```python
from llm_utils.lm.llm_qwen3 import THINK_END

if reasoning_state.stop != THINK_END:
    reasoning_state = reasoning_state.inject(f"\n{THINK_END}")
```

### 4) Force a Single-Token Boxed Answer

For MCQ evaluation, append `\n\nboxed{` and generate a single answer token:

```python
from llm_utils.lm.llm_qwen3 import ASSISTANT_END

answer_state = llm.complete_until(
    messages,
    reasoning_state.inject("\n\nboxed{"),
    stop=ASSISTANT_END,
    max_tokens=1,
    include_stop_in_prefix=False,
)

text = answer_state.assistant_prompt_prefix.strip()
```

This yields assistant text like:

```text
<think>
...
</think>

boxed{C
```

That is acceptable if downstream evaluation falls back to standalone answer letters.

### 5) Use `chat_completion()` Only for the Default Two-Step Chat Flow

`chat_completion()` already composes:
- reasoning generation
- optional synthetic `</think>` closure
- content generation

Use it when you want the default Qwen3 chat behavior, not when you need exact staged control.

### 6) Extract Final Answers Safely

Prefer boxed answers first, then a fallback parser for standalone `A-E` if the final brace is omitted:

```python
import re

BOXED_RE = re.compile(r"\\boxed\{\s*([A-Ea-e])\s*\}")
LETTER_RE = re.compile(r"\b([A-Ea-e])\b")


def extract_answer_letter(text: str) -> str | None:
    boxed_match = BOXED_RE.search(text)
    if boxed_match:
        return boxed_match.group(1).upper()
    letter_match = LETTER_RE.search(text)
    if letter_match:
        return letter_match.group(1).upper()
    return None
```

## Guidelines

- Prefer `Qwen3LLM.complete_until()` for exact staged flows.
- Treat `assistant_prompt_prefix` as assistant-body text, not a full prompt.
- Use `THINK_END` to stop reasoning and `ASSISTANT_END` to stop final content.
- When a reasoning step hits the token cap, explicitly inject `</think>` before continuing.
- Keep answer parsing in evaluation code rather than baking task-specific parsing into the client.

## Common Patterns

### Pattern: MCQ Boxed-Letter Evaluation

```python
reasoning_state = llm.complete_until(
    messages,
    assistant_prompt_prefix="<think>\n",
    stop=THINK_END,
    max_tokens=llm.default_thinking_max_tokens,
)
if reasoning_state.stop != THINK_END:
    reasoning_state = reasoning_state.inject(f"\n{THINK_END}")

answer_state = llm.complete_until(
    messages,
    reasoning_state.inject("\n\nboxed{"),
    stop=ASSISTANT_END,
    max_tokens=1,
    include_stop_in_prefix=False,
)
text = answer_state.assistant_prompt_prefix.strip()
pred = extract_answer_letter(text)
```

## Related Files

- `src/llm_utils/lm/llm_qwen3.py`: `Qwen3LLM`, `complete_until()`, `complete_reasoning()`, `complete_content()`.
- `src/llm_utils/lm/llm.py`: base client health checks and shared client logic.
- `docs/GENERATE_QUICKREF.md`: legacy generate-path reference when you truly need the raw completion API.
