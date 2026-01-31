---
name: 'edit-llm-inference-style'
description: 'Guide for adjusting speedy_utils LLM inference style, including chat templating, <think> prefixes, stop sequences, and boxed-answer handling.'
---

# Edit LLM Inference Style (speedy_utils)

Use this skill to standardize or modify how speedy_utils builds prompts and consumes generation outputs. It focuses on chat templating, reasoning-style prefixes, and safe stopping rules for structured answer extraction.

## When to Use This Skill

Use this skill when you need to:
- Insert or enforce a reasoning prefix (for example, `<think>\n`).
- Switch a flow to `LLM.generate()` instead of chat completion helpers.
- Apply a tokenizer chat template before generating.
- Stop generations on boxed answers (`\boxed{}`) or `<|im_end|>` tokens.
- Normalize outputs before evaluation (for example, GSM8K or math tasks).

## Prerequisites

- A model-backed tokenizer available via `transformers.AutoTokenizer`.
- A speedy_utils `LLM` instance configured to point at the correct backend.

## Core Capabilities

### 1) Build a Chat-Templated Prompt

Use the tokenizer to format messages and append a generation prefix:

```python
from transformers import AutoTokenizer

TOKENIZER_NAME = "Qwen/Qwen3-4B"
THINK_PREFIX = "<think>\n"

messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": user_text},
]

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
prompt = f"{prompt}{THINK_PREFIX}"
```

### 2) Generate with Low-Level `LLM.generate`

`LLM.generate()` routes to `/inference/v1/generate`, which works directly with token IDs or raw text.

```python
result = llm.generate(
    prompt,
    max_tokens=512,
    temperature=temperature,
    stop=["<|im_end|>"],
)
text = result["text"]
```

### 3) Stop on Boxed Answers or `<|im_end|>`

Use a post-processing step to truncate output when a boxed answer appears:

```python
import re

BOXED_PATTERN = re.compile(r"\\boxed\{.*?\}", re.DOTALL)


def truncate_completion(text: str) -> str:
    end_positions = []
    im_end_idx = text.find("<|im_end|>")
    if im_end_idx != -1:
        end_positions.append(im_end_idx)
    boxed_match = BOXED_PATTERN.search(text)
    if boxed_match:
        end_positions.append(boxed_match.end())
    if not end_positions:
        return text
    return text[: min(end_positions)]
```

### 4) Extract a Final Number Safely

Prefer the boxed span when available:

```python
import re

BOXED_PATTERN = re.compile(r"\\boxed\{.*?\}", re.DOTALL)


def extract_final_number(text: str) -> str:
    boxed_match = BOXED_PATTERN.search(text)
    if boxed_match:
        text = boxed_match.group(0)
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else ""
```

## Guidelines

- Always apply the tokenizer chat template when the model expects chat-formatted input.
- Use a consistent reasoning prefix (`<think>\n`) to reduce formatting drift.
- Prefer `LLM.generate()` for low-level control of stop sequences and token handling.
- Post-truncate output to avoid extra text after boxed answers or special tokens.
- Keep evaluation logic (parsing + correctness) in the eval script, not in the LLM client.

## Common Patterns

### Pattern: GSM8K-style Generation

```python
prompt = format_prompt(question, tokenizer)
raw_output = llm.generate(
    prompt,
    max_tokens=512,
    temperature=temperature,
    stop=["<|im_end|>"],
)["text"]
raw_output = truncate_completion(raw_output)
pred = extract_final_number(raw_output)
```

## Related Files

- `src/llm_utils/lm/mixins.py`: `LLM.generate()` implementation.
- `src/llm_utils/chat_format/transform.py`: Chat templating utility.
- `docs/GENERATE_QUICKREF.md`: `generate()` parameters and response format.
