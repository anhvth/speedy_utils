# Tokenizer Status In The Current LLM API

## Current State

The current public `LLM` API in this repository does **not** expose generic
`encode()` or `decode()` methods.

Older docs and examples in this repo used to describe a tokenization mixin and
server-side `/tokenize` / `/detokenize` flows. That does not match the current
source tree anymore.

## What Is Public Today

The stable public LLM entry points are:

- `LLM.chat_completion()`
- `LLM.generate()`
- `LLM.pydantic_parse()`
- `LLM.inspect_history()`
- `Qwen3LLM.chat_completion()`
- `Qwen3LLM.complete_until()`
- `Qwen3LLM.complete_reasoning()`
- `Qwen3LLM.complete_content()`

## Raw Prompt Continuation

If you want low-level prompt continuation, use `generate()`.

```python
from llm_utils import LLM

lm = LLM(client=8000)
result = lm.generate("Write a haiku about coding:", max_tokens=50)
print(result.text)
```

Current behavior:

- input is a string prompt
- return value is a `CompletionChoice`-like object
- `n=1` only

## Structured Output

If you want structured parsing, use `pydantic_parse()`.

```python
from pydantic import BaseModel
from llm_utils import LLM


class Answer(BaseModel):
    answer: str
    confidence: float


lm = LLM(client=8000)
parsed = lm.pydantic_parse(
    "Return JSON with answer='blue' and confidence=0.9.",
    response_model=Answer,
)
print(parsed.model_dump())
```

## Qwen3 Tokenizer Usage

`Qwen3LLM` still has tokenizer-related behavior, but it is **internal** rather
than a generic public tokenization API.

The current implementation uses a tokenizer only to build completion prompts for
Qwen3-style prefix continuation when that tokenizer is available.

```python
from llm_utils import Qwen3LLM

llm = Qwen3LLM(client=8000)
message = llm.chat_completion(
    [{"role": "user", "content": "Solve x^2 + 2x + 1 = 0"}],
    thinking_max_tokens=32,
    content_max_tokens=128,
)
print(message.content)
```

If `transformers` is unavailable, `Qwen3LLM` falls back to text-based prompt
rendering instead of exposing a public `encode()`/`decode()` API.

## Migration From Older Docs

If you still have code that assumes the old tokenization docs were current:

- replace `lm.encode(...)` / `lm.decode(...)` usage with backend-specific tooling
  outside this package
- replace dict-style `generate()` handling with attribute access like
  `result.text`
- replace token-id input examples with plain string prompts

## What This Means For Documentation

For the current branch, treat tokenizer support as:

- internal for `Qwen3LLM`
- not part of the generic `LLM` public surface
- not a supported basis for examples that promise token counting or token-id
  prompt input through `LLM`
