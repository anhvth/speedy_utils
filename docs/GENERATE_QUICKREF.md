# Quick Reference: `LLM.generate()`

## Summary

`LLM.generate()` is the raw prompt-continuation API in `llm_utils`. It calls the
OpenAI-compatible **completions** endpoint and returns a
`CompletionChoice`-like object.

Use it when you already have a plain prompt string and want the backend to
continue that prompt.

## Current Signature

```python
lm.generate(
    prompt: str,
    *,
    cache: bool | None = None,
    enable_thinking: bool | None = None,
    **runtime_kwargs,
) -> CompletionChoice
```

Important current constraints:

- `prompt` must be a string.
- `n=1` only. Multi-choice generation is rejected by the public API.
- constructor defaults such as `model`, `max_tokens`, `temperature`, and `top_p`
  are merged with `runtime_kwargs` at call time.

## Which Method To Use

- `generate()`: continue a raw prompt through the completions API.
- `chat_completion()`: generate the next assistant turn from chat messages.
- `pydantic_parse()`: parse structured output from the chat completions API.
- `llm(...)`: convenience wrapper for the chat path and structured-output path.

## Common Examples

### Simple prompt continuation

```python
from llm_utils import LLM

lm = LLM(client=8000)
result = lm.generate("Write a haiku about coding:", max_tokens=50)
print(result.text)
```

### Temperature and nucleus sampling

```python
result = lm.generate(
    "The best way to learn programming is",
    max_tokens=40,
    temperature=0.8,
    top_p=0.95,
)
print(result.text)
```

### Stop sequences

```python
result = lm.generate(
    "Ingredients for cookies:\n-",
    max_tokens=200,
    stop=["\n\n", "Instructions:"],
)
print(result.text)
```

### Reproducible calls

```python
result1 = lm.generate("Random number:", max_tokens=10, temperature=0.8, seed=42)
result2 = lm.generate("Random number:", max_tokens=10, temperature=0.8, seed=42)
print(result1.text == result2.text)
```

### Backend-specific metadata

Some backends return extra fields such as `token_ids`, `prompt_token_ids`, or
`prompt_logprobs`. The current implementation preserves those when they are
present on the provider response.

```python
result = lm.generate("prompt", max_tokens=0, echo=True, logprobs=1, temperature=0)
print(result.text)
print(getattr(result, "prompt_logprobs", None))
print(getattr(result, "token_ids", None))
```

## Common Runtime Kwargs

`generate()` forwards provider kwargs to `client.completions.create(...)` after
merging them with constructor defaults.

Common examples:

- `max_tokens`
- `temperature`
- `top_p`
- `stop`
- `seed`
- `echo`
- `logprobs`
- `presence_penalty`
- `frequency_penalty`

Support for a given kwarg still depends on the backend you are calling.

## Return Shape

`generate()` returns a `CompletionChoice`-like object, not a dict.

Typical fields:

- `result.text`
- `result.finish_reason`
- `result.usage` when the backend provides usage data

The call also records a simple conversation history internally so you can inspect
it afterwards:

```python
lm.generate("Hello", max_tokens=5)
print(lm.inspect_history())
```

## Not Supported In The Current Public API

These older examples no longer match the code in this branch:

- token-id input to `generate()`
- dict-style access like `result["text"]`
- multi-output generation with `n > 1`
- generic `lm.encode()` / `lm.decode()` methods on `LLM`
