# Qwen3LLM Thinking Guide

`Qwen3LLM` is the model-specific reasoning helper kept in this repository.
It supports staged assistant-prefix continuation for Qwen3-style reasoning
traces and custom tagged flows.

## Basic Usage

```python
from llm_utils import Qwen3LLM

llm = Qwen3LLM(client=8000)

message = llm.chat_completion(
    [{"role": "user", "content": "Solve x^2 + 2x + 1 = 0"}],
    thinking_max_tokens=32,
    content_max_tokens=128,
)

print(message.content)
print(getattr(message, "reasoning_content", None))
print(getattr(message, "call_count", None))
```

## Custom Prefix Flows

Use `complete_until()` when you want to generate a custom tagged block and stop
at a specific closing tag instead of the built-in `<think>...</think>` flow.

```python
from llm_utils import Qwen3LLM

llm = Qwen3LLM(client=8000)

memory_state = llm.complete_until(
    [{"role": "user", "content": "Plan the answer in stages"}],
    "<memory>",
    stop="</memory>",
    max_tokens=128,
)

think_state = llm.complete_until(
    [{"role": "user", "content": "Plan the answer in stages"}],
    memory_state.assistant_prompt_prefix + "\n<think_efficient>",
    stop="</think_efficient>",
    max_tokens=256,
)

final_state = llm.complete_until(
    [{"role": "user", "content": "Plan the answer in stages"}],
    think_state.assistant_prompt_prefix,
    stop="<|im_end|>",
    max_tokens=256,
)

print(final_state.generated_text)
print(final_state.stop)
print(final_state.call_count)
```

## Public Return Types

Current public behavior is:

- `chat_completion()` returns an OpenAI-style `ChatCompletionMessage`
- `complete_until()` returns a continuation-state object
- `complete_reasoning()` returns a reasoning prefix state for `complete_content()`
- `complete_content()` returns an OpenAI-style `ChatCompletionMessage`

For `chat_completion()`, the returned message can include dynamic attributes such
as:

- `reasoning_content`
- `usage`
- `call_count`

For `complete_until()`, the returned state includes:

- `assistant_prompt_prefix`
- `generated_text`
- `stop`
- `stop_reason`
- `call_count`
- `usage`

## Behavior Notes

- the default staged flow uses a `<think>...</think>` block
- `thinking_max_tokens` caps the reasoning phase
- `content_max_tokens` caps the visible answer phase
- `complete_until()` continues from a raw assistant prefix and can include the
  matched stop token in the stored prefix
- prefix continuation is single-path only: use `n=1`

## Tokenizer Behavior

`Qwen3LLM` prefers tokenizer-backed prompt rendering when the Qwen tokenizer is
available.

If `transformers` is unavailable, the implementation falls back to text-based
chat prompt rendering rather than failing the whole chat path.

## When To Use Which Method

- use `chat_completion()` for the normal Qwen3 reasoning + answer flow
- use `complete_until()` for custom staged tags such as `<memory>` or
  `<think_efficient>`
- use `complete_reasoning()` and `complete_content()` when you want the built-in
  think/content split but need to control the two phases separately
