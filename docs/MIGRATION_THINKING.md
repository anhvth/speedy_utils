# Qwen3LLM Thinking Guide

`Qwen3LLM` is the only model-specific thinking helper kept in this repository.
It supports prefix continuation for Qwen3-style reasoning traces.

## Usage

```python
from llm_utils import Qwen3LLM

llm = Qwen3LLM(client=8000)

message = llm.generate_with_prefix(
    [{"role": "user", "content": "Solve x^2 + 2x + 1 = 0"}],
    thinking_max_tokens=32,
    content_max_tokens=128,
)
print(message)
```

## Behavior

- `generate_with_prefix()` continues from a partial assistant prefix.
- The default prefix starts a `<think>` block.
- `thinking_max_tokens` caps the reasoning phase.
- `content_max_tokens` caps the final answer phase.

## Notes

- Use `n=1`; prefix continuation is single-path only.
- The returned object is an OpenAI `ChatCompletionMessage`.
- `reasoning_content` is populated when the model emits reasoning text.
