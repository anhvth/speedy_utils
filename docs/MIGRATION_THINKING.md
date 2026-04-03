# Qwen3LLM Thinking Guide

`Qwen3LLM` is the only model-specific thinking helper kept in this repository.
It supports prefix continuation for Qwen3-style reasoning traces.

## Usage

```python
from llm_utils import Qwen3LLM

llm = Qwen3LLM(client=8000)

message = llm.chat_completion(
    [{"role": "user", "content": "Solve x^2 + 2x + 1 = 0"}],
    thinking_max_tokens=32,
    content_max_tokens=128,
)
print(message)
```

## Behavior

- `chat_completion()` continues from a partial assistant prefix.
- The default prefix starts a `<think>` block.
- `thinking_max_tokens` caps the reasoning phase.
- `content_max_tokens` caps the final answer phase.

## Notes

- Use `n=1`; prefix continuation is single-path only.
- The returned object is an OpenAI `ChatCompletionMessage`.
- The intermediate prefix state is internal; the returned value is an OpenAI
  `ChatCompletionMessage`.
- The returned message includes `call_count`, which is `1` when the model
  finishes in one staged call and `2` when it needs a reasoning call plus a
  follow-up content call.
