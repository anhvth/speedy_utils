# Qwen3 Debug Runner

This helper exercises `Qwen3LLM.chat_completion()` with separate thinking and
answer token budgets.

## What The Script Does

`debug/test_qwen3_generate.py` currently runs:

```python
from llm_utils import Qwen3LLM

llm = Qwen3LLM(client=8001)
msg = llm.chat_completion(
    [{"role": "user", "content": "hi"}],
    thinking_max_tokens=10,
    content_max_tokens=1000,
)
```

So the script is a small local runner for inspecting how Qwen3 reasoning behaves
with a short thinking budget and a larger answer budget.

## Requirements

- an OpenAI-compatible backend should be running on port `8001`
- the repo environment should be active so `llm_utils` imports cleanly
- `transformers` is optional

When `transformers` is available, `Qwen3LLM` uses tokenizer-backed prompt
rendering for Qwen-style prompt construction. Without it, the code falls back to
text prompt rendering.

## Expected Output Shape

The returned object is an OpenAI-style `ChatCompletionMessage`.

It can include:

- `content` for the visible answer
- `reasoning_content` when the backend emits reasoning text
- `usage` when the backend returns token accounting
- `call_count` for how many staged prefix calls were needed internally

## Practical Tweaks

- lower `thinking_max_tokens` to force shorter reasoning traces
- raise `content_max_tokens` when you want longer final answers
- change `client=8001` if your backend is on another port
- keep `n=1`; the Qwen3 prefix flow is single-path only
