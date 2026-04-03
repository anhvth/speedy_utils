# Qwen3 Debug Runner

This script is a small local helper for exercising `Qwen3LLM.generate_with_prefix()`.
It is useful when you want to control how many tokens are spent on reasoning versus the final answer.

## What It Does

`debug/test_qwen3_generate.py` starts a Qwen3 reasoning client and calls:

```python
llm.generate_with_prefix(
    [{"role": "user", "content": "hi"}],
    thinking_max_tokens=10,
    content_max_tokens=1000,
)
```

The helper:

- seeds a partial assistant prefix for Qwen3-style reasoning
- lets you cap the thinking phase separately from the answer phase
- returns an OpenAI-style `ChatCompletionMessage`

## Requirements

- A local OpenAI-compatible backend must be running on port `8001`
- `llm_utils` must be importable from the repo environment
- `transformers` must be available, because the helper uses the Qwen3 tokenizer template

The current script uses:

```python
llm = Qwen3LLM(client=8001)
```

So `8001` is the expected backend port unless you change it in the script.

## How To Use It

1. Start your backend on port `8001`
2. Edit `debug/test_qwen3_generate.py` if you want a different prompt or token budget
3. Run the script with your normal Python entry point for the repo

Example workflow:

```python
from llm_utils import Qwen3LLM

llm = Qwen3LLM(client=8001)
```

Then adjust the budgets:

- `thinking_max_tokens` controls the reasoning budget
- `content_max_tokens` controls the final answer budget

If the model stops too early during reasoning, increase `thinking_max_tokens`.
If the final answer is too short, increase `content_max_tokens`.

## What The Output Means

The return value is a `ChatCompletionMessage` with:

- `role="assistant"`
- `content` for the final visible answer
- `reasoning_content` when the model produced reasoning text

This makes it convenient for conversational debugging and for inspecting how much reasoning the model used before answering.

## Limitation

This helper is good for normal conversation and reasoning control, but it does **not** support tool calling.

In particular:

- generated text is not parsed into tool calls
- tool-call objects are not emitted
- tool execution loops are not part of this runner

If you need tool calling, use the higher-level chat flow instead of this debug script.

## Practical Tweaks

- Use a small `thinking_max_tokens` to force short reasoning traces
- Use a larger `content_max_tokens` when you want long answers
- Change the initial user message in the script to test prompts that are closer to your real workload
- Keep `n=1`; this helper is designed for a single continuation path
