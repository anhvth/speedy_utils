# Qwen3LLM Thinking Guide

`Qwen3LLM` is the only model-specific thinking helper kept in this repository.
It supports prefix continuation for Qwen3-style reasoning traces.
It also includes a raw prefix-conditioned helper for custom staged tag flows.

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
```

## Behavior

- `chat_completion()` continues from a partial assistant prefix.
- The default prefix starts a `<think>` block.
- `thinking_max_tokens` caps the reasoning phase.
- `content_max_tokens` caps the final answer phase.
- `complete_until()` continues from a raw assistant prefix and stops on custom
  tag boundaries.
- The returned `assistant_prompt_prefix` includes the generated text and the
  matched stop token when generation ends on a stop sequence.

## Notes

- Use `n=1`; prefix continuation is single-path only.
- The returned object is an OpenAI `ChatCompletionMessage`.
- The intermediate prefix state is internal; the returned value is an OpenAI
  `ChatCompletionMessage`.
- The returned message includes `call_count`, which is `1` when the model
  finishes in one staged call and `2` when it needs a reasoning call plus a
  follow-up content call.
