# Quick Reference: generate() Method

## Summary

The `LLM.generate()` method provides a HuggingFace Transformers-style interface for low-level text generation, working directly with token IDs.

## Basic Signature

```python
lm.generate(
    input_context: str | list[int],
    max_tokens=512,
    temperature=1.0,
    n=1,
    **kwargs
) -> dict | list[dict]
```

## Common Use Cases

### 1. Simple Generation
```python
result = lm.generate('Hello world', max_tokens=50)
print(result['text'])
```

### 2. From Token IDs
```python
token_ids = lm.encode('Hello')
result = lm.generate(token_ids, max_tokens=50, return_token_ids=True)
```

### 3. Multiple Outputs
```python
results = lm.generate('Start:', max_tokens=30, n=5)  # 5 different completions
```

### 4. Temperature Control
```python
# Deterministic (low temp)
result = lm.generate(prompt, temperature=0.1)

# Creative (high temp)
result = lm.generate(prompt, temperature=1.5)
```

### 5. Advanced Sampling
```python
result = lm.generate(
    prompt,
    temperature=0.8,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.2,
)
```

### 6. Stop Sequences
```python
result = lm.generate(
    'List:\n1.',
    max_tokens=200,
    stop=['\n\n', 'End'],
)
```

### 7. Reproducible
```python
result = lm.generate(prompt, seed=42)  # Same seed = same output
```

## Parameter Guide

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_context` | str \| list[int] | required | Text or token IDs |
| `max_tokens` | int | 512 | Max tokens to generate |
| `temperature` | float | 1.0 | Randomness (0.0-2.0) |
| `top_p` | float | 1.0 | Nucleus sampling |
| `top_k` | int | -1 | Top-k sampling (-1=off) |
| `n` | int | 1 | Number of completions |
| `stop` | str \| list | None | Stop sequences |
| `seed` | int | None | Random seed |
| `repetition_penalty` | float | 1.0 | Repeat penalty (1.0=off) |
| `return_token_ids` | bool | False | Include token IDs |
| `return_text` | bool | True | Include text |

## Return Format

Single generation (n=1):
```python
{
    'text': 'generated text...',
    'token_ids': [1, 2, 3, ...],  # if return_token_ids=True
    'finish_reason': 'length',
    '_raw_response': {...}
}
```

Multiple generations (n>1):
```python
[
    {'text': '...', 'finish_reason': '...'},
    {'text': '...', 'finish_reason': '...'},
    ...
]
```

## Comparison to HuggingFace

| HuggingFace | llm_utils | Notes |
|-------------|-----------|-------|
| `model.generate(input_ids=...)` | `lm.generate(token_ids)` | Same concept |
| `max_length` | `max_tokens` | Different naming |
| `num_return_sequences` | `n` | Different naming |
| `do_sample=True` | `temperature > 0` | Auto-enabled |
| `num_beams` | N/A | Not supported |

## Tips

1. **Token Counting**: Use `len(lm.encode(text))` to count tokens before generating
2. **Reproducibility**: Set `seed` for deterministic output
3. **Quality vs Speed**: Lower temperature for quality, higher for creativity
4. **Stop Early**: Use `stop` sequences to control output format
5. **Debug**: Check `result['_raw_response']` for full API response
