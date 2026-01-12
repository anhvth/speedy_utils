# Tokenization and Generation Support in LLM

The `LLM` class now includes built-in tokenization and low-level generation support through the `TokenizationMixin`, providing methods similar to HuggingFace Transformers.

## Features

- **encode()**: Convert text to token IDs
- **decode()**: Convert token IDs back to text
- **generate()**: HuggingFace-style generation with token-level control
- Support for special tokens (BOS, EOS, etc.)
- Optional token string output for debugging

## API Reference

### encode()

```python
def encode(
    self,
    text: str,
    *,
    add_special_tokens: bool = True,
    return_token_strs: bool = False,
) -> list[int] | tuple[list[int], list[str]]
```

**Parameters:**
- `text` (str): Text to tokenize
- `add_special_tokens` (bool): Whether to add special tokens like BOS/EOS (default: True)
- `return_token_strs` (bool): If True, also return token strings (default: False)

**Returns:**
- `list[int]`: Token IDs (if `return_token_strs=False`)
- `tuple[list[int], list[str]]`: Token IDs and token strings (if `return_token_strs=True`)

### decode()

```python
def decode(
    self,
    token_ids: list[int],
) -> str
```

**Parameters:**
- `token_ids` (list[int]): List of token IDs to decode

**Returns:**
- `str`: Decoded text

### generate()

```python
def generate(
    self,
    input_context: str | list[int],
    *,
    max_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    n: int = 1,
    stop: str | list[str] | None = None,
    stop_token_ids: list[int] | None = None,
    seed: int | None = None,
    return_token_ids: bool = False,
    return_text: bool = True,
    **kwargs,
) -> dict[str, Any] | list[dict[str, Any]]
```

**Parameters:**
- `input_context` (str | list[int]): Input as text or token IDs
- `max_tokens` (int): Maximum tokens to generate (default: 512)
- `temperature` (float): Sampling temperature, higher = more random (default: 1.0)
- `top_p` (float): Nucleus sampling threshold (default: 1.0)
- `top_k` (int): Top-k sampling, -1 to disable (default: -1)
- `min_p` (float): Minimum probability threshold (default: 0.0)
- `repetition_penalty` (float): Penalty for repeating tokens (default: 1.0)
- `presence_penalty` (float): Presence penalty for diversity (default: 0.0)
- `frequency_penalty` (float): Frequency penalty for diversity (default: 0.0)
- `n` (int): Number of sequences to generate (default: 1)
- `stop` (str | list[str]): Stop sequences
- `stop_token_ids` (list[int]): Token IDs to stop at
- `seed` (int): Random seed for reproducibility
- `return_token_ids` (bool): Include token IDs in output (default: False)
- `return_text` (bool): Include decoded text in output (default: True)

**Returns:**
- `dict`: Generation result with 'text', 'token_ids', 'finish_reason' (if n=1)
- `list[dict]`: List of generation results (if n>1)

## Usage Examples

### Basic Encoding/Decoding

```python
from llm_utils.lm import LLM

lm = LLM(base_url='http://localhost:8000/v1')

# Encode text to token IDs
text = 'Hello, world!'
token_ids = lm.encode(text)
print(token_ids)  # [123, 456, 789, ...]

# Decode token IDs back to text
decoded = lm.decode(token_ids)
print(decoded)  # 'Hello, world!'
```

### Getting Token Strings

Useful for debugging and understanding tokenization:

```python
# Get both token IDs and their string representations
token_ids, token_strs = lm.encode('Hello world', return_token_strs=True)

for tid, tstr in zip(token_ids, token_strs):
    print(f'{tid:6d} -> "{tstr}"')
# Output:
#    123 -> "Hello"
#    456 -> " world"
```

### Counting Tokens

Count tokens before making API calls to manage context windows:

```python
text = 'A very long document...'
token_count = len(lm.encode(text))
print(f'This text uses {token_count} tokens')

# Check if it fits in model's context window
MAX_TOKENS = 4096
if token_count > MAX_TOKENS:
    print('Text is too long!')
```

### Working Without Special Tokens

```python
# Without special tokens (useful for token manipulation)
tokens_clean = lm.encode('Hello', add_special_tokens=False)

# With special tokens (default)
tokens_with_special = lm.encode('Hello', add_special_tokens=True)

print(f'Clean: {len(tokens_clean)} tokens')
print(f'With special: {len(tokens_with_special)} tokens')
```

### Token-Level Text Manipulation

```python
# Combine texts at token level
sent1_tokens = lm.encode('Hello', add_special_tokens=False)
sent2_tokens = lm.encode('world', add_special_tokens=False)

# Manually combine
combined = sent1_tokens + sent2_tokens
result = lm.decode(combined)
print(result)  # 'Helloworld'
```

## HuggingFace-Style Generation

The `generate()` method provides a low-level interface similar to HuggingFace Transformers:

### Basic Text Generation

```python
result = lm.generate(
    'The capital of France is',
    max_tokens=50,
    temperature=0.7,
)
print(result['text'])  # ' Paris.'
```

### Working with Token IDs

```python
# Encode input
input_ids = lm.encode('Hello, how are you?')

# Generate from token IDs
result = lm.generate(
    input_ids,
    max_tokens=30,
    temperature=0.8,
    return_token_ids=True,
)
print(result['text'])  # Generated text
print(result['token_ids'])  # Generated token IDs
```

### Multiple Generations

```python
# Generate multiple completions (like num_return_sequences)
results = lm.generate(
    'Once upon a time',
    max_tokens=30,
    temperature=0.9,
    n=3,  # Generate 3 different completions
)
for i, result in enumerate(results, 1):
    print(f'{i}. {result["text"]}')
```

### Advanced Sampling

```python
# Use top-k, top-p, and repetition penalty
result = lm.generate(
    'The best programming language is',
    max_tokens=40,
    temperature=0.8,
    top_k=50,  # Only sample from top 50 tokens
    top_p=0.95,  # Nucleus sampling
    repetition_penalty=1.2,  # Reduce repetition
)
```

### Stop Sequences

```python
# Stop generation at specific sequences
result = lm.generate(
    'List three colors:\n1.',
    max_tokens=100,
    stop=['\n4.', '\n\n'],  # Stop at these sequences
)
```

### Reproducible Generation

```python
# Use seed for reproducibility
result1 = lm.generate('Random:', max_tokens=10, temperature=0.8, seed=42)
result2 = lm.generate('Random:', max_tokens=10, temperature=0.8, seed=42)
assert result1['text'] == result2['text']  # Same output!
```

## Token Budget Management

Manage context windows by counting tokens before generation:

```python
# Check token count before generating
long_prompt = 'Explain quantum computing: ' * 10
prompt_tokens = lm.encode(long_prompt)

MAX_CONTEXT = 4096
available_for_generation = MAX_CONTEXT - len(prompt_tokens)

result = lm.generate(
    long_prompt,
    max_tokens=min(100, available_for_generation),
)
generated_tokens = lm.encode(result['text'])
print(f'Total tokens: {len(prompt_tokens) + len(generated_tokens)}')
```

## Requirements

The tokenization and generation functionality requires a VLLM server (or compatible API) that implements:
- `/tokenize` endpoint (accepts `TokenizeCompletionRequest`)
- `/detokenize` endpoint (accepts `DetokenizeRequest`)
- `/inference/v1/generate` endpoint (accepts `GenerateRequest`)

## Implementation Details

The `TokenizationMixin` is automatically included in the `LLM` class. It uses the model's base URL to make HTTP requests to the tokenization endpoints.

The mixin can be used standalone if needed:

```python
from llm_utils.lm.mixins import TokenizationMixin

class MyCustomLM(TokenizationMixin):
    def __init__(self, base_url):
        self.client = MOpenAI(base_url=base_url, api_key='abc')
```

## See Also

- Tokenization example: `examples/tokenization_example.py`
- Generation example: `examples/generate_example.py`
- Tests: `tests/test_tokenization.py`
- API specification: See OpenAPI schema for endpoint details
