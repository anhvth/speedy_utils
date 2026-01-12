# Tokenization Support in LLM

The `LLM` class now includes built-in tokenization support through the `TokenizationMixin`, providing `encode` and `decode` methods to work with token IDs.

## Features

- **encode()**: Convert text to token IDs
- **decode()**: Convert token IDs back to text
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

## Requirements

The tokenization functionality requires a VLLM server (or compatible API) that implements:
- `/tokenize` endpoint (accepts `TokenizeCompletionRequest`)
- `/detokenize` endpoint (accepts `DetokenizeRequest`)

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

- Example script: `examples/tokenization_example.py`
- Tests: `tests/test_tokenization.py`
- API specification: See OpenAPI schema for endpoint details
