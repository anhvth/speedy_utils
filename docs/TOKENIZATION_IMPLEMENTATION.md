# Tokenization Feature Implementation Summary

## What Was Added

Added tokenization support (encode/decode methods) to the LLM class for converting between text and token IDs.

## Changes Made

### 1. New Mixin: `TokenizationMixin`
**File:** `src/llm_utils/lm/mixins.py`

Added a new mixin class with two methods:
- `encode(text, add_special_tokens=True, return_token_strs=False)` - Convert text to token IDs
- `decode(token_ids)` - Convert token IDs back to text

### 2. Updated LLM Class
**File:** `src/llm_utils/lm/llm.py`

- Added `TokenizationMixin` to the LLM class inheritance
- Imported the new mixin

### 3. Updated Exports
**File:** `src/llm_utils/lm/__init__.py`

- Added `TokenizationMixin` to imports and `__all__`

### 4. Documentation
**File:** `docs/TOKENIZATION.md`

- Comprehensive documentation with API reference
- Usage examples for common scenarios
- Implementation details

### 5. Example Script
**File:** `examples/tokenization_example.py`

- Practical examples demonstrating all features
- Shows token counting, manipulation, debugging

### 6. Tests
**File:** `tests/test_tokenization.py`

- Unit tests for encode/decode functionality
- Tests for special tokens handling
- Manual test runner included

## API Endpoints Used

Based on the provided OpenAPI specification, the implementation uses:

1. **POST /tokenize** - Tokenizes text input
   - Accepts: `TokenizeCompletionRequest` with `prompt`, `add_special_tokens`, `return_token_strs`
   - Returns: `tokens` (list of ints) and optionally `token_strs`

2. **POST /detokenize** - Converts token IDs back to text
   - Accepts: `DetokenizeRequest` with `tokens` (list of ints)
   - Returns: `prompt` (string)

## Usage

```python
from llm_utils.lm import LLM

# Initialize
lm = LLM(base_url='http://localhost:8000/v1')

# Encode
token_ids = lm.encode('Hello, world!')

# Decode
text = lm.decode(token_ids)

# With token strings for debugging
token_ids, token_strs = lm.encode('Hello', return_token_strs=True)
```

## Testing

Run tests with:
```bash
# Using pytest
pytest tests/test_tokenization.py

# Manual test
python tests/test_tokenization.py
```

Run example:
```bash
python examples/tokenization_example.py
```

## Requirements

- VLLM server (or compatible) running with tokenization endpoints
- `requests` library (already a dependency)

## Benefits

1. **Token Counting**: Check token count before API calls to manage context windows
2. **Token-Level Manipulation**: Combine/split text at token boundaries
3. **Debugging**: Inspect exact tokenization with `return_token_strs=True`
4. **Consistency**: Use same tokenizer as the model server
5. **No Local Tokenizer**: No need to install transformers or download tokenizer locally
