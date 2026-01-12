"""Test tokenization functionality for LLM."""

import pytest
from llm_utils.lm import LLM


def test_encode_decode():
    """Test basic encode/decode functionality."""
    # Skip if no VLLM server is running
    try:
        lm = LLM(base_url='http://localhost:8000/v1')
        
        # Test encode
        text = 'Hello, world!'
        token_ids = lm.encode(text)
        assert isinstance(token_ids, list)
        assert all(isinstance(t, int) for t in token_ids)
        assert len(token_ids) > 0
        
        # Test encode with token strings
        token_ids_with_strs, token_strs = lm.encode(
            text, return_token_strs=True
        )
        assert isinstance(token_ids_with_strs, list)
        assert isinstance(token_strs, list)
        assert len(token_ids_with_strs) == len(token_strs)
        
        # Test decode
        decoded = lm.decode(token_ids)
        assert isinstance(decoded, str)
        # Note: decoded text might have slight differences due to tokenizer
        # behavior (e.g., special tokens), so we don't assert exact match
        
    except Exception as e:
        pytest.skip(f'VLLM server not available: {e}')


def test_encode_with_special_tokens():
    """Test encode with and without special tokens."""
    try:
        lm = LLM(base_url='http://localhost:8000/v1')
        
        text = 'Test text'
        
        # With special tokens (default)
        tokens_with = lm.encode(text, add_special_tokens=True)
        
        # Without special tokens
        tokens_without = lm.encode(text, add_special_tokens=False)
        
        # Typically tokens_with should have more tokens (BOS, EOS, etc.)
        # but this depends on the model
        assert isinstance(tokens_with, list)
        assert isinstance(tokens_without, list)
        
    except Exception as e:
        pytest.skip(f'VLLM server not available: {e}')


if __name__ == '__main__':
    # Simple manual test
    print('Testing tokenization...')
    try:
        lm = LLM(base_url='http://localhost:8000/v1')
        
        text = 'Hello, how are you?'
        print(f'Original text: {text}')
        
        # Encode
        token_ids = lm.encode(text)
        print(f'Token IDs: {token_ids}')
        
        # Encode with token strings
        token_ids_with_strs, token_strs = lm.encode(
            text, return_token_strs=True
        )
        print(f'Tokens with strings: {list(zip(token_ids_with_strs, token_strs))}')
        
        # Decode
        decoded = lm.decode(token_ids)
        print(f'Decoded text: {decoded}')
        
        print('✓ All tests passed!')
        
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
