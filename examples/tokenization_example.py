"""Example: Using LLM encode/decode methods for tokenization.

This example demonstrates how to use the tokenization functionality
in the LLM class to encode text to token IDs and decode token IDs
back to text.
"""

from llm_utils.lm import LLM


def main():
    """Demonstrate encode/decode functionality."""
    # Initialize LLM with your VLLM server
    lm = LLM(base_url='http://localhost:8000/v1')
    
    # Example text
    text = 'The quick brown fox jumps over the lazy dog.'
    print(f'Original text: {text}\n')
    
    # 1. Basic encoding
    print('1. Basic encoding:')
    token_ids = lm.encode(text)
    print(f'   Token IDs: {token_ids}')
    print(f'   Number of tokens: {len(token_ids)}\n')
    
    # 2. Encoding with token strings
    print('2. Encoding with token strings:')
    token_ids, token_strs = lm.encode(text, return_token_strs=True)
    for tid, tstr in zip(token_ids, token_strs):
        print(f'   {tid:6d} -> "{tstr}"')
    print()
    
    # 3. Encoding without special tokens
    print('3. Comparing with/without special tokens:')
    tokens_with = lm.encode(text, add_special_tokens=True)
    tokens_without = lm.encode(text, add_special_tokens=False)
    print(f'   With special tokens:    {len(tokens_with)} tokens')
    print(f'   Without special tokens: {len(tokens_without)} tokens\n')
    
    # 4. Decoding
    print('4. Decoding:')
    decoded = lm.decode(token_ids)
    print(f'   Decoded text: {decoded}\n')
    
    # 5. Practical use case: counting tokens before API call
    print('5. Counting tokens for API calls:')
    long_text = ' '.join(['This is a test sentence.'] * 10)
    token_count = len(lm.encode(long_text))
    print(f'   Text length: {len(long_text)} characters')
    print(f'   Token count: {token_count} tokens')
    print(f'   Avg chars per token: {len(long_text) / token_count:.2f}\n')
    
    # 6. Working with custom token sequences
    print('6. Custom token manipulation:')
    # Encode two sentences
    sent1 = 'Hello world'
    sent2 = 'How are you?'
    tokens1 = lm.encode(sent1, add_special_tokens=False)
    tokens2 = lm.encode(sent2, add_special_tokens=False)
    
    # Combine tokens manually
    combined_tokens = tokens1 + tokens2
    combined_text = lm.decode(combined_tokens)
    print(f'   Sentence 1: {sent1}')
    print(f'   Sentence 2: {sent2}')
    print(f'   Combined (token-level): {combined_text}')


if __name__ == '__main__':
    main()
