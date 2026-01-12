"""Test the generate method with HuggingFace-style interface."""

from llm_utils.lm import LLM

# Initialize LLM
lm = LLM(client=8000)

print('=' * 60)
print('Test 1: Basic text generation')
print('=' * 60)
result = lm.generate(
    'The capital of France is',
    max_tokens=50,
    temperature=0.7,
)
print(f"Input: 'The capital of France is'")
print(f"Generated: {result.get('text', 'N/A')}")
print()

print('=' * 60)
print('Test 2: Generation with token IDs input')
print('=' * 60)
# Encode input first
input_text = 'Hello, how are you?'
token_ids = lm.encode(input_text)
print(f'Input text: {input_text}')
print(f'Token IDs: {token_ids}')

result = lm.generate(
    token_ids,
    max_tokens=30,
    temperature=0.8,
    return_token_ids=True,
)
print(f"Generated text: {result.get('text', 'N/A')}")
print(f"Generated token IDs: {result.get('token_ids', 'N/A')}")
print()

print('=' * 60)
print('Test 3: Multiple generations (n=3)')
print('=' * 60)
results = lm.generate(
    'Once upon a time',
    max_tokens=30,
    temperature=0.9,
    n=3,
)
for i, result in enumerate(results, 1):
    print(f"Generation {i}: {result.get('text', 'N/A')}")
print()

print('=' * 60)
print('Test 4: Generation with sampling parameters')
print('=' * 60)
result = lm.generate(
    'The best programming language is',
    max_tokens=40,
    temperature=0.5,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.1,
)
print(f"Generated: {result.get('text', 'N/A')}")
print()

print('=' * 60)
print('Test 5: Generation with stop sequences')
print('=' * 60)
result = lm.generate(
    'List three colors:\n1.',
    max_tokens=100,
    temperature=0.7,
    stop=['\n4.', 'That'],
)
print(f"Generated: {result.get('text', 'N/A')}")
print()

print('✓ All tests completed!')
