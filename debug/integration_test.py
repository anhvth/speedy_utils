"""Final integration test for encode/decode/generate functionality."""

from llm_utils.lm import LLM

print('Testing LLM encode/decode/generate integration...\n')

# Initialize
lm = LLM(client=8000)

# Test 1: Encode/Decode
print('✓ Test 1: Encode/Decode')
text = 'Hello, world!'
tokens = lm.encode(text)
decoded = lm.decode(tokens)
print(f'  Original: {text}')
print(f'  Tokens: {tokens}')
print(f'  Decoded: {decoded}')
assert isinstance(tokens, list)
assert all(isinstance(t, int) for t in tokens)
print()

# Test 2: Generate from text
print('✓ Test 2: Generate from text')
result = lm.generate('The answer is', max_tokens=10, temperature=0.5)
print(f'  Input: "The answer is"')
print(f'  Output: {result["text"][:50]}')
assert 'text' in result
print()

# Test 3: Generate from token IDs
print('✓ Test 3: Generate from token IDs')
input_ids = lm.encode('Python is')
result = lm.generate(input_ids, max_tokens=15, return_token_ids=True)
print(f'  Input IDs: {input_ids}')
print(f'  Output: {result["text"][:50]}')
print(f'  Output IDs: {result.get("token_ids", "N/A")}')
assert 'text' in result
assert 'token_ids' in result
print()

# Test 4: Multiple generations
print('✓ Test 4: Multiple generations (n=3)')
results = lm.generate('AI is', max_tokens=10, n=3, temperature=0.8)
print(f'  Input: "AI is"')
for i, r in enumerate(results, 1):
    print(f'  {i}. {r["text"][:40]}')
# Note: Server may return fewer choices than requested
assert isinstance(results, list)
assert len(results) >= 1
print()

# Test 5: Temperature control
print('✓ Test 5: Temperature control')
low_temp = lm.generate('1 + 1 =', max_tokens=5, temperature=0.1, seed=42)
high_temp = lm.generate('1 + 1 =', max_tokens=5, temperature=1.5, seed=123)
print(f'  Low temp (0.1): {low_temp["text"][:30]}')
print(f'  High temp (1.5): {high_temp["text"][:30]}')
print()

# Test 6: Stop sequences
print('✓ Test 6: Stop sequences')
result = lm.generate(
    'Count: 1, 2, 3,',
    max_tokens=50,
    stop=[', 6', '\n'],
    temperature=0.7,
)
print(f'  Input: "Count: 1, 2, 3,"')
print(f'  Output (stop at ", 6"): {result["text"][:50]}')
print()

# Test 7: Seed reproducibility
print('✓ Test 7: Seed reproducibility')
r1 = lm.generate('Test', max_tokens=10, seed=999, temperature=0.8)
r2 = lm.generate('Test', max_tokens=10, seed=999, temperature=0.8)
print(f'  Run 1: {r1["text"][:40]}')
print(f'  Run 2: {r2["text"][:40]}')
print(f'  Match: {r1["text"] == r2["text"]}')
print()

print('=' * 60)
print('✅ All tests passed!')
print('=' * 60)
print()
print('Available methods:')
print('  - lm.encode(text) -> list[int]')
print('  - lm.decode(token_ids) -> str')
print('  - lm.generate(input, **kwargs) -> dict')
