"""Example: Using the HuggingFace-style generate() method.

This demonstrates the low-level generation interface that works
directly with token IDs, similar to HuggingFace Transformers.
"""

from llm_utils.lm import LLM

# Initialize LLM
lm = LLM(client=8000)

print('=' * 70)
print('HuggingFace-Style Generation Examples')
print('=' * 70)
print()

# Example 1: Basic text generation
print('1. Basic Text Generation')
print('-' * 70)
result = lm.generate(
    'Write a haiku about coding:',
    max_tokens=50,
    temperature=0.8,
)
print(f"Prompt: 'Write a haiku about coding:'")
print(f"Generated:\n{result['text']}")
print()

# Example 2: Working with token IDs
print('2. Working with Token IDs (like HuggingFace)')
print('-' * 70)
input_text = 'The meaning of life is'
input_ids = lm.encode(input_text)
print(f'Input text: "{input_text}"')
print(f'Input IDs: {input_ids}')

result = lm.generate(
    input_ids,  # Pass token IDs directly
    max_tokens=30,
    temperature=0.7,
    return_token_ids=True,  # Get token IDs back
)
print(f'Output IDs: {result["token_ids"]}')
print(f'Output text: {result["text"]}')
print()

# Example 3: Temperature sampling
print('3. Temperature Comparison (Creative vs Deterministic)')
print('-' * 70)
prompt = 'The best way to learn programming is'

# Low temperature (more deterministic)
result_low = lm.generate(prompt, max_tokens=40, temperature=0.1)
print(f'Temperature 0.1 (deterministic):')
print(f'  {result_low["text"][:100]}...')

# High temperature (more creative)
result_high = lm.generate(prompt, max_tokens=40, temperature=1.5)
print(f'Temperature 1.5 (creative):')
print(f'  {result_high["text"][:100]}...')
print()

# Example 4: Top-k and Top-p sampling
print('4. Advanced Sampling (top_k, top_p)')
print('-' * 70)
result = lm.generate(
    'In the future, AI will',
    max_tokens=50,
    temperature=0.9,
    top_k=50,  # Only sample from top 50 tokens
    top_p=0.95,  # Nucleus sampling
    repetition_penalty=1.2,  # Reduce repetition
)
print(f'Generated: {result["text"]}')
print()

# Example 5: Multiple generations (like num_return_sequences)
print('5. Multiple Generations (n=4)')
print('-' * 70)
results = lm.generate(
    'A creative name for a tech startup:',
    max_tokens=20,
    temperature=0.9,
    n=4,  # Generate 4 different completions
)
for i, result in enumerate(results, 1):
    print(f'{i}. {result["text"]}')
print()

# Example 6: Controlled generation with stop sequences
print('6. Controlled Generation with Stop Sequences')
print('-' * 70)
result = lm.generate(
    'Ingredients for chocolate chip cookies:\n-',
    max_tokens=200,
    temperature=0.7,
    stop=['\n\n', 'Instructions:'],  # Stop at these sequences
)
print(f'Generated:\n{result["text"]}')
print()

# Example 7: Reproducible generation with seed
print('7. Reproducible Generation (with seed)')
print('-' * 70)
prompt = 'Random number:'

# Same seed = same output
result1 = lm.generate(prompt, max_tokens=10, temperature=0.8, seed=42)
result2 = lm.generate(prompt, max_tokens=10, temperature=0.8, seed=42)
result3 = lm.generate(prompt, max_tokens=10, temperature=0.8, seed=123)

print(f'Seed 42 (run 1): {result1["text"]}')
print(f'Seed 42 (run 2): {result2["text"]}')
print(f'Seed 123:        {result3["text"]}')
print(f'Results match:   {result1["text"] == result2["text"]}')
print()

# Example 8: Token count estimation
print('8. Token Budget Management')
print('-' * 70)
long_prompt = 'Explain quantum computing in detail: ' * 10
prompt_tokens = lm.encode(long_prompt)
max_context = 4096  # Example context window
available_tokens = max_context - len(prompt_tokens)

print(f'Prompt tokens: {len(prompt_tokens)}')
print(f'Max context: {max_context}')
print(f'Available for generation: {available_tokens}')

result = lm.generate(
    long_prompt,
    max_tokens=min(100, available_tokens),
    temperature=0.7,
)
generated_tokens = lm.encode(result['text'])
print(f'Generated tokens: {len(generated_tokens)}')
print(f'Total tokens used: {len(prompt_tokens) + len(generated_tokens)}')
print()

print('✓ All examples completed!')
