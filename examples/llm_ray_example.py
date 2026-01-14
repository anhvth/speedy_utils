"""
Example: Using LLMRay for distributed offline batch inference.

This demonstrates how to process large batches of OpenAI-style messages
across multiple GPUs in a Ray cluster with automatic data parallelism.

Key concepts:
    - dp (data parallel): Number of model replicas
    - tp (tensor parallel): GPUs per replica
    - Total GPUs = dp * tp
"""
from llm_utils import LLMRay
from speedy_utils import dump_json_or_pickle

# --- Example 1: Simple batch generation ---
print('=== Example 1: Simple batch generation ===')

# Create LLMRay instance
# - dp=4: 4 model replicas (workers)
# - tp=2: each replica uses 2 GPUs
# - Total: 8 GPUs used
# - If cluster has 16 GPUs across 2 nodes, Ray will distribute automatically
llm = LLMRay(
    model_name='Qwen/Qwen3-0.6B',
    dp=4,
    tp=2,
    sampling_params={'temperature': 0.7, 'max_tokens': 128},
)

# Prepare messages (OpenAI format: list of message lists)
messages_list = [
    [{'role': 'user', 'content': 'What is artificial intelligence?'}],
    [{'role': 'user', 'content': 'Explain quantum computing in simple terms.'}],
    [{'role': 'user', 'content': 'Write a haiku about programming.'}],
    [{'role': 'user', 'content': 'What are the benefits of distributed computing?'}],
] + [[{'role': 'user', 'content': f'Summarize document {i}'}] for i in range(20)]

# Generate responses (automatically distributed across all workers)
results = llm.generate(messages_list)

# Save results
dump_json_or_pickle(results, 'llm_ray_results.json')

print(f'\nProcessed {len(results)} messages')
print(f'\nSample result:\n{results[0]}')


# --- Example 2: Multi-turn conversation ---
print('\n=== Example 2: Multi-turn conversation ===')

# Multi-turn conversations with system prompts
inputs = [
    [
        {'role': 'system', 'content': 'You are a creative writer.'},
        {'role': 'user', 'content': 'Write a short story about a robot.'},
    ],
    [
        {'role': 'system', 'content': 'You are a math tutor.'},
        {'role': 'user', 'content': 'What is 2+2?'},
        {'role': 'assistant', 'content': '2+2 equals 4.'},
        {'role': 'user', 'content': 'What about 3+3?'},
    ],
]

# Process conversations
results = llm(inputs)  # Can also use __call__ syntax

for i, result in enumerate(results):
    print(f'\nConversation {i + 1}:')
    print(f'Generated: {result["generated_text"][:100]}...')


print('\n=== All examples completed! ===')
