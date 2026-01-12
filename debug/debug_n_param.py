"""Debug n parameter."""

from llm_utils.lm import LLM
import json

lm = LLM(client=8000)

# Test with different n values
for n_val in [1, 2, 3, 4]:
    print(f'\nTesting n={n_val}')
    results = lm.generate('AI is', max_tokens=10, n=n_val, temperature=0.8)
    
    if isinstance(results, list):
        print(f'  Returned {len(results)} results (expected {n_val})')
        raw = results[0].get('_raw_response', {})
    else:
        print(f'  Returned single dict (expected {n_val})')
        raw = results.get('_raw_response', {})
    
    # Check raw response
    if 'choices' in raw:
        print(f'  Raw response has {len(raw["choices"])} choices')
