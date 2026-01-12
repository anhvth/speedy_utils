"""Debug n parameter structure."""

from llm_utils.lm import LLM
import json

lm = LLM(client=8000)

result = lm.generate('Test', max_tokens=5, n=3, temperature=0.8)

print('Type:', type(result))
print('Length:', len(result) if isinstance(result, list) else 'N/A')
print()

if isinstance(result, list):
    print('First result keys:', result[0].keys())
    raw = result[0]['_raw_response']
    print('Raw response type:', type(raw))
    print('Raw response keys:', raw.keys() if isinstance(raw, dict) else 'N/A')
    print()
    
# Get actual response
import requests
base_url = 'http://localhost:8000'
token_ids = lm.encode('Test')
response = requests.post(
    f'{base_url}/inference/v1/generate',
    json={
        'token_ids': token_ids,
        'sampling_params': {'max_tokens': 5, 'n': 3, 'temperature': 0.8},
    },
)
print('Actual API response:')
print(json.dumps(response.json(), indent=2))
