"""Debug the actual response from generate endpoint."""

import requests
import json

base_url = 'http://localhost:8000'

# Encode input first
response = requests.post(
    f'{base_url}/tokenize',
    json={'prompt': 'Hello, how are you?', 'add_special_tokens': True},
)
token_ids = response.json()['tokens']
print(f'Input token IDs: {token_ids}')
print()

# Call generate
request_data = {
    'token_ids': token_ids,
    'sampling_params': {
        'max_tokens': 20,
        'temperature': 0.7,
        'n': 1,
    },
}

response = requests.post(
    f'{base_url}/inference/v1/generate',
    json=request_data,
)
print(f'Status code: {response.status_code}')
print(f'Response JSON:')
print(json.dumps(response.json(), indent=2))
