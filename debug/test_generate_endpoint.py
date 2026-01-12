"""Check which endpoint path works for generate."""

import requests

# Test different endpoint paths
test_urls = [
    'http://localhost:8000/inference/v1/generate',
    'http://localhost:8000/v1/inference/v1/generate',
    'http://localhost:8000/generate',
]

# Dummy request payload
dummy_payload = {
    'token_ids': [1, 2, 3],
    'sampling_params': {
        'max_tokens': 10,
        'temperature': 1.0,
    },
}

for url in test_urls:
    try:
        response = requests.post(url, json=dummy_payload)
        print(f'✓ {url} - Status: {response.status_code}')
        if response.status_code in [200, 400, 422]:  # 400/422 might be validation error but endpoint exists
            print(f'  Endpoint exists!')
    except Exception as e:
        print(f'✗ {url} - Error: {type(e).__name__}')
