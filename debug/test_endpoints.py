from llm_utils.lm import LLM

lm = LLM(client=8000)

# Debug: Check the base_url
print(f"Client base_url: {lm.client.base_url}")

# Try to manually check what endpoints are available
import requests

# Try different endpoint paths
test_urls = [
    "http://localhost:8000/tokenize",
    "http://localhost:8000/v1/tokenize",
]

for url in test_urls:
    try:
        response = requests.post(
            url,
            json={"prompt": "test", "add_special_tokens": True}
        )
        print(f"✓ {url} - Status: {response.status_code}")
        if response.status_code == 200:
            print(f"  Response: {response.json()}")
    except Exception as e:
        print(f"✗ {url} - Error: {e}")
