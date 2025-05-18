import os
import warnings
import pytest
from llm_utils import LM

# Filter out specific deprecation warnings - need to be more specific with the exact text pattern
warnings.filterwarnings("ignore", message=".*There is no current event loop.*")
warnings.filterwarnings("ignore", message=".*Support for class-based.*")
@pytest.fixture(scope="module")
def oai_key():
    assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY environment variable is not set."
    return os.getenv("OPENAI_API_KEY")

def test_forward(oai_key):
    prompt = "say this is a test"
    model = LM(model="gpt-4.1-nano")
    response = model(prompt=prompt)
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"Response: {response}")
