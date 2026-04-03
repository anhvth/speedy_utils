"""Example demonstrating Qwen3LLM prefix continuation."""

import sys
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    from llm_utils import Qwen3LLM

    llm = Qwen3LLM(client=8000)
    message = llm.chat_completion(
        [{"role": "user", "content": "hi"}],
        thinking_max_tokens=10,
        content_max_tokens=1000,
    )
    print(message)


if __name__ == "__main__":
    main()
