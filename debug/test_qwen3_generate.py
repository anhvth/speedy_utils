from llm_utils import Qwen3LLM


llm = Qwen3LLM(client=8001)

msg = llm.generate_with_prefix(
    [{"role": "user", "content": "hi"}],
    thinking_max_tokens=10,
    content_max_tokens=1000,
)

print(msg)
