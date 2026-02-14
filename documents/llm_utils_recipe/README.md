# LLM Utils Recipe Collection

Practical, copy-paste-ready recipes for using `llm_utils` from the speedy_utils package.

## 📚 Recipe Index

### Core LLM Interfaces
- **[01_basic_llm_usage.md](01_basic_llm_usage.md)** - Get started with AsyncLM and LLM classes
- **[02_model_specific_thinking.md](02_model_specific_thinking.md)** - Control thinking modes for Qwen3, GLM-5, DeepSeek-R1
- **[03_structured_outputs.md](03_structured_outputs.md)** - Generate JSON/Pydantic models with type safety
- **[04_llm_task_pattern.md](04_llm_task_pattern.md)** - Build reusable LLM task abstractions

### Chat & Formatting
- **[05_chat_format_transform.md](05_chat_format_transform.md)** - Convert between ChatML, ShareGPT, Text formats
- **[06_chat_display.md](06_chat_display.md)** - Visualize conversations in notebooks

### Caching & Performance
- **[07_memoized_openai.md](07_memoized_openai.md)** - Auto-cache OpenAI API calls to save costs
- **[08_vector_cache.md](08_vector_cache.md)** - Semantic caching with embeddings

### Distributed & Advanced
- **[09_ray_distributed_llm.md](09_ray_distributed_llm.md)** - Scale LLM inference with Ray
- **[10_temperature_sampling.md](10_temperature_sampling.md)** - Generate diverse responses with temperature ranges
- **[11_tokenization.md](11_tokenization.md)** - Efficient tokenization with caching

### Server Management
- **[12_vllm_server_management.md](12_vllm_server_management.md)** - Start, stop, and manage VLLM servers

## 🎯 Quick Start

```python
# Install
pip install speedy-utils[llm]

# Basic usage
from llm_utils import AsyncLM
import asyncio

async def main():
    lm = AsyncLM(port=8000)  # VLLM server on localhost:8000
    response = await lm.generate("Explain Python decorators")
    print(response)

asyncio.run(main())
```

## 🏗️ Architecture Overview

```
llm_utils/
├── AsyncLM / LLM          # Core async/sync LLM interfaces
├── AsyncLM_Qwen3          # Model-specific: Qwen thinking control
├── AsyncLM_GLM5           # Model-specific: GLM thinking control
├── AsyncLM_DeepSeekR1     # Model-specific: DeepSeek reasoning
├── AsyncLLMTask           # Abstract base for task-oriented LLMs
├── LLMRay                 # Ray-based distributed inference
├── MOpenAI / MAsyncOpenAI # Memoized OpenAI clients
├── chat_format/           # Message transformation utilities
└── vector_cache/          # Semantic caching with embeddings
```

## 🔑 Key Concepts

### 1. **Model-Agnostic Base Classes**
`AsyncLM` and `LLM` work with any OpenAI-compatible API (VLLM, Ollama, OpenAI).

### 2. **Model-Specific Subclasses**
For models with special features (thinking, reasoning), use dedicated classes:
- `AsyncLM_Qwen3` - Qwen's `/think` directives
- `AsyncLM_GLM5` - GLM's extra_body parameters
- `AsyncLM_DeepSeekR1` - DeepSeek's reasoning parser

### 3. **Memoization First**
All LLM calls can be automatically cached via:
- `@memoize` decorator on functions
- `MOpenAI` drop-in replacement for OpenAI client
- `vector_cache` for semantic similarity caching

### 4. **Type-Safe Outputs**
Use Pydantic models for structured generation with validation.

## 📖 Common Patterns

### Pattern: Quick Inference
```python
from llm_utils import AsyncLM

async def quick_inference():
    lm = AsyncLM(port=8000)
    return await lm.generate("Your prompt here")
```

### Pattern: Structured Output
```python
from llm_utils import AsyncLM
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

async def extract_person(text: str):
    lm = AsyncLM(port=8000)
    return await lm.generate(text, response_model=Person)
```

### Pattern: Cached API Calls
```python
from llm_utils import MAsyncOpenAI

client = MAsyncOpenAI(api_key="sk-...")
# Repeated calls with same args are cached automatically
response = await client.chat.completions.create(...)
```

### Pattern: Model-Specific Thinking
```python
from llm_utils import AsyncLM_Qwen3

lm = AsyncLM_Qwen3(enable_thinking=True, port=8000)
response = await lm.generate("Solve: 2x + 5 = 13")
# Qwen will show reasoning process
```

## 🧪 Testing Your Setup

```python
# Test VLLM server connection
from llm_utils import AsyncLM
import asyncio

async def test_connection():
    try:
        lm = AsyncLM(port=8000)
        response = await lm.generate("Say 'Ready'", max_tokens=10)
        print(f"✓ Server connected: {response}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")

asyncio.run(test_connection())
```

## 🚀 Performance Tips

1. **Use async for I/O-bound workloads** - `AsyncLM` is faster for multiple requests
2. **Enable caching** - Wrap calls with `@memoize` or use `MOpenAI`
3. **Batch with Ray** - Use `LLMRay` for processing large datasets
4. **Reuse clients** - Create LM instances once, call many times
5. **Temperature=0 for reproducibility** - Set `temperature=0` for deterministic outputs

## 📚 Further Reading

- [VLLM Documentation](https://docs.vllm.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [speedy_utils Caching Guide](../../.github/skills/caching-utilities/SKILL.md)

## 🐛 Troubleshooting

**Problem**: `Connection refused` error  
**Solution**: Ensure VLLM server is running: `vllm serve model_name --port 8000`

**Problem**: Import takes too long  
**Solution**: llm_utils uses lazy loading. Check `CLAUDE.md` for import time guidelines.

**Problem**: Structured output returns None  
**Solution**: Check if model supports JSON mode. Add explicit instructions in prompt.

**Problem**: Old `think` parameter not working  
**Solution**: Use model-specific classes like `AsyncLM_Qwen3` instead of base `AsyncLM`.

---

**Last Updated**: February 2026  
**Version**: 1.0.0  
**Maintainer**: speedy_utils team
