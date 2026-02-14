# Migration Guide: Model-Specific Thinking Control

## Summary

**Breaking Change**: The generic `think` parameter has been removed from base `AsyncLM` and `AsyncLMBase` classes. Model-specific thinking mechanisms are now implemented in dedicated subclasses.

## What Changed

### Before (Deprecated)

```python
# Generic think parameter (Qwen-specific, applied to all models)
from llm_utils import AsyncLM

lm = AsyncLM(think=True, port=8000)  # ❌ No longer supported
lm = AsyncLM(think=False, port=8000)  # ❌ No longer supported
```

### After (Current)

```python
# Model-specific classes with appropriate thinking mechanisms
from llm_utils import AsyncLM_Qwen3, AsyncLM_GLM5, AsyncLM_DeepSeekR1

# Qwen3: Uses /think and /no_think system directives
lm_qwen = AsyncLM_Qwen3(enable_thinking=True, port=8000)  # ✓

# GLM-5: Uses extra_body.chat_template_kwargs
lm_glm = AsyncLM_GLM5(enable_thinking=True, port=8000)  # ✓

# DeepSeek-R1: Server-side reasoning parser
lm_deepseek = AsyncLM_DeepSeekR1(port=8000)  # ✓
```

## Why This Change?

The original `think` parameter was Qwen-specific (appending `/think` or `/no_think` to system prompts). This design:

- **Assumed all models** use Qwen's thinking directive format
- **Couldn't support** GLM-5's `extra_body` approach
- **Couldn't support** DeepSeek-R1's server-side configuration
- **Created tight coupling** between framework and model-specific behavior

The new design:

- **Decouples** model-specific behavior from base classes
- **Supports** different thinking mechanisms per model family
- **Makes explicit** which mechanism each model uses
- **Follows** open/closed principle (open for extension, closed for modification)

## Migration Steps

### 1. AsyncLM with think parameter

**Before:**
```python
from llm_utils import AsyncLM

lm = AsyncLM(
    think=True,
    port=8000,
    temperature=0.6,
)
```

**After:**
```python
from llm_utils import AsyncLM_Qwen3

lm = AsyncLM_Qwen3(
    enable_thinking=True,
    port=8000,
    temperature=0.6,
)
```

### 2. AsyncLMQwenThink wrapper

**Before:**
```python
from llm_utils.lm.async_lm.lm_specific import AsyncLMQwenThink

lm = AsyncLMQwenThink(port=8000)
```

**After:**
```python
from llm_utils import AsyncLM_Qwen3

lm = AsyncLM_Qwen3(enable_thinking=True, port=8000)
```

**Note:** The old `AsyncLMQwenThink` still works but emits a deprecation warning.

### 3. AsyncLMQwenNoThink wrapper

**Before:**
```python
from llm_utils.lm.async_lm.lm_specific import AsyncLMQwenNoThink

lm = AsyncLMQwenNoThink(port=8000)
```

**After:**
```python
from llm_utils import AsyncLM_Qwen3

lm = AsyncLM_Qwen3(enable_thinking=False, port=8000)
```

### 4. AsyncLLMTask with think parameter

**Before:**
```python
from llm_utils import AsyncLLMTask

class MyTask(AsyncLLMTask):
    DEFAULT_THINK = True  # ❌ No longer supported
    
    def __init__(self, think=None, **kwargs):
        super().__init__(think=think, **kwargs)  # ❌ No longer supported
```

**After:**
```python
from llm_utils import AsyncLM_Qwen3
from llm_utils.lm.async_lm.async_llm_task import AsyncLLMTask

class MyTask(AsyncLLMTask):
    # Override lm property to use model-specific class
    @property
    def lm(self):
        if not hasattr(self, '_lm'):
            self._lm = AsyncLM_Qwen3(
                enable_thinking=True,
                **self._config.to_dict()
            )
        return self._lm
```

## Model-Specific Classes

### AsyncLM_Qwen3

Controls thinking via system prompt directives.

```python
from llm_utils import AsyncLM_Qwen3

# Enable thinking (appends /think to system prompt)
lm = AsyncLM_Qwen3(enable_thinking=True, port=8000)

# Disable thinking (appends /no_think to system prompt)
lm = AsyncLM_Qwen3(enable_thinking=False, port=8000)
```

**Default sampling parameters** (from KNOWN_CONFIG):
- `temperature=0.6` (thinking) / `0.7` (no-think)
- `top_p=0.95` (thinking) / `0.8` (no-think)
- `top_k=20`
- `presence_penalty=1.5`

### AsyncLM_GLM5

Controls thinking via `extra_body` parameter.

```python
from llm_utils import AsyncLM_GLM5

# Enable thinking
lm = AsyncLM_GLM5(enable_thinking=True, port=8000)
# Uses: extra_body={'chat_template_kwargs': {'enable_thinking': True}}

# Disable thinking
lm = AsyncLM_GLM5(enable_thinking=False, port=8000)
# Uses: extra_body={'chat_template_kwargs': {'enable_thinking': False}}
```

**Reference:** [GLM-5 Usage Guide](https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html)

**Default sampling parameters:**
- `temperature=1.0`
- `max_tokens=4096`

### AsyncLM_DeepSeekR1

Requires server-side reasoning parser configuration.

```python
from llm_utils import AsyncLM_DeepSeekR1

# Server must be started with:
# vllm serve deepseek-ai/DeepSeek-R1 \
#   --enable-reasoning --reasoning-parser deepseek_r1

lm = AsyncLM_DeepSeekR1(port=8000)
```

**Default sampling parameters** (from KNOWN_CONFIG):
- `temperature=0.6`
- `top_p=0.95`

**Note:** Thinking mode cannot be toggled per-request. Use a non-reasoning DeepSeek model for no-think mode.

## Backward Compatibility

### Deprecated but Still Working

The old `AsyncLMQwenThink` and `AsyncLMQwenNoThink` classes still work but emit deprecation warnings:

```python
from llm_utils.lm.async_lm.lm_specific import AsyncLMQwenThink

lm = AsyncLMQwenThink()
# DeprecationWarning: AsyncLMQwenThink is deprecated. 
# Use AsyncLM_Qwen3(enable_thinking=True) instead.
```

### No Longer Working

These will raise errors:

```python
from llm_utils import AsyncLM, AsyncLLMTask

# ❌ TypeError: unexpected keyword argument 'think'
lm = AsyncLM(think=True)

# ❌ AttributeError: 'AsyncLLMTask' has no attribute 'DEFAULT_THINK'
class MyTask(AsyncLLMTask):
    DEFAULT_THINK = True
```

## Testing Your Migration

After migrating, verify:

1. **Imports work:**
   ```python
   from llm_utils import AsyncLM_Qwen3, AsyncLM_GLM5, AsyncLM_DeepSeekR1
   ```

2. **System prompts are modified correctly:**
   ```python
   lm = AsyncLM_Qwen3(enable_thinking=True)
   prompt = lm.build_system_prompt(None, False, None, "Test")
   assert "/think" in prompt
   ```

3. **No deprecation warnings** (unless intentionally using old API)

4. **Tests pass:**
   ```bash
   pytest tests/llm_utils/
   ```

## Questions?

- **Q: Can I still use base `AsyncLM` without thinking?**  
  A: Yes! Base `AsyncLM` works fine for models that don't need thinking control.

- **Q: How do I add support for a new model family?**  
  A: Create a new subclass in `src/llm_utils/lm/async_lm/model_specific.py` following the pattern of existing classes.

- **Q: Why not use a strategy pattern instead of subclasses?**  
  A: Subclasses provide better type safety, clearer documentation, and simpler imports. The thinking mechanism is intrinsic to the model, not a swappable behavior.

- **Q: What about sync `LLM` class?**  
  A: The sync `LLM` class never had a `think` parameter. It uses `is_reasoning_model` flag and post-hoc formatting methods like `generate_with_think_prefix()`.

## Timeline

- **v0.x.x:** `think` parameter removed from base classes
- **Future:** May remove `AsyncLMQwenThink`/`AsyncLMQwenNoThink` entirely (currently deprecated)

## Example Code

See [examples/model_specific_thinking_example.py](../examples/model_specific_thinking_example.py) for complete working examples of all three model families.
