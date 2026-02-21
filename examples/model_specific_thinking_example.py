"""
Example demonstrating model-specific thinking control.

This example shows how to use the new model-specific LLM classes that
each implement thinking control appropriate to their model architecture:

- AsyncLM_Qwen3: Controls thinking via /think and /no_think system directives
- AsyncLM_GLM5: Controls thinking via extra_body.chat_template_kwargs
- AsyncLM_DeepSeekR1: Documents server-side reasoning parser requirements

The base AsyncLM class no longer has a generic 'think' parameter, avoiding
Qwen-specific assumptions in the base framework.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


async def example_qwen3_thinking():
    """Example: Qwen3 with thinking enabled via /think directive."""
    from llm_utils import AsyncLM_Qwen3
    
    print('=' * 70)
    print('Example 1: Qwen3 with Thinking Enabled')
    print('=' * 70)
    
    # Create Qwen3 LM with thinking enabled
    # This appends '/think' to the system prompt
    lm = AsyncLM_Qwen3(
        enable_thinking=True,
        port=8000,
        temperature=0.6,
    )
    
    print(f'Model: AsyncLM_Qwen3')
    print(f'Thinking: enabled')
    print(f'Temperature: {lm.model_kwargs["temperature"]}')
    print(f'Top-p: {lm.model_kwargs["top_p"]}')
    print('Note: /think directive will be appended to system prompt')
    print()
    
    # Show what the system prompt looks like
    test_system = "You are a helpful assistant."
    modified_system = lm.build_system_prompt(
        response_model=None,
        add_json_schema_to_instruction=False,
        json_schema=None,
        system_content=test_system,
    )
    print(f'Original system prompt: "{test_system}"')
    print(f'Modified system prompt: "{modified_system}"')
    print()


async def example_qwen3_no_thinking():
    """Example: Qwen3 with thinking disabled via /no_think directive."""
    from llm_utils import AsyncLM_Qwen3
    
    print('=' * 70)
    print('Example 2: Qwen3 with Thinking Disabled')
    print('=' * 70)
    
    # Create Qwen3 LM with thinking disabled
    # This appends '/no_think' to the system prompt
    lm = AsyncLM_Qwen3(
        enable_thinking=False,
        port=8000,
    )
    
    print(f'Model: AsyncLM_Qwen3')
    print(f'Thinking: disabled')
    print('Note: /no_think directive will be appended to system prompt')
    print()
    
    # Show what the system prompt looks like
    test_system = "You are a helpful assistant."
    modified_system = lm.build_system_prompt(
        response_model=None,
        add_json_schema_to_instruction=False,
        json_schema=None,
        system_content=test_system,
    )
    print(f'Original system prompt: "{test_system}"')
    print(f'Modified system prompt: "{modified_system}"')
    print()


async def example_glm5_thinking():
    """Example: GLM-5 with thinking enabled via extra_body."""
    from llm_utils import AsyncLM_GLM5
    
    print('=' * 70)
    print('Example 3: GLM-5 with Thinking Enabled')
    print('=' * 70)
    
    # Create GLM-5 LM with thinking enabled
    # This uses extra_body to control thinking
    lm = AsyncLM_GLM5(
        enable_thinking=True,
        port=8000,
    )
    
    print(f'Model: AsyncLM_GLM5')
    print(f'Thinking: enabled')
    print('Note: Uses extra_body={{"chat_template_kwargs": {{"enable_thinking": True}}}}')
    print('Reference: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html')
    print()
    
    # Show the extra_body parameter
    extra_body = lm._get_extra_body_params()
    print(f'extra_body parameter: {extra_body}')
    print()


async def example_deepseek_r1():
    """Example: DeepSeek-R1 with server-side reasoning parser."""
    from llm_utils import AsyncLM_DeepSeekR1
    
    print('=' * 70)
    print('Example 4: DeepSeek-R1 with Server-Side Reasoning')
    print('=' * 70)
    
    # Create DeepSeek-R1 LM
    # Thinking is controlled server-side via --reasoning-parser flag
    lm = AsyncLM_DeepSeekR1(
        port=8000,
    )
    
    print(f'Model: AsyncLM_DeepSeekR1')
    print(f'Temperature: {lm.model_kwargs["temperature"]}')
    print(f'Top-p: {lm.model_kwargs["top_p"]}')
    print()
    print('Note: Thinking mode is configured server-side:')
    print('  vllm serve deepseek-ai/DeepSeek-R1 \\')
    print('    --enable-reasoning --reasoning-parser deepseek_r1')
    print()
    print('Reasoning content appears in response.choices[0].message.reasoning')
    print()


async def example_migration_from_old_api():
    """Example: Migration from old think parameter to new model-specific classes."""
    print('=' * 70)
    print('Example 5: Migration Guide')
    print('=' * 70)
    
    print('OLD API (deprecated, will raise warning):')
    print('  from llm_utils.lm.async_lm.lm_specific import AsyncLMQwenThink')
    print('  lm = AsyncLMQwenThink(port=8000)')
    print()
    
    print('NEW API (recommended):')
    print('  from llm_utils import AsyncLM_Qwen3')
    print('  lm = AsyncLM_Qwen3(enable_thinking=True, port=8000)')
    print()
    
    print('Base AsyncLM no longer has think parameter:')
    print('  # OLD: AsyncLM(think=True)  # <- No longer supported')
    print('  # NEW: AsyncLM_Qwen3(enable_thinking=True)  # <- Use model-specific class')
    print()


async def main():
    """Run all examples."""
    await example_qwen3_thinking()
    await example_qwen3_no_thinking()
    await example_glm5_thinking()
    await example_deepseek_r1()
    await example_migration_from_old_api()
    
    print('=' * 70)
    print('Summary')
    print('=' * 70)
    print('✓ Model-specific thinking mechanisms are now decoupled from base AsyncLM')
    print('✓ Each model family has a dedicated class with appropriate controls')
    print('✓ Qwen3: /think and /no_think system directives')
    print('✓ GLM-5: extra_body.chat_template_kwargs.enable_thinking')
    print('✓ DeepSeek-R1: Server-side --reasoning-parser configuration')
    print()


if __name__ == '__main__':
    asyncio.run(main())
