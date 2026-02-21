"""Model-specific LLM implementations with custom thinking mechanisms.

This module provides specialized AsyncLM subclasses that implement
model-specific thinking/reasoning control mechanisms:

- AsyncLM_Qwen3: Qwen3 models use /think and /no_think system prompt directives
- AsyncLM_GLM5: GLM-5 uses extra_body.chat_template_kwargs.enable_thinking
- AsyncLM_DeepSeekR1: DeepSeek-R1 requires server-side reasoning parser config

Each class encapsulates the appropriate thinking control mechanism for its
model family, avoiding Qwen-specific assumptions in the base AsyncLM class.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel

from .async_lm import AsyncLM
from .lm_specific import KNOWN_CONFIG


class AsyncLM_Qwen3(AsyncLM):
    """Qwen3 model with thinking control via system prompt directives.
    
    Qwen3 models control thinking mode by appending /think or /no_think
    directives to the system prompt. The model's chat template interprets
    these directives to enable/disable reasoning output.
    
    Args:
        enable_thinking: If True, appends '/think' to system prompt.
                        If False, appends '/no_think' to system prompt.
        model: Model name/identifier
        temperature: Sampling temperature (default from KNOWN_CONFIG)
        top_p: Nucleus sampling probability (default from KNOWN_CONFIG)
        top_k: Top-k sampling parameter (default from KNOWN_CONFIG)
        presence_penalty: Presence penalty (default from KNOWN_CONFIG)
        **kwargs: Additional arguments passed to AsyncLM
        
    Example:
        >>> lm = AsyncLM_Qwen3(enable_thinking=True, port=8000)
        >>> result = await lm.parse("Solve: x^2 + 2x + 1 = 0")
    """
    
    def __init__(
        self,
        *,
        enable_thinking: bool = True,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        presence_penalty: float | None = None,
        **kwargs,
    ):
        # Use Qwen3 config defaults
        config_key = 'qwen3-think' if enable_thinking else 'qwen3-no-think'
        qwen_config = KNOWN_CONFIG[config_key]['sampling_params']
        
        # Apply defaults from KNOWN_CONFIG
        if temperature is None:
            temperature = qwen_config['temperature']
        if top_p is None:
            top_p = qwen_config['top_p']
        if top_k is None:
            top_k = qwen_config['top_k']
        if presence_penalty is None:
            presence_penalty = qwen_config['presence_penalty']
        
        super().__init__(
            model=model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            **kwargs,
        )
        
        self.enable_thinking = enable_thinking
    
    def build_system_prompt(
        self,
        response_model: type[BaseModel] | None,
        add_json_schema_to_instruction: bool | None,
        json_schema: dict[str, Any] | None,
        system_content: str,
        **kwargs,  # Accept but ignore any other params
    ) -> str:
        """Build system prompt with Qwen3 thinking directives.
        
        Appends /think or /no_think directive based on enable_thinking flag.
        """
        # First apply base system prompt logic (JSON schema, etc.)
        system_content = super().build_system_prompt(
            response_model=response_model,
            add_json_schema_to_instruction=add_json_schema_to_instruction,
            json_schema=json_schema,
            system_content=system_content,
        )
        
        # Add Qwen3-specific thinking directive
        if self.enable_thinking:
            if '/think' in system_content:
                pass  # Already has /think
            elif '/no_think' in system_content:
                system_content = system_content.replace('/no_think', '/think')
            else:
                system_content += '\n\n/think'
        else:
            if '/no_think' in system_content:
                pass  # Already has /no_think
            elif '/think' in system_content:
                system_content = system_content.replace('/think', '/no_think')
            else:
                system_content += '\n\n/no_think'
        
        return system_content


class AsyncLM_GLM5(AsyncLM):
    """GLM-5 model with thinking control via extra_body parameters.
    
    GLM-5 controls thinking mode via the chat_template_kwargs.enable_thinking
    parameter in extra_body. When enabled, the model returns reasoning content
    in the response.choices[0].message.reasoning field.
    
    Reference: https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html
    
    Args:
        enable_thinking: If True, adds extra_body to enable thinking mode.
                        If False (default), thinking is disabled.
        model: Model name/identifier
        temperature: Sampling temperature (default: 1.0)
        max_tokens: Maximum tokens to generate (default: 4096)
        **kwargs: Additional arguments passed to AsyncLM
        
    Example:
        >>> lm = AsyncLM_GLM5(enable_thinking=True, port=8000)
        >>> result = await lm.call_with_messages(messages)
        >>> # Reasoning appears in result.choices[0].message.reasoning
    """
    
    def __init__(
        self,
        *,
        enable_thinking: bool = False,
        model: str | None = None,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        **kwargs,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        
        self.enable_thinking = enable_thinking
    
    def _get_extra_body_params(self) -> dict[str, Any] | None:
        """Get extra_body parameters for GLM-5 thinking control."""
        if not self.enable_thinking:
            # Explicitly disable thinking
            return {
                'chat_template_kwargs': {
                    'enable_thinking': False
                }
            }
        # Thinking is enabled by default in GLM-5, but we can be explicit
        return None  # Or could return {'chat_template_kwargs': {'enable_thinking': True}}
    
    async def _call_and_parse(
        self,
        messages: list[dict[str, str]],
        response_model: type[BaseModel] | None,
        json_schema: dict[str, Any] | None,
    ) -> tuple[Any, list[dict[str, str]], BaseModel | str]:
        """Override to inject extra_body for GLM-5 thinking control."""
        # Build kwargs for completion call
        call_kwargs: dict[str, Any] = {
            'model': self.model_name,
            'messages': messages,
            **self.model_kwargs,
        }
        
        # Add GLM-5 thinking control via extra_body
        extra_body = self._get_extra_body_params()
        if extra_body is not None:
            call_kwargs['extra_body'] = extra_body
        
        # Add response format if needed
        if response_model and not self.add_json_schema_to_instruction:
            call_kwargs['response_format'] = response_model
        
        # Make the API call
        completion = await self.client.chat.completions.create(**call_kwargs)
        
        # Parse the response
        if response_model:
            parsed = self._parse_complete_output(completion, response_model)
        else:
            if hasattr(completion, 'model_dump'):
                completion_dict = completion.model_dump()
            else:
                completion_dict = completion
            content = completion_dict['choices'][0]['message']['content']
            parsed = content or ''
        
        return completion, messages, parsed


class AsyncLM_DeepSeekR1(AsyncLM):
    """DeepSeek-R1 model with server-side reasoning parser.
    
    DeepSeek-R1 requires server-side configuration to enable reasoning mode.
    The VLLM server must be started with:
        --enable-reasoning --reasoning-parser deepseek_r1
    
    When configured correctly, the model automatically outputs reasoning
    content in <think>...</think> tags, which are extracted by the parser
    and returned in the response.choices[0].message.reasoning field.
    
    Args:
        model: Model name/identifier
        temperature: Sampling temperature (default from KNOWN_CONFIG: 0.6)
        top_p: Nucleus sampling probability (default from KNOWN_CONFIG: 0.95)
        **kwargs: Additional arguments passed to AsyncLM
        
    Example:
        >>> # Server must be started with:
        >>> # vllm serve deepseek-ai/DeepSeek-R1 \\
        >>> #   --enable-reasoning --reasoning-parser deepseek_r1
        >>> 
        >>> lm = AsyncLM_DeepSeekR1(port=8000)
        >>> result = await lm.parse("Solve: x^2 + 2x + 1 = 0")
    
    Note:
        Thinking mode is controlled server-side and cannot be toggled
        per-request. Use a non-reasoning DeepSeek model for no-think mode.
    """
    
    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        **kwargs,
    ):
        # Use DeepSeek-R1 config defaults
        r1_config = KNOWN_CONFIG['deepseek-r1']['sampling_params']
        
        if temperature is None:
            temperature = r1_config['temperature']
        if top_p is None:
            top_p = r1_config['top_p']
        
        super().__init__(
            model=model,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
    
    # DeepSeek-R1 doesn't need special overrides since thinking is server-side
    # The reasoning content is automatically extracted by vllm's reasoning parser
    # and appears in response.choices[0].message.reasoning
