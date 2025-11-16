# type: ignore

"""
Simplified LLM Task module for handling language model interactions with structured input/output.
"""

import os
import subprocess
from typing import Any, Dict, List, Optional, Type, Union, cast

import requests
from loguru import logger
from openai import AuthenticationError, BadRequestError, OpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from speedy_utils.common.utils_io import jdumps

from .base_prompt_builder import BasePromptBuilder
from .mixins import (
    ModelUtilsMixin,
    TemperatureRangeMixin,
    TwoStepPydanticMixin,
    VLLMMixin,
)
from .utils import (
    _extract_port_from_vllm_cmd,
    _get_port_from_client,
    _is_lora_path,
    _is_server_running,
    _kill_vllm_on_port,
    _load_lora_adapter,
    _start_vllm_server,
    _unload_lora_adapter,
    get_base_client,
    kill_all_vllm_processes,
    stop_vllm_process,
)


# Type aliases for better readability
Messages = list[ChatCompletionMessageParam]


class LLM(
    TemperatureRangeMixin,
    TwoStepPydanticMixin,
    VLLMMixin,
    ModelUtilsMixin,
):
    """LLM task with structured input/output handling."""

    def __init__(
        self,
        instruction: str | None = None,
        input_model: type[BaseModel] | type[str] = str,
        output_model: type[BaseModel] | type[str] = None,
        client: OpenAI | int | str | None = None,
        cache=True,
        is_reasoning_model: bool = False,
        force_lora_unload: bool = False,
        lora_path: str | None = None,
        vllm_cmd: str | None = None,
        vllm_timeout: int = 1200,
        vllm_reuse: bool = True,
        **model_kwargs,
    ):
        """Initialize LLMTask."""
        self.instruction = instruction
        self.input_model = input_model
        self.output_model = output_model
        self.model_kwargs = model_kwargs
        self.is_reasoning_model = is_reasoning_model
        self.force_lora_unload = force_lora_unload
        self.lora_path = lora_path
        self.vllm_cmd = vllm_cmd
        self.vllm_timeout = vllm_timeout
        self.vllm_reuse = vllm_reuse
        self.vllm_process: subprocess.Popen | None = None
        self.last_ai_response = None  # Store raw response from client
        self.cache = cache

        # Handle VLLM server startup if vllm_cmd is provided
        if self.vllm_cmd:
            self._setup_vllm_server()

            # Set client to use the VLLM server port if not explicitly provided
            port = _extract_port_from_vllm_cmd(self.vllm_cmd)
            if client is None:
                client = port

        self.client = get_base_client(
            client, cache=cache, vllm_cmd=self.vllm_cmd, vllm_process=self.vllm_process
        )
        # check connection of client
        try:
            self.client.models.list()
        except Exception as e:
            logger.error(
                f'Failed to connect to OpenAI client: {str(e)}, base_url={self.client.base_url}'
            )
            raise e

        if not self.model_kwargs.get('model', ''):
            self.model_kwargs['model'] = self.client.models.list().data[0].id

        # Handle LoRA loading if lora_path is provided
        if self.lora_path:
            self._load_lora_adapter()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_vllm_server()

    def _prepare_input(self, input_data: str | BaseModel | list[dict]) -> Messages:
        """Convert input to messages format."""
        if isinstance(input_data, list):
            assert isinstance(input_data[0], dict) and 'role' in input_data[0], (
                "If input_data is a list, it must be a list of messages with 'role' and 'content' keys."
            )
            return cast(Messages, input_data)
        # Convert input to string format
        if isinstance(input_data, str):
            user_content = input_data
        elif hasattr(input_data, 'model_dump_json'):
            user_content = input_data.model_dump_json()
        elif isinstance(input_data, dict):
            user_content = jdumps(input_data)
        else:
            user_content = str(input_data)

        # Build messages
        messages = (
            [
                {'role': 'system', 'content': self.instruction},
            ]
            if self.instruction is not None
            else []
        )

        messages.append({'role': 'user', 'content': user_content})
        return cast(Messages, messages)

    def text_completion(
        self, input_data: str | BaseModel | list[dict], **runtime_kwargs
    ) -> list[dict[str, Any]]:
        """Execute LLM task and return text responses."""
        # Prepare messages
        messages = self._prepare_input(input_data)

        # Merge runtime kwargs with default model kwargs (runtime takes precedence)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name = effective_kwargs.get('model', self.model_kwargs['model'])

        # Extract model name from kwargs for API call
        api_kwargs = {k: v for k, v in effective_kwargs.items() if k != 'model'}

        try:
            completion = self.client.chat.completions.create(
                model=model_name, messages=messages, **api_kwargs
            )
            # Store raw response from client
            self.last_ai_response = completion
        except (AuthenticationError, RateLimitError, BadRequestError) as exc:
            error_msg = f'OpenAI API error ({type(exc).__name__}): {exc}'
            logger.error(error_msg)
            raise
        except Exception as e:
            is_length_error = 'Length' in str(e) or 'maximum context length' in str(e)
            if is_length_error:
                raise ValueError(
                    f'Input too long for model {model_name}. Error: {str(e)[:100]}...'
                ) from e
            # Re-raise all other exceptions
            raise
        # print(completion)

        results: list[dict[str, Any]] = []
        for choice in completion.choices:
            choice_messages = cast(
                Messages,
                messages + [{'role': 'assistant', 'content': choice.message.content}],
            )
            result_dict = {
                'parsed': choice.message.content,
                'messages': choice_messages,
            }

            # Add reasoning content if this is a reasoning model
            if self.is_reasoning_model and hasattr(choice.message, 'reasoning_content'):
                result_dict['reasoning_content'] = choice.message.reasoning_content

            results.append(result_dict)
        return results

    def pydantic_parse(
        self,
        input_data: str | BaseModel | list[dict],
        response_model: type[BaseModel] | None | type[str] = None,
        **runtime_kwargs,
    ) -> list[dict[str, Any]]:
        """Execute LLM task and return parsed Pydantic model responses."""
        # Prepare messages
        messages = self._prepare_input(input_data)

        # Merge runtime kwargs with default model kwargs (runtime takes precedence)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name = effective_kwargs.get('model', self.model_kwargs['model'])

        # Extract model name from kwargs for API call
        api_kwargs = {k: v for k, v in effective_kwargs.items() if k != 'model'}

        pydantic_model_to_use_opt = response_model or self.output_model
        if pydantic_model_to_use_opt is None:
            raise ValueError(
                'No response model specified. Either set output_model in constructor or pass response_model parameter.'
            )
        pydantic_model_to_use: type[BaseModel] = cast(
            type[BaseModel], pydantic_model_to_use_opt
        )
        try:
            completion = self.client.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=pydantic_model_to_use,
                **api_kwargs,
            )
            # Store raw response from client
            self.last_ai_response = completion
        except (AuthenticationError, RateLimitError, BadRequestError) as exc:
            error_msg = f'OpenAI API error ({type(exc).__name__}): {exc}'
            logger.error(error_msg)
            raise
        except Exception as e:
            is_length_error = 'Length' in str(e) or 'maximum context length' in str(e)
            if is_length_error:
                raise ValueError(
                    f'Input too long for model {model_name}. Error: {str(e)[:100]}...'
                ) from e
            raise

        results: list[dict[str, Any]] = []
        for choice in completion.choices:  # type: ignore[attr-defined]
            choice_messages = cast(
                Messages,
                messages + [{'role': 'assistant', 'content': choice.message.content}],
            )

            # Ensure consistent Pydantic model output for both fresh and cached responses
            parsed_content = choice.message.parsed  # type: ignore[attr-defined]
            if isinstance(parsed_content, dict):
                # Cached response: validate dict back to Pydantic model
                parsed_content = pydantic_model_to_use.model_validate(parsed_content)
            elif not isinstance(parsed_content, pydantic_model_to_use):
                # Fallback: ensure it's the correct type
                parsed_content = pydantic_model_to_use.model_validate(parsed_content)

            result_dict = {'parsed': parsed_content, 'messages': choice_messages}

            # Add reasoning content if this is a reasoning model
            if self.is_reasoning_model and hasattr(choice.message, 'reasoning_content'):
                result_dict['reasoning_content'] = choice.message.reasoning_content

            results.append(result_dict)
        return results

    def __call__(
        self,
        input_data: str | BaseModel | list[dict],
        response_model: type[BaseModel] | type[str] | None = None,
        two_step_parse_pydantic: bool = False,
        temperature_ranges: tuple[float, float] | None = None,
        n: int = 1,
        cache=None,
        **openai_client_kwargs,
    ) -> list[dict[str, Any]]:
        """
        Execute LLM task.

        Args:
            input_data: Input data (string, BaseModel, or message list)
            response_model: Optional response model override
            two_step_parse_pydantic: Use two-step parsing (text then parse)
            temperature_ranges: If set, tuple of (min_temp, max_temp) to sample
            n: Number of temperature samples (only used with temperature_ranges, must be >= 2)
            **runtime_kwargs: Additional runtime parameters

        Returns:
            List of response dictionaries
        """
        if cache is not None:
            if hasattr(self.client, 'set_cache'):
                self.client.set_cache(cache)
            else:
                logger.warning('Client does not support caching.')
        # Handle temperature range sampling
        if temperature_ranges is not None:
            if n < 2:
                raise ValueError(
                    f'n must be >= 2 when using temperature_ranges, got {n}'
                )
            return self.temperature_range_sampling(
                input_data,
                temperature_ranges=temperature_ranges,
                n=n,
                response_model=response_model,
                **openai_client_kwargs,
            )
        openai_client_kwargs['n'] = n

        # Handle two-step Pydantic parsing
        pydantic_model = response_model or self.output_model
        if two_step_parse_pydantic and pydantic_model not in (str, None):
            choices = self.two_step_pydantic_parse(
                input_data,
                response_model=pydantic_model,
                **openai_client_kwargs,
            )
        else:
            choices = self.__inner_call__(
                input_data,
                response_model=response_model,
                two_step_parse_pydantic=False,
                **openai_client_kwargs,
            )

        # Track conversation history
        _last_conv = choices[0]['messages'] if choices else []
        if not hasattr(self, '_last_conversations'):
            self._last_conversations = []
        else:
            self._last_conversations = self._last_conversations[-100:]
        self._last_conversations.append(_last_conv)
        return choices

    def inspect_history(
        self, idx: int = -1, k_last_messages: int = 2
    ) -> list[dict[str, Any]]:
        """Inspect the message history of a specific response choice."""
        if hasattr(self, '_last_conversations'):
            from llm_utils import show_chat_v2

            conv = self._last_conversations[idx]
            if k_last_messages > 0:
                conv = conv[-k_last_messages:]
            return show_chat_v2(conv)
        raise ValueError('No message history available. Make a call first.')

    def __inner_call__(
        self,
        input_data: str | BaseModel | list[dict],
        response_model: type[BaseModel] | type[str] | None = None,
        two_step_parse_pydantic: bool = False,
        **runtime_kwargs,
    ) -> list[dict[str, Any]]:
        """
        Internal call handler. Delegates to text() or parse() based on model.

        Note: two_step_parse_pydantic is deprecated here; use the public
        __call__ method which routes to the mixin.
        """
        pydantic_model_to_use = response_model or self.output_model

        if pydantic_model_to_use is str or pydantic_model_to_use is None:
            return self.text_completion(input_data, **runtime_kwargs)
        return self.pydantic_parse(
            input_data,
            response_model=response_model,
            **runtime_kwargs,
        )

    # Backward compatibility aliases
    def text(self, *args, **kwargs) -> list[dict[str, Any]]:
        """Alias for text_completion() for backward compatibility."""
        return self.text_completion(*args, **kwargs)

    def parse(self, *args, **kwargs) -> list[dict[str, Any]]:
        """Alias for pydantic_parse() for backward compatibility."""
        return self.pydantic_parse(*args, **kwargs)

    @classmethod
    def from_prompt_builder(
        cls: BasePromptBuilder,
        client: OpenAI | int | str | None = None,
        cache=True,
        is_reasoning_model: bool = False,
        lora_path: str | None = None,
        vllm_cmd: str | None = None,
        vllm_timeout: int = 120,
        vllm_reuse: bool = True,
        **model_kwargs,
    ) -> 'LLM':
        """
        Create an LLMTask instance from a BasePromptBuilder instance.

        This method extracts the instruction, input model, and output model
        from the provided builder and initializes an LLMTask accordingly.

        Args:
            builder: BasePromptBuilder instance
            client: OpenAI client, port number, or base_url string
            cache: Whether to use cached responses (default True)
            is_reasoning_model: Whether model is reasoning model (default False)
            lora_path: Optional path to LoRA adapter directory
            vllm_cmd: Optional VLLM command to start server automatically
            vllm_timeout: Timeout in seconds to wait for VLLM server (default 120)
            vllm_reuse: If True (default), reuse existing server on target port
            **model_kwargs: Additional model parameters
        """
        instruction = cls.get_instruction()
        input_model = cls.get_input_model()
        output_model = cls.get_output_model()

        # Extract data from the builder to initialize LLMTask
        return LLM(
            instruction=instruction,
            input_model=input_model,
            output_model=output_model,
            client=client,
            cache=cache,
            is_reasoning_model=is_reasoning_model,
            lora_path=lora_path,
            vllm_cmd=vllm_cmd,
            vllm_timeout=vllm_timeout,
            vllm_reuse=vllm_reuse,
            **model_kwargs,
        )
