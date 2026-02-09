# type: ignore

"""
Simplified LLM Task module for handling language model interactions with structured input/output.
"""

import subprocess
from typing import TYPE_CHECKING, Any, Dict, List, Optional, cast

from httpx import Timeout
from loguru import logger
from pydantic import BaseModel

from speedy_utils import clean_traceback
from speedy_utils.common.utils_io import jdumps

from .base_prompt_builder import BasePromptBuilder
from .mixins import (
    ModelUtilsMixin,
    TemperatureRangeMixin,
    TokenizationMixin,
    TwoStepPydanticMixin,
    VLLMMixin,
)
from .utils import (
    _extract_port_from_vllm_cmd,
    get_base_client,
)

# Typing imports
from collections.abc import Iterator, AsyncIterator

# Lazy import openai types for type checking only
if TYPE_CHECKING:
    from openai import (
        AuthenticationError,
        BadRequestError,
        OpenAI,
        RateLimitError,
        APITimeoutError,
    )
    from openai.types.chat import ChatCompletionMessageParam

# Type aliases for better readability
Messages = list[dict]  # Simplified type, actual type validated at runtime


class LLM(
    TemperatureRangeMixin,
    TwoStepPydanticMixin,
    VLLMMixin,
    ModelUtilsMixin,
    TokenizationMixin,
):
    """LLM task with structured input/output handling."""

    def __init__(
        self,
        instruction: str | None = None,
        input_model: type[BaseModel] | type[str] = str,
        output_model: type[BaseModel] | type[str] = None,
        client: 'OpenAI | int | str | None' = None,  # type: ignore[name-defined]
        cache=True,
        is_reasoning_model: bool = False,
        force_lora_unload: bool = False,
        lora_path: str | None = None,
        vllm_cmd: str | None = None,
        vllm_timeout: int = 1200,
        vllm_reuse: bool = True,
        verbose=False,
        timeout: float | Timeout | None = None,
        **model_kwargs,
    ):
        """Initialize LLMTask."""
        if verbose:
            available_models = LLM.list_models(client=client)
            logger.info(f'Available models: {available_models}')
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
        self.timeout = timeout
        self.last_ai_response = None  # Store raw response from client
        self.cache = cache
        # Avoid importing OpenAI client class at module import time.
        # If a client object provides an api_key attribute, use it.
        self.api_key = 'abc'
        if client is not None:
            api_key = getattr(client, 'api_key', None)
            if isinstance(api_key, str) and api_key:
                self.api_key = api_key

        # Handle VLLM server startup if vllm_cmd is provided
        if self.vllm_cmd:
            self._setup_vllm_server()

            # Set client to use the VLLM server port if not explicitly provided
            port = _extract_port_from_vllm_cmd(self.vllm_cmd)
            if client is None:
                client = port

        self.client = get_base_client(
            client,
            cache=cache,
            api_key=self.api_key,
            vllm_cmd=self.vllm_cmd,
            vllm_process=self.vllm_process,
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

    @clean_traceback
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

        if 'timeout' not in api_kwargs and self.timeout is not None:
            api_kwargs['timeout'] = self.timeout

        try:
            completion = self.client.chat.completions.create(
                model=model_name, messages=messages, **api_kwargs
            )
            # Store raw response from client
            self.last_ai_response = completion
        except Exception as exc:
            # Import openai exceptions for type checking
            from openai import (
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
                RateLimitError,
            )

            if isinstance(exc, APITimeoutError):
                error_msg = f'OpenAI API timeout ({api_kwargs["timeout"]}) error: {exc} for model {model_name}'
                logger.error(error_msg)
                raise
            elif isinstance(
                exc, (AuthenticationError, RateLimitError, BadRequestError)
            ):
                error_msg = f'OpenAI API error ({type(exc).__name__}): {exc}'
                logger.error(error_msg)
                raise
            elif isinstance(exc, ValueError):
                logger.error(f'ValueError during API call: {exc}')
                raise
            else:
                is_length_error = 'Length' in str(
                    exc
                ) or 'maximum context length' in str(exc)
                if is_length_error:
                    raise ValueError(
                        f'Input too long for model {model_name}. Error: {str(exc)[:100]}...'
                    ) from exc
                raise
        # print(completion)

        results: list[dict[str, Any]] = []
        for choice in completion.choices:
            assistant_message = [
                {'role': 'assistant', 'content': choice.message.content}
            ]
            try:
                reasoning_content = choice.message.reasoning
            except:
                reasoning_content = None
            if reasoning_content:
                assistant_message[0]['reasoning_content'] = reasoning_content

            choice_messages = cast(
                Messages,
                messages + assistant_message,
            )
            result_dict = {
                'parsed': choice.message.content,
                'messages': choice_messages,
            }

            results.append(result_dict)
        return results

    def stream_text_completion(
        self, input_data: str | BaseModel | list[dict], **runtime_kwargs
    ) -> Iterator[str]:
        """
        Stream text completion with caching support.

        If response is cached, yields tokenized chunks from the cache.
        If response is not cached, streams directly from the API.

        Args:
            input_data: Input data (string, BaseModel, or message list)
            **runtime_kwargs: Additional runtime parameters

        Yields:
            Token strings from the completion
        """
        messages = self._prepare_input(input_data)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name = effective_kwargs.get('model', self.model_kwargs['model'])

        # Remove stream from effective_kwargs if present to avoid conflicts
        api_kwargs = {
            k: v for k, v in effective_kwargs.items() if k not in ('model', 'stream')
        }

        if 'timeout' not in api_kwargs and self.timeout is not None:
            api_kwargs['timeout'] = self.timeout

        # Try cache first by making a non-streaming call
        cache_hit = False
        cached_text = None

        try:
            # Attempt non-streaming call to check cache
            cached_completion = self.client.chat.completions.create(
                model=model_name, messages=messages, **api_kwargs
            )
            cache_hit = True
            cached_text = cached_completion.choices[0].message.content
            self.last_ai_response = cached_completion
        except Exception as e:
            # If cache lookup fails, we'll try streaming from API
            cache_hit = False
            logger.debug(f'Cache lookup failed, attempting stream from API: {e}')

        if cache_hit and cached_text:
            # Stream from cached response by tokenizing
            yield from self._stream_from_cache(cached_text, model_name)
        else:
            # Stream directly from API
            try:
                stream_response = self.client.chat.completions.create(
                    model=model_name, messages=messages, stream=True, **api_kwargs
                )
                for chunk in stream_response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield delta.content
            except Exception as exc:
                from openai import (
                    APITimeoutError,
                    AuthenticationError,
                    BadRequestError,
                    RateLimitError,
                )

                if isinstance(exc, APITimeoutError):
                    error_msg = f'OpenAI API timeout ({api_kwargs.get("timeout")}) error: {exc} for model {model_name}'
                    logger.error(error_msg)
                    raise
                elif isinstance(
                    exc, (AuthenticationError, RateLimitError, BadRequestError)
                ):
                    error_msg = f'OpenAI API error ({type(exc).__name__}): {exc}'
                    logger.error(error_msg)
                    raise
                else:
                    is_length_error = 'Length' in str(
                        exc
                    ) or 'maximum context length' in str(exc)
                    if is_length_error:
                        raise ValueError(
                            f'Input too long for model {model_name}. Error: {str(exc)[:100]}...'
                        ) from exc
                    raise

    def _stream_from_cache(self, text: str, model_name: str) -> Iterator[str]:
        """
        Stream cached response text by tokenizing it using the model's tokenizer.

        Args:
            text: The cached response text
            model_name: Name of the model for tokenization

        Yields:
            Token strings from the cached text
        """
        try:
            # Use TokenizationMixin.encode to tokenize the text
            tokens, token_strs = self.encode(
                text, add_special_tokens=False, return_token_strs=True
            )
            # Yield each token string
            for token in token_strs:
                yield token
        except Exception as e:
            # Fallback: if tokenization fails, yield the text as a single chunk
            logger.warning(
                f'Failed to tokenize cached response: {e}. Yielding full text as single chunk.'
            )
            yield text

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think> tags if present, returning only the reasoning body."""
        cleaned = text.strip()
        if cleaned.startswith('<think>'):
            cleaned = cleaned[len('<think>') :].lstrip()
        if '</think>' in cleaned:
            cleaned = cleaned.split('</think>', 1)[0].rstrip()
        return cleaned

    def generate_with_think_prefix(
        self, input_data: str | BaseModel | list[dict], **runtime_kwargs
    ) -> list[dict[str, Any]]:
        """
        Generate text and format output as:
        <think>reasoning</think>
        """
        results = self.text_completion(input_data, **runtime_kwargs)

        for result in results:
            content = result.get('parsed') or ''
            reasoning = result.get('reasoning_content') or ''

            if not reasoning and str(content).lstrip().startswith('<think>'):
                formatted = str(content)
            else:
                reasoning_body = self._strip_think_tags(str(reasoning))
                formatted = (
                    f'<think>\n{reasoning_body}\n</think>\n\n{str(content).lstrip()}'
                )

            result['parsed'] = formatted
            messages = result.get('messages')
            if isinstance(messages, list) and messages:
                last_msg = messages[-1]
                if isinstance(last_msg, dict) and last_msg.get('role') == 'assistant':
                    last_msg['content'] = formatted

        return results

    @clean_traceback
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

        if 'timeout' not in api_kwargs and self.timeout is not None:
            api_kwargs['timeout'] = self.timeout

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
        except Exception as exc:
            # Import openai exceptions for type checking
            from openai import AuthenticationError, BadRequestError, RateLimitError

            if isinstance(exc, (AuthenticationError, RateLimitError, BadRequestError)):
                error_msg = f'OpenAI API error ({type(exc).__name__}): {exc}'
                logger.error(error_msg)
                raise
            is_length_error = 'Length' in str(exc) or 'maximum context length' in str(
                exc
            )
            if is_length_error:
                raise ValueError(
                    f'Input too long for model {model_name}. Error: {str(exc)[:100]}...'
                ) from exc
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
        stream: bool = False,
        **openai_client_kwargs,
    ) -> list[dict[str, Any]] | Iterator[str]:
        """
        Execute LLM task.

        Args:
            input_data: Input data (string, BaseModel, or message list)
            response_model: Optional response model override
            two_step_parse_pydantic: Use two-step parsing (text then parse)
            temperature_ranges: If set, tuple of (min_temp, max_temp) to sample
            n: Number of temperature samples (only used with temperature_ranges, must be >= 2)
            stream: If True, stream text completion with cache support (only for text output)
            cache: Whether to use caching (default None uses instance setting)
            **runtime_kwargs: Additional runtime parameters

        Returns:
            List of response dictionaries, or Iterator[str] if stream=True
        """
        if cache is not None:
            if hasattr(self.client, 'set_cache'):
                self.client.set_cache(cache)
            else:
                logger.warning('Client does not support caching.')

        # Handle streaming (only for text completion)
        if stream:
            pydantic_model = response_model or self.output_model
            if pydantic_model not in (str, None):
                raise ValueError(
                    'Streaming is only supported for text completions, not structured outputs. '
                    'Set response_model=None or response_model=str to use streaming.'
                )
            return self.stream_text_completion(input_data, **openai_client_kwargs)

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
            from llm_utils import show_chat

            conv = self._last_conversations[idx]
            if k_last_messages > 0:
                conv = conv[-k_last_messages:]
            return show_chat(conv)
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
        client: 'OpenAI | int | str | None' = None,  # type: ignore[name-defined]
        cache=True,
        is_reasoning_model: bool = False,
        lora_path: str | None = None,
        vllm_cmd: str | None = None,
        vllm_timeout: int = 0.1,
        vllm_reuse: bool = True,
        timeout: float | Timeout | None = None,
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
            timeout: Optional OpenAI client timeout in seconds
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
            timeout=timeout,
            **model_kwargs,
        )


class LLM_NEMOTRON3(LLM):
    """
    Custom implementation for NVIDIA Nemotron-3 reasoning models.
    Supports thinking budget control and native reasoning tags.
    """

    def __init__(
        self,
        model: str = 'nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
        thinking_budget: int = 1024,
        enable_thinking: bool = True,
        **kwargs,
    ):
        # Force reasoning_model to True to enable reasoning_content extraction
        kwargs['is_reasoning_model'] = True
        super().__init__(**kwargs)

        self.model_kwargs['model'] = model
        self.thinking_budget = thinking_budget
        self.enable_thinking = enable_thinking

    def _prepare_input(self, input_data: str | BaseModel | list[dict]) -> Messages:
        """Override to ensure Nemotron chat template requirements are met."""
        messages = super()._prepare_input(input_data)
        return messages

    def __call__(
        self,
        input_data: str | BaseModel | list[dict],
        thinking_budget: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        budget = thinking_budget or self.thinking_budget

        if not self.enable_thinking:
            # Simple pass with thinking disabled in template
            return super().__call__(
                input_data, chat_template_kwargs={'enable_thinking': False}, **kwargs
            )

        # --- STEP 1: Generate Thinking Trace ---
        # We manually append <think> to force the reasoning MoE layers
        messages = self._prepare_input(input_data)

        # We use the raw text completion for the budget phase
        # Stop at the closing tag or budget limit
        thinking_response = self.text_completion(
            input_data, max_tokens=budget, stop=['</think>'], **kwargs
        )[0]

        reasoning_content = thinking_response['parsed']

        # Ensure proper tag closing for the second pass
        if '</think>' not in reasoning_content:
            reasoning_content = f'{reasoning_content}\n</think>'
        elif not reasoning_content.endswith('</think>'):
            # Ensure it ends exactly with the tag for continuity
            reasoning_content = reasoning_content.split('</think>')[0] + '</think>'

        # --- STEP 2: Generate Final Answer ---
        # Append the thought to the assistant role and continue
        final_messages = messages + [
            {'role': 'assistant', 'content': f'<think>\n{reasoning_content}\n'}
        ]

        # Use continue_final_message to prevent the model from repeating the header
        results = super().__call__(
            final_messages, extra_body={'continue_final_message': True}, **kwargs
        )

        # Inject the reasoning back into the result for the UI/API
        for res in results:
            res['reasoning_content'] = reasoning_content

        return results
