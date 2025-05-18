import random
import time
from typing import Any, List, Optional, Type, Union, cast, TypeVar, Generic

from openai import AuthenticationError, RateLimitError
from pydantic import BaseModel

from speedy_utils.common.logger import logger
from speedy_utils.common.utils_cache import identify_uuid

from .base_lm import LM

T = TypeVar("T", bound=BaseModel)


class PydanticLM(LM):
    """
    Language model that returns outputs as Pydantic models.
    """

    def _generate_cache_key(
        self,
        messages: List[Any],
        response_format: Optional[Type[BaseModel]],
        kwargs: dict,
    ) -> str:
        """
        Generate a cache key based on input parameters.
        """
        cache_key_base = self._generate_cache_key_base(messages, kwargs)
        cache_key_base.insert(
            1, (response_format.model_json_schema() if response_format else None)
        )
        return identify_uuid(str(cache_key_base))

    def _parse_cached_result(
        self, cached_result: Any, response_format: Optional[Type[BaseModel]]
    ) -> Optional[BaseModel]:
        """
        Parse cached result into a BaseModel instance.
        """
        if isinstance(cached_result, BaseModel):
            return cached_result
        elif isinstance(cached_result, str):
            if response_format is None:
                raise ValueError(
                    "response_format must be provided to parse cached string result."
                )
            import json

            return response_format.model_validate_json(cached_result)
        elif (
            isinstance(cached_result, list)
            and cached_result
            and isinstance(cached_result[0], (str, BaseModel))
        ):
            first = cached_result[0]
            if isinstance(first, BaseModel):
                return first
            elif isinstance(first, str):
                if response_format is None:
                    raise ValueError(
                        "response_format must be provided to parse cached string result."
                    )
                import json

                return response_format.model_validate_json(first)
        else:
            logger.warning(
                f"Cached result has unexpected type {type(cached_result)}. Ignoring cache."
            )
            return None

    def _check_cache(
        self,
        effective_cache: bool,
        messages: List[Any],
        response_format: Optional[Type[BaseModel]],
        effective_kwargs: dict,
    ):
        """Check if result is in cache and return it if available."""
        if not effective_cache:
            return None, None

        cache_id = self._generate_cache_key(messages, response_format, effective_kwargs)
        cached_result = self.load_cache(cache_id)
        parsed_cache = self._parse_cached_result(cached_result, response_format)

        return cache_id, parsed_cache

    def _call_llm(
        self,
        dspy_main_input: List[Any],
        response_format: Type[BaseModel],
        current_port: Optional[int],
        use_loadbalance: Optional[bool],
        **kwargs,
    ):
        """Call the LLM with response format support using OpenAI's parse method."""
        # Use messages directly
        messages = dspy_main_input

        # Use OpenAI's parse method for structured output
        try:
            response = self.openai_client.beta.chat.completions.parse(
                model=self.model,
                messages=messages,
                response_format=response_format,
                **kwargs,
            )
        except AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise
        except TimeoutError as e:
            logger.error(f"Timeout error: {e}")
            raise
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise
        # Update port usage stats if needed
        if current_port and use_loadbalance is True:
            self._update_port_use(current_port, -1)

        return response.choices[0].message.parsed

    def _parse_llm_output(
        self, llm_output: Any, response_format: Optional[Type[BaseModel]]
    ) -> BaseModel:
        """Parse the LLM output into the correct format."""
        if isinstance(llm_output, BaseModel):
            return llm_output
        elif isinstance(llm_output, dict):
            if not response_format:
                raise ValueError("response_format required to parse dict output.")
            return response_format.model_validate(llm_output)
        elif isinstance(llm_output, str):
            if not response_format:
                raise ValueError("response_format required to parse string output.")
            import json

            return response_format.model_validate_json(llm_output)
        else:
            if not response_format:
                raise ValueError("response_format required to parse output.")
            return response_format.model_validate_json(str(llm_output))

    def _store_in_cache(
        self, effective_cache: bool, cache_id: Optional[str], result: BaseModel
    ):
        """Store the result in cache if caching is enabled."""
        if result and isinstance(result, BaseModel):
            self._store_in_cache_base(
                effective_cache, cache_id, result.model_dump_json()
            )

    def forward_messages(
        self,
        response_format: Type[T],
        messages: List[Any],
        cache: Optional[bool] = None,
        port: Optional[int] = None,
        use_loadbalance: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> T:
        # 1. Prepare inputs
        effective_kwargs, effective_cache, current_port, dspy_main_input = (
            self._prepare_call_inputs(
                messages, max_tokens, port, use_loadbalance, cache, **kwargs
            )
        )

        # 2. Check cache
        cache_id, cached_result = self._check_cache(
            effective_cache, messages, response_format, effective_kwargs
        )
        if cached_result:
            return cast(T, cached_result)

        # 3. Call LLM using OpenAI's parse method
        llm_output = self._call_llm(
            dspy_main_input,
            response_format,
            current_port,
            use_loadbalance,
            **effective_kwargs,
        )

        # 4. Parse output
        result = self._parse_llm_output(llm_output, response_format)

        # 5. Store in cache
        self._store_in_cache(effective_cache, cache_id, result)

        return cast(T, result)
