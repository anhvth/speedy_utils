from typing import Any, List, Optional, Union, Tuple
from .base_lm import OAI_LM
import random
import logging
import json
from speedy_utils import identify_uuid

logger = logging.getLogger(__name__)


class TextLM(OAI_LM):
    """
    Language model that returns outputs as plain text (str).
    """

    def _generate_cache_key(
        self,
        prompt: Optional[str],
        messages: Optional[List[Any]],
        effective_kwargs: dict,
    ) -> str:
        """Generate a cache key based on input parameters."""
        cache_key_list = self._generate_cache_key_base(prompt, messages, effective_kwargs)
        return identify_uuid(str(cache_key_list))

    def _check_cache(
        self,
        effective_cache: bool,
        prompt: Optional[str],
        messages: Optional[List[Any]],
        effective_kwargs: dict,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Check if result is in cache and return it if available."""
        if not effective_cache:
            return None, None

        id_for_cache = self._generate_cache_key(prompt, messages, effective_kwargs)
        cached_result = self.load_cache(id_for_cache)

        if cached_result is not None:
            if isinstance(cached_result, str):
                return id_for_cache, cached_result
            elif (
                isinstance(cached_result, list)
                and cached_result
                and isinstance(cached_result[0], str)
            ):
                return id_for_cache, cached_result[0]
            else:
                logger.warning(
                    f"Cached result for {id_for_cache} has unexpected type {type(cached_result)}. Ignoring cache."
                )

        return id_for_cache, None

    def _parse_llm_output(self, llm_output: Any) -> str:
        """Parse the LLM output into the correct format."""
        if isinstance(llm_output, dict):
            return json.dumps(llm_output)

        if isinstance(llm_output, str):
            return llm_output
        else:
            logger.warning(
                f"LLM output type mismatch. Expected str, got {type(llm_output)}. Returning str."
            )
            return str(llm_output)

    def _store_in_cache(
        self, effective_cache: bool, id_for_cache: Optional[str], result: str
    ):
        """Store the result in cache if caching is enabled."""
        self._store_in_cache_base(effective_cache, id_for_cache, result)

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Any]] = None,
        cache: Optional[bool] = None,
        port: Optional[int] = None,
        use_loadbalance: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        # 1. Prepare inputs
        effective_kwargs, effective_cache, current_port, dspy_main_input = self._prepare_call_inputs(
            prompt, messages, max_tokens, port, use_loadbalance, cache, **kwargs
        )

        # 2. Check cache
        cache_id, cached_result = self._check_cache(
            effective_cache, prompt, messages, effective_kwargs
        )
        if cached_result:
            return cached_result

        # 3. Call LLM
        llm_output = self._call_llm(
            dspy_main_input, current_port, use_loadbalance, **effective_kwargs
        )

        # 4. Parse output
        result = self._parse_llm_output(llm_output)

        # 5. Store in cache
        self._store_in_cache(effective_cache, cache_id, result)

        return result
