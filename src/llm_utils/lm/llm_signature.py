"""LLM wrapper that binds a `Signature` to runtime generation."""

from httpx import Timeout
from typing import Any

from pydantic import BaseModel

from .llm import LLM
from .signature import Signature


class LLMSignature(LLM):
    """LLM wrapper that derives prompt and schema defaults from a Signature."""

    def __init__(
        self,
        signature: type[Signature],
        client: Any = None,
        cache: bool = True,
        verbose: bool = False,
        timeout: float | Timeout | None = None,
        enable_thinking: bool | None = None,
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | tuple[str, ...] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        **model_kwargs: Any,
    ):
        """
        Initialize a signature-backed LLM wrapper.

        Args:
            signature: Signature class for structured I/O
            client: OpenAI-compatible client, base URL, port, or list of clients
            cache: Whether to cache requests by default
            verbose: Whether to log available models and load-balancing details
            timeout: Optional request timeout forwarded to the client
            enable_thinking: Default thinking mode for supported backends
            model: Default model name
            max_tokens: Default generation token limit
            temperature: Default sampling temperature
            top_p: Default nucleus sampling value
            stop: Default stop sequence or sequences
            presence_penalty: Default OpenAI presence penalty
            frequency_penalty: Default OpenAI frequency penalty
            **model_kwargs: Additional provider-specific model options
        """
        legacy_keys = [
            key
            for key in (
                "input_model",
                "output_model",
                "response_model",
                "is_reasoning_model",
            )
            if key in model_kwargs
        ]
        if legacy_keys:
            raise TypeError(
                "LLMSignature no longer accepts legacy constructor arguments: "
                + ", ".join(legacy_keys)
            )
        self.signature = signature
        self.input_model = signature.get_input_model()
        self.output_model = signature.get_output_model()
        self.sft_data: list[dict[str, Any]] = []  # Store SFT training examples

        super().__init__(
            client=client,
            cache=cache,
            verbose=verbose,
            timeout=timeout,
            enable_thinking=enable_thinking,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            **model_kwargs,
        )

    def _resolve_response_model(
        self,
        response_model: type[BaseModel] | None,
    ) -> type[BaseModel]:
        if response_model is None:
            return self.output_model
        return response_model

    def pydantic_parse(
        self,
        input_data: str | list[dict],
        *,
        response_model: type[BaseModel] | None = None,
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        **runtime_kwargs,
    ) -> BaseModel:
        if not isinstance(input_data, str) and not isinstance(input_data, list):
            raise TypeError(
                "pydantic_parse expects `input_data` to be a string or message list."
            )
        return super().pydantic_parse(
            input_data,
            response_model=self._resolve_response_model(response_model),
            cache=cache,
            enable_thinking=enable_thinking,
            **runtime_kwargs,
        )

    def __call__(
        self,
        input_data: str | BaseModel | list[dict],
        response_model: type[BaseModel] | type[str] | None = None,
        n: int = 1,
        cache=None,
        stream: bool = False,
        enable_thinking: bool | None = None,
        return_dict: bool = False,
        **openai_client_kwargs,
    ) -> Any:
        return super().__call__(
            input_data,
            response_model=self._resolve_response_model(response_model),  # type: ignore[arg-type]
            n=n,
            cache=cache,
            stream=stream,
            enable_thinking=enable_thinking,
            return_dict=return_dict,
            **openai_client_kwargs,
        )
