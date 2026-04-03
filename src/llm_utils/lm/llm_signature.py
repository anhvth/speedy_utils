"""LLM wrapper that binds a `Signature` to runtime generation."""

from typing import Any

from pydantic import BaseModel

from .llm import LLM
from .signature import Signature


class LLMSignature(LLM):
    """LLM wrapper that derives prompt and schema defaults from a Signature."""

    def __init__(self, signature: type[Signature], **kwargs):
        """
        Initialize a signature-backed LLM wrapper.

        Args:
            signature: Signature class for structured I/O
            **kwargs: Additional arguments passed to LLM
        """
        legacy_keys = [
            key for key in ("input_model", "output_model", "response_model") if key in kwargs
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

        super().__init__(**kwargs)

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
            response_model=self._resolve_response_model(response_model),
            n=n,
            cache=cache,
            stream=stream,
            enable_thinking=enable_thinking,
            return_dict=return_dict,
            **openai_client_kwargs,
        )
