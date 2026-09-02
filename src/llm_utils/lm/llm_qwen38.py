"""Model-specific wrapper for Qwen3.8 chat completions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from httpx import Timeout

from .llm import LLM


if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessage
    from pydantic import BaseModel


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
NON_REASONING_EFFORT: ReasoningEffort = "none"

DEFAULT_QWEN38_MODEL = "Qwen3.8-27B"
DEFAULT_REASONING_EFFORT: ReasoningEffort = "medium"


class Qwen38LLM(LLM):
    """Small model-specific shim adding a typed `reasoning_effort` default."""

    def __init__(
        self,
        client: "OpenAI | int | str | list | None" = None,  # type: ignore[name-defined]
        cache: bool = True,
        verbose: bool = False,
        timeout: float | Timeout | None = None,
        enable_thinking: bool | None = None,
        *,
        model: str | None = DEFAULT_QWEN38_MODEL,
        reasoning_effort: ReasoningEffort | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | tuple[str, ...] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        **model_kwargs: Any,
    ):
        self._reasoning_effort_default: ReasoningEffort | None = reasoning_effort
        if reasoning_effort is not None:
            model_kwargs["reasoning_effort"] = reasoning_effort

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

    def _resolve_reasoning_effort(
        self,
        reasoning_effort: ReasoningEffort | None,
        enable_thinking: bool | None,
    ) -> ReasoningEffort:
        if reasoning_effort is not None:
            return reasoning_effort

        if enable_thinking is not None:
            return (
                NON_REASONING_EFFORT
                if not enable_thinking
                else DEFAULT_REASONING_EFFORT
            )

        if self.enable_thinking is not None:
            return (
                NON_REASONING_EFFORT
                if self.enable_thinking is False
                else DEFAULT_REASONING_EFFORT
            )

        if self._reasoning_effort_default is not None:
            return self._reasoning_effort_default

        return DEFAULT_REASONING_EFFORT

    def get_model_sampling_params(
        self,
        reasoning_effort: ReasoningEffort | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        """Return resolved runtime sampling parameters for this model."""
        resolved_reasoning_effort = self._resolve_reasoning_effort(
            reasoning_effort,
            enable_thinking,
        )
        params = {
            "reasoning_effort": resolved_reasoning_effort,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        return {k: v for k, v in params.items() if v is not None}

    @classmethod
    def _with_reasoning_effort(
        cls,
        runtime_kwargs: dict[str, Any],
        reasoning_effort: ReasoningEffort,
    ) -> dict[str, Any]:
        next_kwargs = dict(runtime_kwargs)
        next_kwargs["reasoning_effort"] = reasoning_effort
        return next_kwargs

    def chat_completion(
        self,
        input_data: str | BaseModel | list[dict],
        *,
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        **runtime_kwargs,
    ) -> "ChatCompletionMessage":
        return super().chat_completion(
            input_data,
            cache=cache,
            enable_thinking=enable_thinking,
            **self._with_reasoning_effort(
                runtime_kwargs,
                self.get_model_sampling_params(reasoning_effort, enable_thinking)[
                    "reasoning_effort"
                ],
            ),
        )
