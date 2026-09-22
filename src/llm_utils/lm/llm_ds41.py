"""Model-specific wrapper for DeepSeek V4.1 chat completions."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, cast

from httpx import Timeout

from .llm import LLM


if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletionMessage
    from pydantic import BaseModel


ReasoningEffortName = Literal["low", "high", "max"]
ReasoningEffort: TypeAlias = int | ReasoningEffortName

DEFAULT_DS41_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_REASONING_EFFORT: ReasoningEffortName = "high"


class DS41LLM(LLM):
    """DeepSeek V4.1 shim with explicit thinking and effort controls."""

    TOKENIZER_NAME: ClassVar[str] = DEFAULT_DS41_MODEL
    REASONING_EFFORT_MAPPINGS: ClassVar[dict[ReasoningEffortName, int]] = {
        "low": 50,
        "high": 75,
        "max": 100,
    }

    def __init__(
        self,
        client: "OpenAI | int | str | list | None" = None,  # type: ignore[name-defined]
        cache: bool = True,
        verbose: bool = False,
        timeout: float | Timeout | None = None,
        enable_thinking: bool | None = True,
        *,
        model: str | None = DEFAULT_DS41_MODEL,
        reasoning_effort: ReasoningEffort = DEFAULT_REASONING_EFFORT,
        max_tokens: int | None = None,
        temperature: float | None = 1.0,
        top_p: float | None = 0.95,
        stop: str | list[str] | tuple[str, ...] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        **model_kwargs: Any,
    ):
        self._reasoning_effort_default: ReasoningEffort = (
            self._validate_reasoning_effort(reasoning_effort)
        )
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

    @classmethod
    def _validate_reasoning_effort(
        cls,
        reasoning_effort: Any,
    ) -> ReasoningEffort:
        if isinstance(reasoning_effort, bool):
            raise ValueError("reasoning_effort must be an integer from 1 to 100")
        if isinstance(reasoning_effort, int):
            if 1 <= reasoning_effort <= 100:
                return reasoning_effort
            raise ValueError("reasoning_effort must be an integer from 1 to 100")
        if (
            isinstance(reasoning_effort, str)
            and reasoning_effort in cls.REASONING_EFFORT_MAPPINGS
        ):
            return cast(ReasoningEffortName, reasoning_effort)
        raise ValueError("reasoning_effort must be 1-100 or one of: low, high, max")

    def _resolve_reasoning_effort(
        self,
        reasoning_effort: ReasoningEffort | None,
    ) -> ReasoningEffort:
        if reasoning_effort is None:
            return self._reasoning_effort_default
        return self._validate_reasoning_effort(reasoning_effort)

    @staticmethod
    def _extract_reasoning(message: Any) -> str | None:
        reasoning = LLM._extract_reasoning(message)
        if reasoning is not None:
            return reasoning
        reasoning_content = getattr(message, "reasoning_content", None)
        return reasoning_content if isinstance(reasoning_content, str) else None

    def _reasoning_is_enabled(
        self,
        enable_thinking: bool | None,
        runtime_kwargs: dict[str, Any],
    ) -> bool:
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        extra_body = effective_kwargs.get("extra_body") or {}
        chat_template_kwargs = extra_body.get("chat_template_kwargs") or {}
        if "thinking" in chat_template_kwargs:
            return chat_template_kwargs["thinking"] is not False
        if "enable_thinking" in chat_template_kwargs:
            return chat_template_kwargs["enable_thinking"] is not False
        effective_enable_thinking = (
            self.enable_thinking if enable_thinking is None else enable_thinking
        )
        return effective_enable_thinking is not False

    @staticmethod
    def _raise_if_disabled_reasoning_was_emitted(
        results: list[dict[str, Any]],
        *,
        reasoning_enabled: bool,
    ) -> None:
        if reasoning_enabled:
            return
        if any(result.get("reasoning") for result in results):
            raise RuntimeError(
                "DeepSeek V4.1 emitted reasoning although reasoning was disabled"
            )

    def _chat_completion_result(
        self,
        input_data: str | BaseModel | list[dict],
        *,
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        **runtime_kwargs: Any,
    ) -> list[dict[str, Any]]:
        runtime_kwargs = dict(runtime_kwargs)
        if "reasoning_effort" in runtime_kwargs:
            runtime_kwargs["reasoning_effort"] = self._validate_reasoning_effort(
                runtime_kwargs["reasoning_effort"]
            )
        reasoning_enabled = self._reasoning_is_enabled(
            enable_thinking,
            runtime_kwargs,
        )
        results = super()._chat_completion_result(
            input_data,
            cache=cache,
            enable_thinking=enable_thinking,
            **runtime_kwargs,
        )
        self._raise_if_disabled_reasoning_was_emitted(
            results,
            reasoning_enabled=reasoning_enabled,
        )
        return results

    def get_model_sampling_params(
        self,
        reasoning_effort: ReasoningEffort | None = None,
        enable_thinking: bool | None = None,
    ) -> dict[str, Any]:
        """Return the resolved DeepSeek V4.1 request settings."""
        effective_enable_thinking = (
            self.enable_thinking if enable_thinking is None else enable_thinking
        )
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "stop": self.stop,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
        }
        if effective_enable_thinking is not False:
            params = {
                "reasoning_effort": self._resolve_reasoning_effort(reasoning_effort),
                **params,
            }
        return {key: value for key, value in params.items() if value is not None}

    def _build_api_kwargs(
        self,
        effective_kwargs: dict[str, Any],
        *,
        model_name: str | None = None,
        enable_thinking: bool | None = None,
        drop_keys: tuple[str, ...] = (),
    ) -> tuple[str, dict[str, Any]]:
        request_kwargs = dict(effective_kwargs)
        reasoning_effort = self._resolve_reasoning_effort(
            request_kwargs.pop("reasoning_effort", None)
        )
        resolved_model, api_kwargs = super()._build_api_kwargs(
            request_kwargs,
            model_name=model_name,
            enable_thinking=enable_thinking,
            drop_keys=drop_keys,
        )

        extra_body = deepcopy(api_kwargs.get("extra_body") or {})
        chat_template_kwargs = deepcopy(extra_body.get("chat_template_kwargs") or {})
        thinking = chat_template_kwargs.get(
            "thinking",
            chat_template_kwargs.get("enable_thinking", self.enable_thinking),
        )
        if (
            "thinking" in chat_template_kwargs
            and "enable_thinking" in chat_template_kwargs
            and chat_template_kwargs["thinking"]
            != chat_template_kwargs["enable_thinking"]
        ):
            raise ValueError(
                "DeepSeek V4.1 thinking and enable_thinking settings must agree"
            )

        if thinking is False:
            chat_template_kwargs.pop("reasoning_effort", None)
        else:
            chat_template_kwargs["reasoning_effort"] = reasoning_effort
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        api_kwargs["extra_body"] = extra_body
        return resolved_model, api_kwargs

    def chat_completion(
        self,
        input_data: str | BaseModel | list[dict],
        *,
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        **runtime_kwargs: Any,
    ) -> "ChatCompletionMessage":
        if reasoning_effort is not None:
            runtime_kwargs["reasoning_effort"] = self._validate_reasoning_effort(
                reasoning_effort
            )
        result = self._chat_completion_result(
            input_data,
            cache=cache,
            enable_thinking=enable_thinking,
            **runtime_kwargs,
        )[0]
        message = result.get("message")
        if message is None:
            raise ValueError("No message returned from completion.")
        return cast("ChatCompletionMessage", message)
