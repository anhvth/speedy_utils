from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

from loguru import logger
from pydantic import BaseModel

from speedy_utils import clean_traceback

from .llm import LLM, Messages


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessage
    from openai.types.completion import CompletionChoice


ASSISTANT_PREFIX = "<|im_start|>assistant"
THINK_START = "<think>"
THINK_END = "</think>"
ASSISTANT_END = "<|im_end|>"
DEFAULT_THINKING_MAX_TOKENS = 8192
DEFAULT_CONTENT_MAX_TOKENS = 2048
TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV = "TRANSFORMERS_NO_ADVISORY_WARNINGS"


class _PrefixCompletionState(NamedTuple):
    assistant_prompt_prefix: str
    reasoning: str | None
    content: str | None
    think_done: bool
    stop_reason: str | None
    call_count: int
    usage: Any | None


def split_assistant_parts(text: str) -> tuple[str | None, str | None, bool]:
    """Split an assistant prefix into reasoning, content, and think completion."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if not text.startswith(ASSISTANT_PREFIX):
        text = f"{ASSISTANT_PREFIX}{text}"

    body = text[len(ASSISTANT_PREFIX) :].lstrip()

    reasoning: str | None = None
    content: str | None = None
    think_done = False

    if body.startswith(THINK_START):
        body = body[len(THINK_START) :]
        if THINK_END in body:
            reasoning, content = body.split(THINK_END, 1)
            think_done = True
        else:
            reasoning = body
            think_done = False
    else:
        content = body
        think_done = True

    if reasoning is not None:
        reasoning = reasoning.strip()

    if content is not None:
        content = content.strip()
        if content.endswith(ASSISTANT_END):
            content = content[: -len(ASSISTANT_END)].rstrip()

    return reasoning, content, think_done


def build_assistant_prefix(
    reasoning: str | None,
    content: str | None,
    think_done: bool,
) -> str:
    """Build a normalized assistant prefix for prefix continuation."""
    reasoning_text = "" if reasoning is None else str(reasoning)
    content_text = "" if content is None else str(content)

    if think_done:
        return (
            f"{ASSISTANT_PREFIX}\n{THINK_START}\n"
            f"{reasoning_text}\n{THINK_END}{content_text}"
        )

    return f"{ASSISTANT_PREFIX}\n{THINK_START}\n{reasoning_text}"


def is_content_done(content: str | None, stop_reason: str | None) -> bool:
    """Return True when the model produced content and stopped normally."""
    return content not in ("", None) and stop_reason == "stop"


def strip_assistant_end(text: str) -> str:
    """Remove the assistant end token when the backend appends it."""
    if text.endswith(ASSISTANT_END):
        return text[: -len(ASSISTANT_END)]
    return text


class Qwen3LLM(LLM):
    """Qwen3 helper for staged generation with a partial assistant prefix."""

    TOKENIZER_NAME: ClassVar[str] = "Qwen/Qwen3-0.6B"
    _tokenizer: ClassVar[Any | None] = None

    def __init__(self, *args, enable_thinking: bool = True, **kwargs):
        super().__init__(*args, enable_thinking=enable_thinking, **kwargs)

    @classmethod
    def _get_tokenizer(cls):
        tokenizer = cls._tokenizer
        if tokenizer is None:
            # Transformers emits an import-time advisory when no backend
            # framework is installed; suppress it only while loading the
            # tokenizer helper.
            previous = os.environ.get(TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV)
            os.environ[TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV] = "1"
            try:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    cls.TOKENIZER_NAME,
                    trust_remote_code=True,
                )
            finally:
                if previous is None:
                    os.environ.pop(TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV, None)
                else:
                    os.environ[TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV] = previous
            cls._tokenizer = tokenizer
        return tokenizer

    @staticmethod
    def _validate_prefix_completion_kwargs(
        *,
        n: int,
        thinking_max_tokens: int | None = None,
        content_max_tokens: int | None = None,
        require_thinking_max_tokens: bool = False,
        require_content_max_tokens: bool = False,
    ) -> None:
        if n != 1:
            raise ValueError("prefix completion only supports n=1")

        if thinking_max_tokens is None:
            if require_thinking_max_tokens:
                raise ValueError("thinking_max_tokens must not be None")
        elif thinking_max_tokens <= 0:
            raise ValueError("thinking_max_tokens must be > 0")

        if content_max_tokens is None:
            if require_content_max_tokens:
                raise ValueError("content_max_tokens must not be None")
        elif content_max_tokens <= 0:
            raise ValueError("content_max_tokens must be > 0")

    @staticmethod
    def _build_prefix_state(
        assistant_prompt_prefix: str,
        *,
        stop_reason: str | None = None,
        call_count: int = 0,
        usage: Any | None = None,
    ) -> _PrefixCompletionState:
        reasoning, content, think_done = split_assistant_parts(assistant_prompt_prefix)
        return _PrefixCompletionState(
            assistant_prompt_prefix=build_assistant_prefix(
                reasoning,
                content,
                think_done,
            ),
            reasoning=reasoning,
            content=content,
            think_done=think_done,
            stop_reason=stop_reason,
            call_count=call_count,
            usage=usage,
        )

    def _generate_with_prefix_step(
        self,
        messages: Messages,
        assistant_prompt_prefix: str,
        **runtime_kwargs,
    ) -> "CompletionChoice":
        tokenizer = self._get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_to_continue = prompt + assistant_prompt_prefix

        call_kwargs = {k: v for k, v in runtime_kwargs.items() if k != "extra_body"}
        cache = call_kwargs.pop("cache", None)
        enable_thinking = call_kwargs.pop("enable_thinking", None)
        self._set_cache(cache)
        self._require_single_choice(call_kwargs)
        if call_kwargs.get("max_tokens") is None:
            call_kwargs["max_tokens"] = 1

        effective_kwargs = {**self.model_kwargs, **call_kwargs}
        model_name, api_kwargs = self._build_api_kwargs(
            effective_kwargs,
            enable_thinking=enable_thinking,
        )

        try:
            completion = self.client.completions.create(
                model=model_name,
                prompt=prompt_to_continue,
                **api_kwargs,
            )
            self.last_ai_response = completion
        except Exception as exc:
            from openai import (
                APITimeoutError,
                AuthenticationError,
                BadRequestError,
                RateLimitError,
            )

            if isinstance(exc, APITimeoutError):
                error_msg = f"OpenAI API timeout ({api_kwargs['timeout']}) error: {exc} for model {model_name}"
                logger.error(error_msg)
                raise
            if isinstance(exc, (AuthenticationError, RateLimitError, BadRequestError)):
                error_msg = f"OpenAI API error ({type(exc).__name__}): {exc}"
                logger.error(error_msg)
                raise
            if isinstance(exc, ValueError):
                logger.error(f"ValueError during API call: {exc}")
                raise
            is_length_error = "Length" in str(exc) or "maximum context length" in str(
                exc
            )
            if is_length_error:
                raise ValueError(
                    f"Input too long for model {model_name}. Error: {str(exc)[:100]}..."
                ) from exc
            raise

        choice = self._get_completion_choice(completion)
        usage = self._get_completion_usage(completion)
        if usage is not None:
            choice.usage = usage

        result = {
            "text": choice.text,
            "finish_reason": choice.finish_reason,
        }
        if getattr(choice, "usage", None) is not None:
            result["usage"] = choice.usage
        if os.environ.get("LLM_UTILS_DEBUG_PREFIX_COMPLETION", "0") == "1":
            print(
                f"INPUT:\n```{prompt_to_continue}```\n----\nOUTPUT:\n```{result}```\n{'-' * 40}",
            )

        state = self._build_prefix_state(
            assistant_prompt_prefix + strip_assistant_end(str(choice.text or "")),
            stop_reason=choice.finish_reason,
        )
        choice_messages = cast(
            Messages,
            messages
            + [
                self._build_openai_message(
                    reasoning=state.reasoning,
                    content=state.content or "",
                ).model_dump(exclude_none=True),
            ],
        )
        self._record_history(choice_messages)
        return choice

    def _complete_reasoning(
        self,
        messages: Messages,
        assistant_prompt_prefix: str,
        *,
        thinking_max_tokens: int,
        **runtime_kwargs,
    ) -> _PrefixCompletionState:
        state = self._build_prefix_state(assistant_prompt_prefix)
        if state.think_done:
            return state._replace(stop_reason=None)

        choice = self._generate_with_prefix_step(
            messages,
            state.assistant_prompt_prefix,
            max_tokens=thinking_max_tokens,
            **runtime_kwargs,
        )

        state = self._build_prefix_state(
            state.assistant_prompt_prefix + strip_assistant_end(str(choice.text or "")),
            stop_reason=choice.finish_reason,
            call_count=state.call_count + 1,
            usage=getattr(choice, "usage", None),
        )

        if not state.think_done:
            state = self._build_prefix_state(
                state.assistant_prompt_prefix + "\n</think>\n\n",
                stop_reason=choice.finish_reason,
                call_count=state.call_count,
                usage=state.usage,
            )

        return state

    def _complete_content(
        self,
        messages: Messages,
        completion_state: _PrefixCompletionState,
        *,
        content_max_tokens: int,
        **runtime_kwargs,
    ) -> "ChatCompletionMessage":
        choice = self._generate_with_prefix_step(
            messages,
            completion_state.assistant_prompt_prefix,
            max_tokens=content_max_tokens,
            **runtime_kwargs,
        )

        state = self._build_prefix_state(
            completion_state.assistant_prompt_prefix
            + strip_assistant_end(str(choice.text or "")),
            usage=getattr(choice, "usage", None),
        )
        message = self._build_openai_message(
            reasoning=state.reasoning,
            content=state.content or "",
            usage=state.usage,
        )
        message.call_count = completion_state.call_count + 1
        return message

    def _build_openai_message(
        self,
        content: str,
        reasoning: str | None,
        usage: Any | None = None,
    ) -> "ChatCompletionMessage":
        from openai.types.chat import ChatCompletionMessage

        message_kwargs = {
            "role": "assistant",
            "content": content,
        }
        if reasoning:
            message_kwargs["reasoning_content"] = reasoning
        message = ChatCompletionMessage(**message_kwargs)
        if usage is not None:
            message.usage = usage
        return message

    def complete_reasoning(
        self,
        input_data: str | BaseModel | list[dict],
        assistant_prompt_prefix: str = "<think>\n",
        *,
        thinking_max_tokens: int = DEFAULT_THINKING_MAX_TOKENS,
        **runtime_kwargs,
    ) -> _PrefixCompletionState:
        """
        Complete the reasoning phase and return the updated prefix state.

        The returned state can be passed to `complete_content()` to finish the
        answer generation.
        """
        self._validate_prefix_completion_kwargs(
            n=runtime_kwargs.get("n", 1),
            thinking_max_tokens=thinking_max_tokens,
            require_thinking_max_tokens=True,
        )

        messages = self._prepare_input(input_data)
        return self._complete_reasoning(
            messages,
            assistant_prompt_prefix,
            thinking_max_tokens=thinking_max_tokens,
            **runtime_kwargs,
        )

    def complete_content(
        self,
        input_data: str | BaseModel | list[dict],
        completion_state: _PrefixCompletionState | str,
        *,
        content_max_tokens: int = DEFAULT_CONTENT_MAX_TOKENS,
        **runtime_kwargs,
    ) -> "ChatCompletionMessage":
        """
        Complete the visible answer from an already completed reasoning state.
        """
        self._validate_prefix_completion_kwargs(
            n=runtime_kwargs.get("n", 1),
            content_max_tokens=content_max_tokens,
            require_content_max_tokens=True,
        )

        if isinstance(completion_state, str):
            completion_state = self._build_prefix_state(completion_state)

        messages = self._prepare_input(input_data)
        return self._complete_content(
            messages,
            completion_state,
            content_max_tokens=content_max_tokens,
            **runtime_kwargs,
        )

    @clean_traceback
    def chat_completion(
        self,
        input_data: str | BaseModel | list[dict],
        assistant_prompt_prefix: str = "<think>\n",
        *,
        thinking_max_tokens: int = DEFAULT_THINKING_MAX_TOKENS,
        content_max_tokens: int = DEFAULT_CONTENT_MAX_TOKENS,
        **runtime_kwargs,
    ) -> "ChatCompletionMessage":
        """
        Continue generation from a partial assistant prefix.

        This is the primary Qwen3 prefix-completion entry point. It seeds or
        continues a `<think>...</think>` block before the final answer.
        """
        self._validate_prefix_completion_kwargs(
            n=runtime_kwargs.get("n", 1),
            thinking_max_tokens=thinking_max_tokens,
            content_max_tokens=content_max_tokens,
            require_thinking_max_tokens=True,
            require_content_max_tokens=True,
        )

        messages = self._prepare_input(input_data)
        reasoning_state = self._complete_reasoning(
            messages,
            assistant_prompt_prefix,
            thinking_max_tokens=thinking_max_tokens,
            **runtime_kwargs,
        )
        if is_content_done(
            reasoning_state.content,
            reasoning_state.stop_reason,
        ):
            message = self._build_openai_message(
                reasoning=reasoning_state.reasoning,
                content=reasoning_state.content,
                usage=reasoning_state.usage,
            )
            message.call_count = reasoning_state.call_count
            return message

        return self._complete_content(
            messages,
            reasoning_state,
            content_max_tokens=content_max_tokens,
            **runtime_kwargs,
        )
