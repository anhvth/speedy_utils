from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from pydantic import BaseModel

from .llm import LLM, Messages


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessage


ASSISTANT_PREFIX = "<|im_start|>assistant"
THINK_START = "<think>"
THINK_END = "</think>"
ASSISTANT_END = "<|im_end|>"


class PrefixCompletionState(NamedTuple):
    assistant_prompt_prefix: str
    reasoning: str | None
    content: str | None
    think_done: bool
    stop_reason: str | None


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
    _TRANSFORMERS_BACKEND_WARNING: ClassVar[str] = (
        r"^None of PyTorch, TensorFlow >= 2\.0, or Flax have been found\."
        r" Models won't be available and only tokenizers, configuration and "
        r"file/data utilities can be used\."
    )

    def __init__(self, *args, enable_thinking: bool = True, **kwargs):
        super().__init__(*args, enable_thinking=enable_thinking, **kwargs)

    @classmethod
    def _get_tokenizer(cls):
        tokenizer = cls._tokenizer
        if tokenizer is None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=cls._TRANSFORMERS_BACKEND_WARNING,
                    category=UserWarning,
                )
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    cls.TOKENIZER_NAME,
                    trust_remote_code=True,
                )
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
            raise ValueError("generate_with_prefix only supports n=1")

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
    ) -> PrefixCompletionState:
        reasoning, content, think_done = split_assistant_parts(
            assistant_prompt_prefix
        )
        return PrefixCompletionState(
            assistant_prompt_prefix=build_assistant_prefix(
                reasoning,
                content,
                think_done,
            ),
            reasoning=reasoning,
            content=content,
            think_done=think_done,
            stop_reason=stop_reason,
        )

    def _generate_with_prefix_step(
        self,
        messages: Messages,
        assistant_prompt_prefix: str,
        **runtime_kwargs,
    ) -> tuple[str, str | None]:
        tokenizer = self._get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        prompt_to_continue = prompt + assistant_prompt_prefix

        call_kwargs = {
            k: v
            for k, v in runtime_kwargs.items()
            if k not in {"extra_body", "max_tokens"}
        }
        call_kwargs.setdefault("max_tokens", 1)

        result = self._generate_response(prompt_to_continue, **call_kwargs)
        text = str(result.get("text") or "")
        stop_reason = result.get("stop", result.get("finish_reason"))
        return text, stop_reason

    def _complete_reasoning(
        self,
        messages: Messages,
        assistant_prompt_prefix: str,
        *,
        thinking_max_tokens: int,
        **runtime_kwargs,
    ) -> PrefixCompletionState:
        state = self._build_prefix_state(assistant_prompt_prefix)
        if state.think_done:
            return state._replace(stop_reason=None)

        ret, stop_reason = self._generate_with_prefix_step(
            messages,
            state.assistant_prompt_prefix,
            max_tokens=thinking_max_tokens,
            **runtime_kwargs,
        )

        state = self._build_prefix_state(
            state.assistant_prompt_prefix + strip_assistant_end(ret),
            stop_reason=stop_reason,
        )

        if not state.think_done:
            state = self._build_prefix_state(
                state.assistant_prompt_prefix + "\n</think>\n\n",
                stop_reason=stop_reason,
            )

        return state

    def _complete_content(
        self,
        messages: Messages,
        completion_state: PrefixCompletionState,
        *,
        content_max_tokens: int,
        **runtime_kwargs,
    ) -> "ChatCompletionMessage":
        ret, _ = self._generate_with_prefix_step(
            messages,
            completion_state.assistant_prompt_prefix,
            max_tokens=content_max_tokens,
            **runtime_kwargs,
        )

        state = self._build_prefix_state(
            completion_state.assistant_prompt_prefix + strip_assistant_end(ret)
        )

        return self._build_openai_message(
            reasoning=state.reasoning,
            content=state.content or "",
        )

    def _build_openai_message(
        self,
        content: str,
        reasoning: str | None,
    ) -> "ChatCompletionMessage":
        from openai.types.chat import ChatCompletionMessage

        message_kwargs = {
            "role": "assistant",
            "content": content,
        }
        if reasoning:
            message_kwargs["reasoning_content"] = reasoning
        return ChatCompletionMessage(**message_kwargs)

    def complete_reasoning(
        self,
        input_data: str | BaseModel | list[dict],
        assistant_prompt_prefix: str = "<think>\n",
        *,
        thinking_max_tokens: int | None = None,
        **runtime_kwargs,
    ) -> PrefixCompletionState:
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
        assert thinking_max_tokens is not None

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
        completion_state: PrefixCompletionState | str,
        *,
        content_max_tokens: int | None = None,
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
        assert content_max_tokens is not None

        if isinstance(completion_state, str):
            completion_state = self._build_prefix_state(completion_state)

        messages = self._prepare_input(input_data)
        return self._complete_content(
            messages,
            completion_state,
            content_max_tokens=content_max_tokens,
            **runtime_kwargs,
        )

    def generate_with_prefix(
        self,
        input_data: str | BaseModel | list[dict],
        assistant_prompt_prefix: str = "<think>\n",
        *,
        thinking_max_tokens: int | None = None,
        content_max_tokens: int | None = None,
        **runtime_kwargs,
    ) -> "ChatCompletionMessage":
        """
        Continue generation from a partial assistant prefix.

        This is useful for Qwen3-style reasoning traces where we want to seed
        or continue a `<think>...</think>` block before the final answer.
        """
        self._validate_prefix_completion_kwargs(
            n=runtime_kwargs.get("n", 1),
            thinking_max_tokens=thinking_max_tokens,
            content_max_tokens=content_max_tokens,
            require_thinking_max_tokens=True,
            require_content_max_tokens=True,
        )
        assert thinking_max_tokens is not None
        assert content_max_tokens is not None

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
            return self._build_openai_message(
                reasoning=reasoning_state.reasoning,
                content=reasoning_state.content,
            )

        return self._complete_content(
            messages,
            reasoning_state,
            content_max_tokens=content_max_tokens,
            **runtime_kwargs,
        )
