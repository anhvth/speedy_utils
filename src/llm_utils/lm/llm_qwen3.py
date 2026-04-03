from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from .llm import LLM, Messages


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessage


ASSISTANT_PREFIX = "<|im_start|>assistant"
THINK_START = "<think>"
THINK_END = "</think>"
ASSISTANT_END = "<|im_end|>"


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


class LLM_Qwen3_Reasoning(LLM):
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
            k: v for k, v in runtime_kwargs.items() if k != "extra_body"
        }
        call_kwargs.setdefault("max_tokens", 1)

        result = self.generate(prompt_to_continue, **call_kwargs)
        text = str(result.get("text") or "")
        stop_reason = result.get("stop", result.get("finish_reason"))
        return text, stop_reason

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
        if runtime_kwargs.get("n", 1) != 1:
            raise ValueError("generate_with_prefix only supports n=1")
        if thinking_max_tokens is None:
            raise ValueError("thinking_max_tokens must not be None")
        if content_max_tokens is None:
            raise ValueError("content_max_tokens must not be None")
        if thinking_max_tokens <= 0:
            raise ValueError("thinking_max_tokens must be > 0")
        if content_max_tokens <= 0:
            raise ValueError("content_max_tokens must be > 0")

        messages = self._prepare_input(input_data)
        reasoning, content, think_done = split_assistant_parts(
            assistant_prompt_prefix
        )
        assistant_prompt_prefix = build_assistant_prefix(
            reasoning,
            content,
            think_done,
        )

        thinking_stop_reason: str | None = None

        if not think_done:
            ret, thinking_stop_reason = self._generate_with_prefix_step(
                messages,
                assistant_prompt_prefix,
                max_tokens=thinking_max_tokens,
                **runtime_kwargs,
            )

            ret = strip_assistant_end(ret)
            assistant_prompt_prefix += ret
            reasoning, content, think_done = split_assistant_parts(
                assistant_prompt_prefix
            )

            if not think_done:
                # Force the model out of the reasoning phase once the
                # dedicated thinking budget is exhausted.
                assistant_prompt_prefix += "\n</think>\n\n"
                reasoning, content, think_done = split_assistant_parts(
                    assistant_prompt_prefix
                )

            if is_content_done(content, thinking_stop_reason):
                return self._build_openai_message(
                    reasoning=reasoning,
                    content=content,
                )

        ret, _ = self._generate_with_prefix_step(
            messages,
            assistant_prompt_prefix,
            max_tokens=content_max_tokens,
            **runtime_kwargs,
        )

        ret = strip_assistant_end(ret)
        assistant_prompt_prefix += ret
        reasoning, content, think_done = split_assistant_parts(
            assistant_prompt_prefix
        )

        return self._build_openai_message(
            reasoning=reasoning,
            content=content or "",
        )

    def generate_with_think_prefix(
        self,
        input_data: str | BaseModel | list[dict],
        **runtime_kwargs,
    ) -> "ChatCompletionMessage":
        """Backward-compatible alias for prefilling a `<think>` block."""
        runtime_kwargs.setdefault("assistant_prompt_prefix", "<think>\n")
        return self.generate_with_prefix(input_data, **runtime_kwargs)


LLM_Qwen3 = LLM_Qwen3_Reasoning
