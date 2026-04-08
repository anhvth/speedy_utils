from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
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
DEFAULT_CONTENT_MAX_TOKENS = 4096
TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV = "TRANSFORMERS_NO_ADVISORY_WARNINGS"
_TOKENIZER_CACHE_ROOT = Path("/tmp/tokenizers")
_AUTO_TOKENIZER_CLS = None


class _PrefixCompletionState(NamedTuple):
    assistant_prompt_prefix: str
    reasoning: str | None
    content: str | None
    think_done: bool
    stop_reason: str | None
    call_count: int
    usage: Any | None


@dataclass(frozen=True)
class _CustomPrefixCompletionState:
    assistant_prompt_prefix: str
    generated_text: str
    stop: str | None
    stop_reason: str | None
    call_count: int
    usage: Any | None
    client_idx: int | None = None

    def inject(self, text: str) -> "_CustomPrefixCompletionState":
        """Return a new continuation state with extra raw prefix text appended."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return self
        return replace(
            self,
            assistant_prompt_prefix=f"{self.assistant_prompt_prefix}{text}",
            generated_text=f"{self.generated_text}{text}",
            stop=None,
            stop_reason=None,
        )


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


def _load_auto_tokenizer():
    """Import AutoTokenizer lazily so tests can monkeypatch it cheaply."""
    global _AUTO_TOKENIZER_CLS
    if _AUTO_TOKENIZER_CLS is None:
        try:
            from transformers import AutoTokenizer as tokenizer_cls
        except ImportError:
            from transformers.models.auto.tokenization_auto import (
                AutoTokenizer as tokenizer_cls,
            )
            print("Warning: Imported AutoTokenizer from transformers.models.auto.tokenization_auto, which may cause issues if transformers is updated. Please ensure transformers is up to date to avoid this warning.")
        _AUTO_TOKENIZER_CLS = tokenizer_cls
    return _AUTO_TOKENIZER_CLS


def _get_tokenizer(name_or_path: str | os.PathLike[str]):
    """Load and cache a tokenizer from a local path or remote model name."""
    # Transformers emits an import-time advisory when no backend framework is
    # installed; suppress it only while loading tokenizer helpers.
    previous = os.environ.get(TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV)
    os.environ[TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV] = "1"
    try:
        AutoTokenizer = _load_auto_tokenizer()

        source = Path(name_or_path).expanduser()
        if source.is_dir() or source.is_file():
            local_tokenizer = source
        else:
            cache_name = str(name_or_path).replace("/", "_")
            local_tokenizer = _TOKENIZER_CACHE_ROOT / cache_name
            local_tokenizer.parent.mkdir(parents=True, exist_ok=True)
            lock_path = local_tokenizer.with_suffix(local_tokenizer.suffix + ".lock")
            import fcntl

            with open(lock_path, "w") as lock_fd:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
                if not local_tokenizer.exists():
                    tokenizer = AutoTokenizer.from_pretrained(
                        str(name_or_path),
                        trust_remote_code=True,
                    )
                    tokenizer.save_pretrained(str(local_tokenizer))

        tokenizer = AutoTokenizer.from_pretrained(
            str(local_tokenizer),
            trust_remote_code=True,
        )
        return tokenizer
    except Exception as exc:
        logger.error(f"Error loading tokenizer for {name_or_path}: {exc}")
        raise
    finally:
        
        if previous is None:
            os.environ.pop(TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV, None)
        else:
            os.environ[TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV] = previous


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
            tokenizer = _get_tokenizer(cls.TOKENIZER_NAME)
            cls._tokenizer = tokenizer
        return tokenizer

    def _resolve_enable_thinking(self, enable_thinking: bool | None = None) -> bool:
        effective_enable_thinking = (
            self.enable_thinking if enable_thinking is None else enable_thinking
        )
        return effective_enable_thinking is not False

    def _normalize_assistant_prefix(
        self,
        assistant_prompt_prefix: str,
        *,
        enable_thinking: bool | None = None,
    ) -> str:
        if self._resolve_enable_thinking(enable_thinking):
            return self._build_prefix_state(assistant_prompt_prefix).assistant_prompt_prefix
        return build_assistant_prefix("", None, True)

    @staticmethod
    def _normalize_raw_assistant_prefix(assistant_prompt_prefix: str) -> str:
        if not isinstance(assistant_prompt_prefix, str):
            raise TypeError("assistant_prompt_prefix must be a string")
        if assistant_prompt_prefix.startswith(ASSISTANT_PREFIX):
            return assistant_prompt_prefix
        if assistant_prompt_prefix.startswith("\n"):
            return f"{ASSISTANT_PREFIX}{assistant_prompt_prefix}"
        return f"{ASSISTANT_PREFIX}\n{assistant_prompt_prefix}"

    @staticmethod
    def _normalize_stop_sequences(stop: str | list[str] | tuple[str, ...]) -> list[str]:
        if isinstance(stop, str):
            if not stop:
                raise ValueError("stop must not be empty")
            return [stop]
        stop_list = list(stop)
        if not stop_list or any(not item for item in stop_list):
            raise ValueError("stop must contain non-empty strings")
        return stop_list

    @staticmethod
    def _append_stop_text(
        generated_text: str,
        *,
        stop_reason: str | None,
        stop_sequences: list[str],
        include_stop: bool,
    ) -> tuple[str, str | None]:
        matched_stop = None
        if stop_reason == "stop" and stop_sequences:
            matched_stop = stop_sequences[0] if len(stop_sequences) == 1 else None
            if (
                include_stop
                and matched_stop
                and matched_stop not in generated_text
            ):
                generated_text = f"{generated_text}{matched_stop}"
        return generated_text, matched_stop

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
        *,
        prefix_mode: str = "think",
        client_idx: int | None = None,
        return_client_idx: bool = False,
        **runtime_kwargs,
    ) -> "CompletionChoice | tuple[CompletionChoice, int]":
        tokenizer = self._get_tokenizer()
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        call_kwargs = dict(runtime_kwargs)
        enable_thinking = call_kwargs.pop("enable_thinking", None)
        if prefix_mode == "think":
            assistant_prompt_prefix = self._normalize_assistant_prefix(
                assistant_prompt_prefix,
                enable_thinking=enable_thinking,
            )
        elif prefix_mode == "raw":
            assistant_prompt_prefix = self._normalize_raw_assistant_prefix(
                assistant_prompt_prefix
            )
        else:
            raise ValueError(f"Unsupported prefix_mode: {prefix_mode}")
        prompt_to_continue = prompt + assistant_prompt_prefix

        cache = call_kwargs.pop("cache", None)
        self._require_single_choice(call_kwargs)
        if call_kwargs.get("max_tokens") is None:
            call_kwargs["max_tokens"] = 1

        effective_kwargs = {**self.model_kwargs, **call_kwargs}
        model_name, api_kwargs = self._build_api_kwargs(
            effective_kwargs,
            drop_keys=("extra_body",),
        )
        api_kwargs.pop("extra_body", None)

        borrow_client = (
            self._borrow_client()
            if client_idx is None
            else self._borrow_client_by_index(client_idx)
        )
        with borrow_client as client:
            self._set_cache(cache, client=client)
            try:
                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt_to_continue,
                    **api_kwargs,
                )
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
        resolved_client_idx = self._client_index_by_id[id(client)]

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
        if return_client_idx:
            return choice, resolved_client_idx
        return choice

    @staticmethod
    def _resolve_custom_prefix_state(
        assistant_prompt_prefix: str | _CustomPrefixCompletionState,
    ) -> tuple[str, int | None, int]:
        if isinstance(assistant_prompt_prefix, _CustomPrefixCompletionState):
            return (
                assistant_prompt_prefix.assistant_prompt_prefix,
                assistant_prompt_prefix.client_idx,
                assistant_prompt_prefix.call_count,
            )
        return assistant_prompt_prefix, None, 0

    @classmethod
    def _custom_state_to_prefix_state(
        cls,
        completion_state: _CustomPrefixCompletionState,
    ) -> _PrefixCompletionState:
        return cls._build_prefix_state(
            completion_state.assistant_prompt_prefix,
            stop_reason=completion_state.stop_reason,
            call_count=completion_state.call_count,
            usage=completion_state.usage,
        )

    def complete_until(
        self,
        input_data: str | BaseModel | list[dict],
        assistant_prompt_prefix: str | _CustomPrefixCompletionState,
        *,
        stop: str | list[str] | tuple[str, ...],
        max_tokens: int = DEFAULT_CONTENT_MAX_TOKENS,
        include_stop_in_prefix: bool = True,
        **runtime_kwargs,
    ) -> _CustomPrefixCompletionState:
        """
        Run one raw prefix-conditioned completion until a stop sequence.

        This is the custom path for staged generation flows such as
        `<memory>...</memory>` followed by `<think_efficient>...</think_efficient>`.
        """
        self._validate_prefix_completion_kwargs(
            n=runtime_kwargs.get("n", 1),
            content_max_tokens=max_tokens,
            require_content_max_tokens=True,
        )
        stop_sequences = self._normalize_stop_sequences(stop)
        messages = self._prepare_input(input_data)
        prior_prefix, client_idx, prior_call_count = self._resolve_custom_prefix_state(
            assistant_prompt_prefix
        )
        normalized_prefix = self._normalize_raw_assistant_prefix(prior_prefix)
        generation_result = self._generate_with_prefix_step(
            messages,
            normalized_prefix,
            prefix_mode="raw",
            client_idx=client_idx,
            return_client_idx=True,
            max_tokens=max_tokens,
            stop=stop_sequences,
            **runtime_kwargs,
        )
        if isinstance(generation_result, tuple):
            choice, resolved_client_idx = generation_result
        else:
            choice = generation_result
            resolved_client_idx = client_idx
        generated_text = strip_assistant_end(str(choice.text or ""))
        generated_text, matched_stop = self._append_stop_text(
            generated_text,
            stop_reason=choice.finish_reason,
            stop_sequences=stop_sequences,
            include_stop=include_stop_in_prefix,
        )
        usage = getattr(choice, "usage", None)
        return _CustomPrefixCompletionState(
            assistant_prompt_prefix=normalized_prefix + generated_text,
            generated_text=generated_text,
            stop=matched_stop,
            stop_reason=choice.finish_reason,
            call_count=prior_call_count + 1,
            usage=usage,
            client_idx=resolved_client_idx,
        )

    def _complete_reasoning(
        self,
        messages: Messages,
        assistant_prompt_prefix: str,
        *,
        thinking_max_tokens: int,
        **runtime_kwargs,
    ) -> _PrefixCompletionState:
        enable_thinking = runtime_kwargs.get("enable_thinking")
        state = self._build_prefix_state(
            self._normalize_assistant_prefix(
                assistant_prompt_prefix,
                enable_thinking=enable_thinking,
            )
        )
        if state.think_done:
            return state._replace(stop_reason=None)

        custom_state = self.complete_until(
            messages,
            state.assistant_prompt_prefix,
            stop=THINK_END,
            max_tokens=thinking_max_tokens,
            **runtime_kwargs,
        )
        state = self._custom_state_to_prefix_state(custom_state)

        if not state.think_done:
            state = self._build_prefix_state(
                state.assistant_prompt_prefix + "\n</think>\n\n",
                stop_reason=state.stop_reason,
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
        enable_thinking = runtime_kwargs.get("enable_thinking")
        completion_state = self._build_prefix_state(
            self._normalize_assistant_prefix(
                completion_state.assistant_prompt_prefix,
                enable_thinking=enable_thinking,
            ),
            stop_reason=completion_state.stop_reason,
            call_count=completion_state.call_count,
            usage=completion_state.usage,
        )
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
            completion_state = self._build_prefix_state(
                self._normalize_assistant_prefix(
                    completion_state,
                    enable_thinking=runtime_kwargs.get("enable_thinking"),
                )
            )

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
        Continue a chat-based assistant turn from a partial assistant prefix.

        This is the primary Qwen3 chat-completions entry point. It seeds or
        continues a `<think>...</think>` block inside the assistant turn before
        the final answer, and is distinct from the raw prompt `generate()` API.
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


    def _assistant_message_to_content(self, message: "ChatCompletionMessage") -> str:
        # format thinking and content into a single assistant message text for history recording
        reasoning = getattr(message, "reasoning_content", "")
        content = message.content
        return f"{THINK_START}\n{reasoning}\n{THINK_END}\n\n{content}"
