from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from httpx import Timeout
from loguru import logger
from pydantic import BaseModel

from .._traceback import clean_traceback
from ..chat_format.transform import transform_messages
from .llm import LLM, Messages


if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessage
    from openai.types.completion_choice import CompletionChoice


ASSISTANT_PREFIX = "<|im_start|>assistant"
THINK_START = "<think>"
THINK_END = "</think>"
ASSISTANT_END = "<|im_end|>"
DEFAULT_THINKING_MAX_TOKENS = 8192
DEFAULT_CONTENT_MAX_TOKENS = 4096
DEFAULT_EARLY_THINKING_STOP_MESSAGE = (
    "\n\n[SYSTEM] Thinking budget exhausted. Stopping now.\n\n"
)
TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV = "TRANSFORMERS_NO_ADVISORY_WARNINGS"
_TOKENIZER_CACHE_ROOT = Path("/tmp/tokenizers")
_AUTO_TOKENIZER_CLS = None
_TOKENIZER_LOAD_LOCK = threading.Lock()
_TRANSFORMERS_IMPORT_ERROR_MESSAGE = (
    "transformers is required for Qwen3 tokenizer-backed prompt rendering. "
    "Install with: pip install 'speedy-utils[transformers]'"
)


class _PrefixCompletionState(BaseModel):
    assistant_prompt_prefix: str
    reasoning: str | None = None
    content: str | None = None
    think_done: bool = False
    stop_reason: str | None = None
    call_count: int = 0
    usage: Any | None = None


_PrefixCompletionState.model_rebuild()


class _CustomPrefixCompletionState(BaseModel):
    assistant_prompt_prefix: str = ""
    generated_text: str = ""
    stop: str | None = None
    stop_reason: str | None = None
    call_count: int = 0
    usage: Any | None = None
    client_idx: int | None = None

    def inject(self, text: str) -> "_CustomPrefixCompletionState":
        """Return a new continuation state with extra assistant-body text appended."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return self
        # Only strip wrapper tokens, preserve the rest verbatim
        body = text
        if body.startswith(ASSISTANT_PREFIX):
            body = body[len(ASSISTANT_PREFIX) :]
        if body.endswith(ASSISTANT_END):
            body = body[: -len(ASSISTANT_END)]
        return self.model_copy(
            update={
                "assistant_prompt_prefix": f"{self.assistant_prompt_prefix}{body}",
                "generated_text": f"{self.generated_text}{body}",
                "stop": None,
                "stop_reason": None,
            }
        )


_CustomPrefixCompletionState.model_rebuild()


def _sanitize_assistant_body(text: str) -> str:
    """Remove only explicit assistant wrapper tokens from assistant-body text."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    body = text
    if body.startswith(ASSISTANT_PREFIX):
        body = body[len(ASSISTANT_PREFIX) :]
    if body.endswith(ASSISTANT_END):
        body = body[: -len(ASSISTANT_END)]
    return body


def _merge_generated_with_stop(
    generated_text: str,
    *,
    stop_reason: str | None,
    stop_sequences: list[str],
    include_stop: bool,
) -> tuple[str, str | None]:
    """Merge generated output with stop-token handling, return (text, matched_stop)."""
    matched_stop = None
    if stop_reason == "stop" and stop_sequences:
        matched_stop = stop_sequences[0] if len(stop_sequences) == 1 else None
        if (
            matched_stop
            and include_stop
            and matched_stop not in generated_text
            and matched_stop != ASSISTANT_END
        ):
            generated_text = f"{generated_text}{matched_stop}"
    return generated_text, matched_stop


def split_assistant_parts(text: str) -> tuple[str | None, str | None, bool]:
    """Split an assistant prefix into reasoning, content, and think completion."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    body = text
    if body.startswith(ASSISTANT_PREFIX):
        body = body[len(ASSISTANT_PREFIX) :]
    body = body.lstrip()

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
        return f"{THINK_START}\n{reasoning_text}\n{THINK_END}{content_text}"

    return f"{THINK_START}\n{reasoning_text}"


def is_content_done(content: str | None, stop_reason: str | None) -> bool:
    """Return True when the model produced content and stopped normally."""
    return content not in ("", None) and stop_reason == "stop"


def strip_assistant_end(text: str) -> str:
    """Remove the assistant end token when the backend appends it (legacy helper)."""
    stripped_text = text.rstrip()
    if stripped_text.endswith(ASSISTANT_END):
        return stripped_text[: -len(ASSISTANT_END)].rstrip()
    return text


def _add_assistant_wrapper(body: str) -> str:
    """Wrap assistant-body text with the assistant turn marker for the actual prompt."""
    if not isinstance(body, str):
        raise TypeError("body must be a string")
    if body.startswith(ASSISTANT_PREFIX):
        return body
    if body.startswith("\n"):
        return f"{ASSISTANT_PREFIX}{body}"
    return f"{ASSISTANT_PREFIX}\n{body}"


def _load_auto_tokenizer():
    """Import AutoTokenizer lazily so tests can monkeypatch it cheaply."""
    global _AUTO_TOKENIZER_CLS
    if _AUTO_TOKENIZER_CLS is None:
        try:
            from transformers import AutoTokenizer as tokenizer_cls
        except ImportError as exc:
            raise ImportError(_TRANSFORMERS_IMPORT_ERROR_MESSAGE) from exc
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
    except ImportError:
        raise
    except Exception as exc:
        logger.error(f"Error loading tokenizer for {name_or_path}: {exc}")
        raise
    finally:
        if previous is None:
            os.environ.pop(TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV, None)
        else:
            os.environ[TRANSFORMERS_NO_ADVISORY_WARNINGS_ENV] = previous


def _render_chatml_prompt(messages: Messages) -> str:
    """Render ChatML text without requiring a Hugging Face tokenizer."""
    prompt = transform_messages(
        messages,
        frm="chatml",
        to="text",
        add_generation_prompt=False,
    )
    if not isinstance(prompt, str):
        raise TypeError("Qwen3 chat prompt must serialize to a string")
    return prompt


class Qwen3LLM(LLM):
    """Qwen3 helper for staged generation with a partial assistant prefix."""

    TOKENIZER_NAME: ClassVar[str] = "Qwen/Qwen3-0.6B"
    _tokenizer: ClassVar[Any | None] = None
    _tokenizer_checked: ClassVar[bool] = False
    _tokenizer_import_error: ClassVar[str | None] = None

    def __init__(
        self,
        client: Any = None,
        cache: bool = True,
        verbose: bool = False,
        timeout: float | Timeout | None = None,
        *,
        enable_thinking: bool = True,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | tuple[str, ...] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        thinking_max_tokens: int = DEFAULT_THINKING_MAX_TOKENS,
        content_max_tokens: int = DEFAULT_CONTENT_MAX_TOKENS,
        **model_kwargs: Any,
    ) -> None:
        self._validate_prefix_completion_kwargs(
            n=1,
            thinking_max_tokens=thinking_max_tokens,
            content_max_tokens=content_max_tokens,
            require_thinking_max_tokens=True,
            require_content_max_tokens=True,
        )
        self.default_thinking_max_tokens = thinking_max_tokens
        self.default_content_max_tokens = content_max_tokens
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
    def _get_tokenizer(cls):
        tokenizer = cls._tokenizer
        if tokenizer is None:
            with _TOKENIZER_LOAD_LOCK:
                tokenizer = cls._tokenizer
                if tokenizer is not None:
                    return tokenizer
                if cls._tokenizer_checked:
                    raise ImportError(
                        cls._tokenizer_import_error
                        or _TRANSFORMERS_IMPORT_ERROR_MESSAGE
                    )
                cls._tokenizer_checked = True
                try:
                    tokenizer = _get_tokenizer(cls.TOKENIZER_NAME)
                except ImportError as exc:
                    cls._tokenizer_import_error = str(exc)
                    raise
                cls._tokenizer = tokenizer
                cls._tokenizer_import_error = None
        return tokenizer

    def _build_completion_prompt(self, messages: Messages) -> str:
        try:
            tokenizer = self._get_tokenizer()
        except ImportError:
            return _render_chatml_prompt(messages)

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(prompt, str):
            raise TypeError("Qwen3 tokenizer prompt must be a string")
        return prompt

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
            return self._build_prefix_state(
                assistant_prompt_prefix
            ).assistant_prompt_prefix
        return build_assistant_prefix("", None, True)

    @staticmethod
    def _normalize_raw_assistant_prefix(assistant_prompt_prefix: str) -> str:
        """Strip ASSISTANT_PREFIX and ASSISTANT_END wrappers, returning assistant-body text."""
        return _sanitize_assistant_body(assistant_prompt_prefix)

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
            if include_stop and matched_stop and matched_stop not in generated_text:
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
        assistant_body: str,
        *,
        client_idx: int | None = None,
        return_client_idx: bool = False,
        **runtime_kwargs,
    ) -> "CompletionChoice | tuple[CompletionChoice, int]":
        if client_idx is not None:
            self._get_tracked_client(client_idx)

        prompt = self._build_completion_prompt(messages)
        call_kwargs = dict(runtime_kwargs)
        call_kwargs.pop("enable_thinking", None)
        prompt_to_continue = prompt + _add_assistant_wrapper(assistant_body)

        cache = call_kwargs.pop("cache", None)
        call_kwargs.pop("extra_body", None)
        generation_result = self._raw_completion_step(
            prompt_to_continue,
            cache=cache,
            client_idx=client_idx,
            return_client_idx=return_client_idx,
            drop_keys=("extra_body",),
            **call_kwargs,
        )
        if return_client_idx:
            choice, resolved_client_idx = cast(tuple[Any, int], generation_result)
        else:
            choice = generation_result
            resolved_client_idx = None

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

        output_text = _sanitize_assistant_body(str(choice.text or ""))
        state = self._build_prefix_state(
            assistant_body + output_text,
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
            if resolved_client_idx is None:
                raise RuntimeError("Expected tracked client index for prefix step.")
            return cast("CompletionChoice", choice), resolved_client_idx
        return cast("CompletionChoice", choice)

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
        max_tokens: int | None = None,
        include_stop_in_prefix: bool = True,
        **runtime_kwargs,
    ) -> _CustomPrefixCompletionState:
        """
        Run one raw prefix-conditioned completion until a stop sequence.

        This is the custom path for staged generation flows such as
        `<memory>...</memory>` followed by `<think_efficient>...</think_efficient>`.
        """
        if max_tokens is None:
            max_tokens = cast(
                int,
                self.model_kwargs.get("max_tokens", self.default_content_max_tokens),
            )
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
        assistant_body = self._normalize_raw_assistant_prefix(prior_prefix)
        generation_result = self._generate_with_prefix_step(
            messages,
            assistant_body,
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
        raw_text = str(choice.text or "")
        generated_text = _sanitize_assistant_body(raw_text)
        generated_text, matched_stop = _merge_generated_with_stop(
            generated_text,
            stop_reason=choice.finish_reason,
            stop_sequences=stop_sequences,
            include_stop=include_stop_in_prefix,
        )
        usage = getattr(choice, "usage", None)
        return _CustomPrefixCompletionState(
            assistant_prompt_prefix=assistant_body + generated_text,
            generated_text=generated_text,
            stop=matched_stop,
            stop_reason=choice.finish_reason,
            call_count=prior_call_count + 1,
            usage=usage,
            client_idx=resolved_client_idx,
        )

    def complete_reasoning(
        self,
        messages: str | Messages,
        assistant_prompt_prefix: str,
        *,
        thinking_max_tokens: int | None = None,
        early_thinking_stop_message: str | None | bool = None,
        **runtime_kwargs,
    ) -> _PrefixCompletionState:
        resolved_thinking_max_tokens = (
            thinking_max_tokens
            if thinking_max_tokens is not None
            else self.default_thinking_max_tokens
        )
        enable_thinking = runtime_kwargs.get("enable_thinking")
        state = self._build_prefix_state(
            self._normalize_assistant_prefix(
                assistant_prompt_prefix,
                enable_thinking=enable_thinking,
            )
        )
        if state.think_done:
            return state.model_copy(update={"stop_reason": None})

        custom_state = self.complete_until(
            messages,
            state.assistant_prompt_prefix,
            stop=THINK_END,
            max_tokens=resolved_thinking_max_tokens,
            **runtime_kwargs,
        )
        state = self._custom_state_to_prefix_state(custom_state)

        if not state.think_done:
            close_text = (
                f"{early_thinking_stop_message}\n{THINK_END}\n\n"
                if early_thinking_stop_message
                else f"\n{THINK_END}\n\n"
            )
            state = self._build_prefix_state(
                state.assistant_prompt_prefix + close_text,
                stop_reason=state.stop_reason,
                call_count=state.call_count,
                usage=state.usage,
            )

        return state

    def complete_content(
        self,
        messages: str | Messages,
        completion_state: str | _PrefixCompletionState,
        *,
        content_max_tokens: int | None = None,
        **runtime_kwargs,
    ) -> "ChatCompletionMessage":
        if isinstance(messages, str):
            messages = self._prepare_input(messages)
        if isinstance(completion_state, str):
            completion_state = self._build_prefix_state(completion_state)
        resolved_content_max_tokens = (
            content_max_tokens
            if content_max_tokens is not None
            else self.default_content_max_tokens
        )
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
            max_tokens=resolved_content_max_tokens,
            **runtime_kwargs,
        )

        output_text = _sanitize_assistant_body(str(choice.text or ""))
        state = self._build_prefix_state(
            completion_state.assistant_prompt_prefix + output_text,
            usage=getattr(choice, "usage", None),
        )
        message = self._build_openai_message(
            reasoning=state.reasoning,
            content=state.content or "",
            usage=state.usage,
            call_count=completion_state.call_count + 1,
        )
        return message

    def _build_openai_message(
        self,
        content: str,
        reasoning: str | None,
        usage: Any | None = None,
        call_count: int | None = None,
    ) -> "ChatCompletionMessage":
        from openai.types.chat import ChatCompletionMessage

        # Pydantic v2 lazily builds serializers via MockValSer placeholders.
        # When `usage` is used as a pydantic extra field inside
        # ChatCompletionMessage, the outer serializer requires usage's class
        # serializer to already be a real SchemaSerializer.  Calling
        # model_rebuild() forces that serializer to be built now, so
        # model_dump() on the returned message works correctly.
        if usage is not None and hasattr(usage, "model_rebuild"):
            type(usage).model_rebuild()

        extra: dict[str, Any] = {}
        if reasoning:
            extra["reasoning"] = reasoning
        if usage is not None:
            extra["usage"] = usage
        if call_count is not None:
            extra["call_count"] = call_count
        return ChatCompletionMessage(role="assistant", content=content, **extra)

    @clean_traceback
    def chat_completion(
        self,
        input_data: str | BaseModel | list[dict],
        assistant_prompt_prefix: str = "<think>\n",
        *,
        thinking_max_tokens: int | None = None,
        content_max_tokens: int | None = None,
        max_tokens: int | None = None,
        early_thinking_stop_message: str | None | bool = None,
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
            thinking_max_tokens=thinking_max_tokens or self.default_thinking_max_tokens,
            content_max_tokens=content_max_tokens or self.default_content_max_tokens,
            require_thinking_max_tokens=True,
            require_content_max_tokens=True,
        )

        messages = self._prepare_input(input_data)
        effective_enable_thinking = self._resolve_enable_thinking(
            runtime_kwargs.get("enable_thinking")
        )
        if effective_enable_thinking:
            resolved_thinking_max_tokens = (
                thinking_max_tokens
                if thinking_max_tokens is not None
                else self.default_thinking_max_tokens
            )
            resolved_content_max_tokens = (
                content_max_tokens
                if content_max_tokens is not None
                else self.default_content_max_tokens
            )
            reasoning_state = self.complete_reasoning(
                messages,
                assistant_prompt_prefix,
                thinking_max_tokens=resolved_thinking_max_tokens,
                early_thinking_stop_message=early_thinking_stop_message,
                **runtime_kwargs,
            )
            if is_content_done(
                reasoning_state.content,
                reasoning_state.stop_reason,
            ):
                message = self._build_openai_message(
                    reasoning=reasoning_state.reasoning,
                    content=reasoning_state.content or "",
                    usage=reasoning_state.usage,
                    call_count=reasoning_state.call_count,
                )
                return message

            return self.complete_content(
                messages,
                reasoning_state,
                content_max_tokens=resolved_content_max_tokens,
                **runtime_kwargs,
            )

        # Single-step path when enable_thinking=False.
        normalized_prefix = self._normalize_assistant_prefix(
            assistant_prompt_prefix,
            enable_thinking=False,
        )
        custom_state = self.complete_until(
            messages,
            normalized_prefix,
            stop=ASSISTANT_END,
            max_tokens=max_tokens,
            include_stop_in_prefix=False,
            **runtime_kwargs,
        )
        # parse to assistant chat message
        text = custom_state.assistant_prompt_prefix
        reasoning, content, _ = split_assistant_parts(text)
        chat_msg = self._build_openai_message(
            reasoning=reasoning,
            content=content or "",
            usage=custom_state.usage,
            call_count=1,
        )
        return chat_msg
