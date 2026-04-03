# type: ignore

"""Runtime LLM wrapper for OpenAI-compatible chat and text completions."""

# Typing imports
import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from httpx import Timeout
from loguru import logger
from pydantic import BaseModel

from speedy_utils import clean_traceback

from .utils import (
    get_base_client,
)


# Lazy import openai types for type checking only
if TYPE_CHECKING:
    from openai import (
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        OpenAI,
        RateLimitError,
    )
    from openai.types.chat import ChatCompletionMessageParam

# Type aliases for better readability
Messages = list[dict]  # Simplified type, actual type validated at runtime


class LLM:
    """LLM task with runtime response-model selection."""

    _LEGACY_CONSTRUCTOR_KEYS = {
        "input_model",
        "output_model",
        "response_model",
        "is_reasoning_model",
    }

    def __init__(
        self,
        client: "OpenAI | int | str | None" = None,  # type: ignore[name-defined]
        cache=True,
        verbose=False,
        timeout: float | Timeout | None = None,
        enable_thinking: bool | None = None,
        **model_kwargs,
    ):
        """Initialize LLMTask."""
        legacy_keys = [
            key for key in self._LEGACY_CONSTRUCTOR_KEYS if key in model_kwargs
        ]
        if legacy_keys:
            legacy_keys = sorted(legacy_keys)
            raise TypeError(
                "LLM no longer accepts legacy constructor arguments: "
                + ", ".join(legacy_keys)
                + ". Pass response_model at call time or use LLMSignature for "
                "signature-backed defaults."
            )
        self.model_kwargs = model_kwargs
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.last_ai_response = None  # Store raw response from client
        self.cache = cache
        # Avoid importing OpenAI client class at module import time.
        # If a client object provides an api_key attribute, use it.
        self.api_key = "abc"
        if client is not None:
            api_key = getattr(client, "api_key", None)
            if isinstance(api_key, str) and api_key:
                self.api_key = api_key

        self.client = get_base_client(
            client,
            cache=cache,
            api_key=self.api_key,
        )
        # check connection of client
        try:
            self.client.models.list()
        except Exception as e:
            logger.error(
                f"Failed to connect to OpenAI client: {str(e)}, base_url={self.client.base_url}"
            )
            raise e

        if verbose:
            available_models = [model.id for model in self.client.models.list().data]
            logger.info(f"Available models: {available_models}")

        if not self.model_kwargs.get("model", ""):
            self.model_kwargs["model"] = self.client.models.list().data[0].id

    @property
    def model(self) -> str:
        """Return the model name from model_kwargs."""
        model = self.model_kwargs.get("model")
        if not model:
            logger.warning("No model specified in model_kwargs")
        return model

    def _prepare_input(self, input_data: str | BaseModel | list[dict]) -> Messages:
        """Convert input to messages format."""
        if isinstance(input_data, list):
            assert isinstance(input_data[0], dict) and "role" in input_data[0], (
                "If input_data is a list, it must be a list of messages with 'role' and 'content' keys."
            )
            return cast(Messages, input_data)
        # Convert input to string format
        if isinstance(input_data, str):
            user_content = input_data
        elif hasattr(input_data, "model_dump_json"):
            user_content = input_data.model_dump_json()
        elif isinstance(input_data, dict):
            user_content = json.dumps(input_data, ensure_ascii=False, indent=2)
        else:
            user_content = str(input_data)

        return cast(Messages, [{"role": "user", "content": user_content}])

    def _build_api_kwargs(
        self,
        effective_kwargs: dict[str, Any],
        *,
        enable_thinking: bool | None = None,
        drop_keys: tuple[str, ...] = (),
    ) -> tuple[str, dict[str, Any]]:
        """Normalize API kwargs shared by sync completion paths."""
        model_name = effective_kwargs.get("model", self.model_kwargs["model"])
        filtered_drop_keys = {"model", "enable_thinking", *drop_keys}
        api_kwargs = {
            k: v for k, v in effective_kwargs.items() if k not in filtered_drop_keys
        }

        effective_enable_thinking = (
            self.enable_thinking if enable_thinking is None else enable_thinking
        )
        if effective_enable_thinking is not None:
            extra_body = deepcopy(api_kwargs.get("extra_body") or {})
            chat_template_kwargs = deepcopy(
                extra_body.get("chat_template_kwargs") or {}
            )
            chat_template_kwargs.setdefault(
                "enable_thinking", effective_enable_thinking
            )
            extra_body["chat_template_kwargs"] = chat_template_kwargs
            api_kwargs["extra_body"] = extra_body

        if "timeout" not in api_kwargs and self.timeout is not None:
            api_kwargs["timeout"] = self.timeout

        return model_name, api_kwargs

    @staticmethod
    def _extract_reasoning_content(message: Any) -> str | None:
        """Extract reasoning content from a response message when present."""
        for attr_name in ("reasoning_content", "reasoning"):
            reasoning = getattr(message, attr_name, None)
            if isinstance(reasoning, str):
                return reasoning
        return None

    @staticmethod
    def _get_completion_choices(completion: Any) -> Any:
        """Return the choices collection from a completion-like object."""
        choices = getattr(completion, "choices", None)
        if choices is None and isinstance(completion, dict):
            choices = completion.get("choices")
        return choices

    @staticmethod
    def _get_choice_message(choice: Any) -> Any:
        """Return the assistant message from a choice-like object."""
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        return message

    def _set_cache(self, cache: bool | None) -> None:
        """Update client-side caching when supported."""
        if cache is None:
            return
        if hasattr(self.client, "set_cache"):
            self.client.set_cache(cache)
            return
        logger.warning("Client does not support caching.")

    @staticmethod
    def _require_single_choice(runtime_kwargs: dict[str, Any]) -> None:
        """Reject multi-choice requests in the simplified public API."""
        n = runtime_kwargs.get("n", 1)
        if n != 1:
            raise ValueError("LLM only supports n=1")

    def _record_history(self, choice_messages: Messages) -> None:
        """Track recent message history for debugging helpers."""
        if not hasattr(self, "_last_conversations"):
            self._last_conversations = []
        else:
            self._last_conversations = self._last_conversations[-100:]
        self._last_conversations.append(choice_messages)

    @clean_traceback
    def _text_completion(
        self,
        input_data: str | BaseModel | list[dict],
        *,
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        **runtime_kwargs,
    ) -> list[dict[str, Any]]:
        """Execute a text completion and return normalized internal results."""
        self._set_cache(cache)
        self._require_single_choice(runtime_kwargs)
        # Prepare messages
        messages = self._prepare_input(input_data)

        # Merge runtime kwargs with default model kwargs (runtime takes precedence)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name, api_kwargs = self._build_api_kwargs(
            effective_kwargs,
            enable_thinking=enable_thinking,
        )

        try:
            completion = self.client.chat.completions.create(
                model=model_name, messages=messages, **api_kwargs
            )
            # Store raw response from client
            self.last_ai_response = completion
        except Exception as exc:
            # Import openai exceptions for type checking
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
        # print(completion)

        choices = getattr(completion, "choices", None)
        if not choices:
            raise ValueError("No choices returned from completion.")

        choice = choices[0]
        assistant_message = [{"role": "assistant", "content": choice.message.content}]
        reasoning_content = self._extract_reasoning_content(choice.message)
        if reasoning_content:
            assistant_message[0]["reasoning_content"] = reasoning_content

        choice_messages = cast(Messages, messages + assistant_message)
        result = {
            "parsed": choice.message.content,
            "messages": choice_messages,
        }
        if reasoning_content:
            result["reasoning_content"] = reasoning_content

        self._record_history(choice_messages)
        return [result]

    @clean_traceback
    def chat_completion(
        self,
        input_data: str | BaseModel | list[dict],
        *,
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        **runtime_kwargs,
    ) -> Any:
        """Execute a chat completion and return the first assistant message."""
        self._text_completion(
            input_data,
            cache=cache,
            enable_thinking=enable_thinking,
            **runtime_kwargs,
        )
        completion = self.last_ai_response
        if completion is None:
            raise ValueError("No completion returned from API.")
        completion_choices = self._get_completion_choices(completion)
        if not completion_choices:
            raise ValueError("No choices returned from completion.")
        message = self._get_choice_message(completion_choices[0])
        if message is None:
            raise ValueError("No message returned from completion.")
        return message

    def _generate_response(
        self,
        prompt: str,
        *,
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        **runtime_kwargs,
    ) -> dict[str, Any]:
        """Return normalized generation metadata for raw prompt usage."""
        if not isinstance(prompt, str):
            raise TypeError("generate expects `prompt` to be a string")

        self._set_cache(cache)
        self._require_single_choice(runtime_kwargs)

        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name, api_kwargs = self._build_api_kwargs(
            effective_kwargs,
            enable_thinking=enable_thinking,
        )

        try:
            completion = self.client.completions.create(
                model=model_name,
                prompt=prompt,
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

        choices = getattr(completion, "choices", None)
        if not choices:
            raise ValueError("No choices returned from completion.")

        choice = choices[0]
        text = getattr(choice, "text", None)
        if text is None and isinstance(choice, dict):
            text = choice.get("text")
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is None and isinstance(choice, dict):
            finish_reason = choice.get("finish_reason")

        choice_messages = cast(
            Messages,
            self._prepare_input(prompt)
            + [{"role": "assistant", "content": str(text or "")}],
        )
        self._record_history(choice_messages)

        output = {
            "text": str(text or ""),
            "finish_reason": finish_reason,
        }
        if finish_reason is not None:
            output["stop"] = finish_reason
        return output

    def generate(
        self,
        prompt: str,
        *,
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        **runtime_kwargs,
    ) -> str:
        """Generate text from a raw prompt string via the completions API."""
        result = self._generate_response(
            prompt,
            cache=cache,
            enable_thinking=enable_thinking,
            **runtime_kwargs,
        )
        return str(result.get("text") or "")

    def _stream_chat_completion(
        self,
        input_data: str | BaseModel | list[dict],
        enable_thinking: bool | None = None,
        **runtime_kwargs,
    ):
        """
        Stream text completion directly from the API.

        Note: Caching is not supported when streaming to avoid compatibility
        issues with providers that don't support the /tokenize endpoint.

        Args:
            input_data: Input data (string, BaseModel, or message list)
            **runtime_kwargs: Additional runtime parameters

        Returns:
            Raw stream response object from the API. Caller should iterate over
            it using `for chunk in response`.
        """
        messages = self._prepare_input(input_data)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name, api_kwargs = self._build_api_kwargs(
            effective_kwargs,
            enable_thinking=enable_thinking,
            drop_keys=("stream",),
        )

        # Disable caching for streaming - streaming responses cannot be cached
        if hasattr(self.client, "set_cache"):
            self.client.set_cache(False)

        # Stream directly from API (caching not supported in streaming mode)
        return self.client.chat.completions.create(
            model=model_name, messages=messages, stream=True, **api_kwargs
        )

    def _stream_print(
        self,
        input_data: str | BaseModel | list[dict],
        show_reasoning: bool | None = None,
        **runtime_kwargs,
    ) -> str:
        """
        Stream and print completion with clean output formatting.

        When thinking is enabled, reasoning tokens are streamed by default.
        Callers can still override that behavior by passing show_reasoning explicitly.

        Args:
            input_data: Input data (string, BaseModel, or message list)
            show_reasoning: Whether to print reasoning tokens while streaming.
                When omitted, this follows the effective enable_thinking value.
            **runtime_kwargs: Additional runtime parameters (e.g., max_tokens=500)

        Returns:
            The complete response text (final answer only)
        """
        import sys

        effective_enable_thinking = runtime_kwargs.get(
            "enable_thinking",
            self.enable_thinking,
        )
        should_show_reasoning = (
            bool(effective_enable_thinking)
            if show_reasoning is None
            else show_reasoning
        )

        stream = self._stream_chat_completion(input_data, **runtime_kwargs)

        content_parts: list[str] = []
        reasoning_printed = False

        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Only show reasoning if explicitly requested
            if should_show_reasoning:
                reasoning = self._extract_reasoning_content(delta)
                if reasoning:
                    reasoning_printed = True
                    sys.stdout.write(reasoning)
                    sys.stdout.flush()

            content = delta.content
            if content:
                # Add separator if we showed reasoning
                if should_show_reasoning and reasoning_printed and not content_parts:
                    sys.stdout.write("\n\n" + "=" * 40 + "\n\n")
                    sys.stdout.flush()
                content_parts.append(content)
                sys.stdout.write(content)
                sys.stdout.flush()

        # Final newline
        if content_parts or reasoning_printed:
            sys.stdout.write("\n")
            sys.stdout.flush()

        return "".join(content_parts)

    @clean_traceback
    def pydantic_parse(
        self,
        input_data: str | list[dict],
        *,
        response_model: type[BaseModel],
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        **runtime_kwargs,
    ) -> BaseModel:
        """Execute a structured completion and return the parsed model."""
        if not isinstance(input_data, str) and not isinstance(input_data, list):
            raise TypeError(
                "pydantic_parse expects `input_data` to be a string or message list."
            )
        if not isinstance(response_model, type) or not issubclass(
            response_model, BaseModel
        ):
            raise TypeError(
                "pydantic_parse expects `response_model` to be a Pydantic BaseModel subclass."
            )

        self._set_cache(cache)
        self._require_single_choice(runtime_kwargs)
        # Prepare messages
        messages = self._prepare_input(input_data)

        # Merge runtime kwargs with default model kwargs (runtime takes precedence)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name, api_kwargs = self._build_api_kwargs(
            effective_kwargs,
            enable_thinking=enable_thinking,
        )

        pydantic_model_to_use: type[BaseModel] = cast(
            type[BaseModel], response_model
        )
        try:
            completion = self.client.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=pydantic_model_to_use,
                **api_kwargs,
            )
            # Store raw response from client
            self.last_ai_response = completion
        except Exception as exc:
            # Import openai exceptions for type checking
            from openai import AuthenticationError, BadRequestError, RateLimitError

            if isinstance(exc, (AuthenticationError, RateLimitError, BadRequestError)):
                error_msg = f"OpenAI API error ({type(exc).__name__}): {exc}"
                logger.error(error_msg)
                raise
            is_length_error = "Length" in str(exc) or "maximum context length" in str(
                exc
            )
            if is_length_error:
                raise ValueError(
                    f"Input too long for model {model_name}. Error: {str(exc)[:100]}..."
                ) from exc
            raise

        choices = getattr(completion, "choices", None)
        if not choices:
            raise ValueError("No choices returned from completion.")

        choice = choices[0]
        choice_messages = cast(
            Messages,
            messages + [{"role": "assistant", "content": choice.message.content}],
        )

        # Ensure consistent Pydantic model output for both fresh and cached responses
        parsed_content = choice.message.parsed  # type: ignore[attr-defined]
        if isinstance(parsed_content, dict):
            # Cached response: validate dict back to Pydantic model
            parsed_content = pydantic_model_to_use.model_validate(parsed_content)
        elif not isinstance(parsed_content, pydantic_model_to_use):
            # Fallback: ensure it's the correct type
            parsed_content = pydantic_model_to_use.model_validate(parsed_content)

        reasoning_content = self._extract_reasoning_content(choice.message)
        if reasoning_content:
            choice_messages[-1]["reasoning_content"] = reasoning_content
        self._record_history(choice_messages)
        return cast(BaseModel, parsed_content)

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
        """Convenience wrapper around the explicit runtime methods."""

        if n != 1:
            raise ValueError("LLM only supports n=1")

        # Handle streaming (only for text completion)
        if stream:
            if return_dict:
                raise ValueError(
                    "Streaming is only supported with the default return value."
                )
            pydantic_model = response_model
            if pydantic_model not in (str, None):
                raise ValueError(
                    "Streaming is only supported for text completions, not structured outputs. "
                    "Set response_model=None or response_model=str to use streaming."
                )
            # Disable caching when streaming - warn user and disable on client
            if cache is not False and self.cache:
                logger.warning(
                    "Caching is disabled when streaming. "
                    "Responses will be streamed directly from the API without caching."
                )
            # Explicitly disable caching on the client to prevent pickle errors
            if hasattr(self.client, "set_cache"):
                self.client.set_cache(False)
            return self._stream_chat_completion(
                input_data,
                enable_thinking=enable_thinking,
                **openai_client_kwargs,
            )

        pydantic_model = response_model
        if return_dict:
            if pydantic_model in (str, None):
                result = self._text_completion(
                    input_data,
                    cache=cache,
                    enable_thinking=enable_thinking,
                    **openai_client_kwargs,
                )[0]
            else:
                parsed = self.pydantic_parse(
                    input_data,
                    response_model=cast(type[BaseModel], pydantic_model),
                    cache=cache,
                    enable_thinking=enable_thinking,
                    **openai_client_kwargs,
                )
                completion = self.last_ai_response
                if completion is None:
                    raise ValueError("No completion returned from API.")
                completion_choices = self._get_completion_choices(completion)
                if not completion_choices:
                    raise ValueError("No choices returned from completion.")
                message = self._get_choice_message(completion_choices[0])
                if message is None:
                    raise ValueError("No message returned from completion.")
                result = {
                    "parsed": parsed,
                    "messages": self._last_conversations[-1],
                    "completion": completion,
                    "message": message,
                }
                reasoning_content = self._extract_reasoning_content(message)
                if reasoning_content:
                    result["reasoning_content"] = reasoning_content
                return result

            completion = self.last_ai_response
            if completion is None:
                raise ValueError("No completion returned from API.")
            completion_choices = self._get_completion_choices(completion)
            if not completion_choices:
                raise ValueError("No choices returned from completion.")
            message = self._get_choice_message(completion_choices[0])
            if message is None:
                raise ValueError("No message returned from completion.")
            result["completion"] = completion
            result["message"] = message
            return result

        if pydantic_model in (str, None):
            return self.chat_completion(
                input_data,
                cache=cache,
                enable_thinking=enable_thinking,
                **openai_client_kwargs,
            )
        return self.pydantic_parse(
            input_data,
            response_model=cast(type[BaseModel], pydantic_model),
            cache=cache,
            enable_thinking=enable_thinking,
            **openai_client_kwargs,
        )

    def _inspect_history(
        self, idx: int = -1, k_last_messages: int = 2
    ) -> list[dict[str, Any]]:
        """Inspect the message history of a specific response choice."""
        if hasattr(self, "_last_conversations"):
            from llm_utils import show_chat

            conv = self._last_conversations[idx]
            if k_last_messages > 0:
                conv = conv[-k_last_messages:]
            return show_chat(conv)
        raise ValueError("No message history available. Make a call first.")

from .llm_qwen3 import Qwen3LLM
