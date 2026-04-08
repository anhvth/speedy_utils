# type: ignore

"""Runtime LLM wrapper for OpenAI-compatible chat and text completions."""

# Typing imports
import json
import random
import threading
import time
import weakref
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from httpx import Timeout
from loguru import logger
from pydantic import BaseModel

from speedy_utils import clean_traceback

from .utils import get_base_client


# Lazy import openai types for type checking only
if TYPE_CHECKING:
    from openai import (
        APITimeoutError,
        AuthenticationError,
        BadRequestError,
        OpenAI,
        RateLimitError,
    )
    from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageParam
    from openai.types.completion import CompletionChoice

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
        client: "OpenAI | int | str | list | None" = None,  # type: ignore[name-defined]
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
        self.cache = cache
        self._verbose = verbose
        # Avoid importing OpenAI client class at module import time.
        # If a client object provides an api_key attribute, use it.
        self.api_key = "abc"
        if client is not None and not isinstance(client, list):
            api_key = getattr(client, "api_key", None)
            if isinstance(api_key, str) and api_key:
                self.api_key = api_key

        raw_clients = get_base_client(
            client,
            cache=cache,
            api_key=self.api_key,
        )

        # Handle list of clients for load balancing
        if isinstance(raw_clients, list):
            self._clients = raw_clients
            self._alive_clients: list[Any] = []
            self._check_clients_health()
            if not self._alive_clients:
                raise RuntimeError(
                    f"None of the {len(self._clients)} provided clients are alive."
                )
        else:
            # Single client mode
            self._clients = [raw_clients]
            # check connection of client
            try:
                raw_clients.models.list()
                self._alive_clients = [raw_clients]
            except Exception as e:
                logger.error(
                    f"Failed to connect to OpenAI client: {str(e)}, base_url={raw_clients.base_url}"
                )
                raise e

        if verbose:
            available_models = [
                model.id for model in self._alive_clients[0].models.list().data
            ]
            logger.info(f"Available models: {available_models}")

        if not self.model_kwargs.get("model", ""):
            self.model_kwargs["model"] = self._alive_clients[0].models.list().data[0].id

        self._multiple_clients = len(self._alive_clients) > 1
        self._client_balance_lock = threading.Lock()
        self._client_inflight_counts = [0 for _ in self._alive_clients]
        self._client_total_counts = [0 for _ in self._alive_clients]
        self._client_index_by_id = {
            id(client): idx for idx, client in enumerate(self._alive_clients)
        }
        self._client_last_activity = 0.0
        self._load_balance_report_interval = 5.0
        self._load_balance_report_stop = threading.Event()
        self._load_balance_report_thread: threading.Thread | None = None
        if self._multiple_clients and self._verbose:
            self._start_load_balance_reporter()

    def _check_clients_health(self) -> None:
        """Check health of all clients and populate _alive_clients list."""
        self._alive_clients = []
        dead_urls = []

        for client in self._clients:
            try:
                client.models.list()
                self._alive_clients.append(client)
            except Exception as e:
                url = getattr(client, "base_url", str(client))
                dead_urls.append(url)
                logger.debug(f"Client {url} is not alive: {e}")

        if dead_urls:
            logger.warning(
                f"{len(dead_urls)} of {len(self._clients)} clients are not alive. "
                f"Using {len(self._alive_clients)} alive clients. "
                f"Dead clients: {dead_urls}"
            )

    @property
    def client(self) -> Any:
        """Return a random alive client for load balancing."""
        return self._select_client()

    def _select_client(self) -> Any:
        """Pick one alive client without modifying in-flight accounting."""
        if not self._alive_clients:
            raise RuntimeError("No alive clients available.")
        if len(self._alive_clients) == 1:
            return self._alive_clients[0]
        return random.choice(self._alive_clients)

    def _client_label(self, client: Any) -> str:
        """Return a compact label for a client in load-balance logs."""
        base_url = getattr(client, "base_url", None)
        if base_url is not None:
            return str(base_url)
        return getattr(client, "__class__", type(client)).__name__

    def _load_balance_snapshot(self) -> str | None:
        """Build a snapshot string for current in-flight client usage."""
        with self._client_balance_lock:
            total_inflight = sum(self._client_inflight_counts)
            if total_inflight <= 0:
                return None
            parts = [
                f"{self._client_label(client)}={count}"
                for client, count in zip(
                    self._alive_clients, self._client_inflight_counts, strict=False
                )
            ]
        return f"active={total_inflight} | " + ", ".join(parts)

    def _start_load_balance_reporter(self) -> None:
        """Start a daemon thread that prints the current client split."""
        if self._load_balance_report_thread is not None:
            return

        weak_self = weakref.ref(self)

        def _report() -> None:
            while True:
                obj = weak_self()
                if obj is None:
                    return
                if obj._load_balance_report_stop.wait(
                    obj._load_balance_report_interval
                ):
                    return
                snapshot = obj._load_balance_snapshot()
                if snapshot is not None:
                    logger.info(f"LLM client load balance: {snapshot}")

        thread = threading.Thread(
            target=_report,
            name="LLMClientLoadBalanceReporter",
            daemon=True,
        )
        self._load_balance_report_thread = thread
        thread.start()

    @contextmanager
    def _borrow_client(self):
        """Yield one client while tracking its in-flight request count."""
        client = self._select_client()
        with self._borrow_specific_client(client) as borrowed_client:
            yield borrowed_client

    def _get_tracked_client(self, client_idx: int) -> Any:
        """Return the alive client at a tracked index or raise clearly."""
        if client_idx < 0 or client_idx >= len(self._alive_clients):
            raise RuntimeError(f"Tracked client index {client_idx} is out of range.")
        return self._alive_clients[client_idx]

    @contextmanager
    def _borrow_client_by_index(self, client_idx: int):
        """Yield one specific tracked client while tracking in-flight usage."""
        client = self._get_tracked_client(client_idx)
        with self._borrow_specific_client(client) as borrowed_client:
            yield borrowed_client

    @contextmanager
    def _borrow_specific_client(self, client: Any):
        """Yield a specific tracked client while tracking in-flight usage."""
        client_idx = self._client_index_by_id.get(id(client))
        if client_idx is None:
            raise RuntimeError("Selected client is not tracked.")

        with self._client_balance_lock:
            self._client_inflight_counts[client_idx] += 1
            self._client_total_counts[client_idx] += 1
            self._client_last_activity = time.monotonic()

        try:
            yield client
        finally:
            with self._client_balance_lock:
                self._client_inflight_counts[client_idx] -= 1

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
    def _get_choice_message(choice: Any) -> Any:
        """Return the assistant message from a choice-like object."""
        message = getattr(choice, "message", None)
        if message is None and isinstance(choice, dict):
            message = choice.get("message")
        return message

    @staticmethod
    def _coerce_completion_choice(choice: Any) -> "CompletionChoice":
        """Normalize a choice-like object into an OpenAI CompletionChoice."""
        from openai.types.completion import CompletionChoice

        if isinstance(choice, CompletionChoice):
            return deepcopy(choice)

        choice_data = LLM._completion_choice_data(choice)
        return CompletionChoice.model_validate(choice_data)

    @staticmethod
    def _completion_choice_data(choice: Any) -> dict[str, Any]:
        """Extract completion choice data while preserving backend metadata."""
        if hasattr(choice, "model_dump"):
            try:
                choice_data = dict(
                    choice.model_dump(mode="python", round_trip=True)
                )
            except TypeError:
                choice_data = dict(choice.model_dump())
        elif isinstance(choice, dict):
            choice_data = dict(choice)
        elif hasattr(choice, "__dict__"):
            choice_data = dict(vars(choice))
        else:
            choice_data = {}

        choice_data.setdefault(
            "finish_reason", getattr(choice, "finish_reason", None)
        )
        choice_data.setdefault("index", getattr(choice, "index", 0))
        choice_data.setdefault("logprobs", getattr(choice, "logprobs", None))
        choice_data.setdefault("text", getattr(choice, "text", None))

        for extra_name in (
            "stop_reason",
            "token_ids",
            "prompt_logprobs",
            "prompt_token_ids",
        ):
            if extra_name not in choice_data and hasattr(choice, extra_name):
                choice_data[extra_name] = getattr(choice, extra_name)

        model_extra = getattr(choice, "model_extra", None)
        if isinstance(model_extra, dict):
            for key, value in model_extra.items():
                choice_data.setdefault(key, value)

        return choice_data

    @staticmethod
    def _get_completion_choice(completion: Any) -> "CompletionChoice":
        """Return the first normalized choice from a completion-like object."""
        choices = getattr(completion, "choices", None)
        if choices is None and isinstance(completion, dict):
            choices = completion.get("choices")
        if not choices:
            raise ValueError("No choices returned from completion.")
        return LLM._coerce_completion_choice(choices[0])

    @staticmethod
    def _get_completion_usage(completion: Any) -> Any | None:
        """Return the usage payload from a completion-like object."""
        usage = getattr(completion, "usage", None)
        if usage is None and isinstance(completion, dict):
            usage = completion.get("usage")
        return usage

    def _set_cache(self, cache: bool | None, *, client: Any | None = None) -> None:
        """Update client-side caching when supported."""
        if cache is None:
            return
        client_to_use = self.client if client is None else client
        if hasattr(client_to_use, "set_cache"):
            client_to_use.set_cache(cache)
            return
        logger.warning("Client does not support caching.")

    @staticmethod
    def _require_single_choice(runtime_kwargs: dict[str, Any]) -> None:
        """Reject multi-choice requests in the simplified public API."""
        n = runtime_kwargs.get("n", 1)
        if n != 1:
            raise ValueError("LLM only supports n=1")

    def _record_history(self, conversation_messages: Messages) -> None:
        """Track recent message history for debugging helpers."""
        if not hasattr(self, "_last_conversations"):
            self._last_conversations = []
        else:
            self._last_conversations = self._last_conversations[-100:]
        self._last_conversations.append(conversation_messages)

    @clean_traceback
    def _chat_completion_result(
        self,
        input_data: str | BaseModel | list[dict],
        *,
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        **runtime_kwargs,
    ) -> list[dict[str, Any]]:
        """Execute a chat completion call and return normalized internal results."""
        self._require_single_choice(runtime_kwargs)
        # Prepare messages
        messages = self._prepare_input(input_data)

        # Merge runtime kwargs with default model kwargs (runtime takes precedence)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name, api_kwargs = self._build_api_kwargs(
            effective_kwargs,
            enable_thinking=enable_thinking,
        )

        with self._borrow_client() as client:
            self._set_cache(cache, client=client)
            try:
                completion = client.chat.completions.create(
                    model=model_name, messages=messages, **api_kwargs
                )
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
        message = self._get_choice_message(choice)
        if message is None:
            raise ValueError("No message returned from completion.")

        assistant_message = [{"role": "assistant", "content": message.content}]
        reasoning_content = self._extract_reasoning_content(message)
        if reasoning_content:
            assistant_message[0]["reasoning_content"] = reasoning_content

        conversation_messages = cast(Messages, messages + assistant_message)
        result = {
            "parsed": message.content,
            "messages": conversation_messages,
            "completion": completion,
            "message": message,
        }
        if reasoning_content:
            result["reasoning_content"] = reasoning_content

        self._record_history(conversation_messages)
        return [result]

    @clean_traceback
    def chat_completion(
        self,
        input_data: str | BaseModel | list[dict],
        *,
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        **runtime_kwargs,
    ) -> "ChatCompletionMessage":
        """Call the chat completions API and return the first assistant message."""
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

    def generate(
        self,
        prompt: str,
        *,
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        **runtime_kwargs,
    ) -> "CompletionChoice":
        """Call the completions API to continue a raw prompt with more tokens."""
        if not isinstance(prompt, str):
            raise TypeError("generate expects `prompt` to be a string")

        self._require_single_choice(runtime_kwargs)

        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name, api_kwargs = self._build_api_kwargs(
            effective_kwargs,
            enable_thinking=enable_thinking,
        )

        with self._borrow_client() as client:
            self._set_cache(cache, client=client)
            try:
                completion = client.completions.create(
                    model=model_name,
                    prompt=prompt,
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

        conversation_messages = cast(
            Messages,
            self._prepare_input(prompt)
            + [{"role": "assistant", "content": str(choice.text or "")}],
        )
        self._record_history(conversation_messages)
        return choice

    @clean_traceback
    def _pydantic_completion(
        self,
        input_data: str | list[dict],
        *,
        response_model: type[BaseModel],
        cache: bool | None = None,
        enable_thinking: bool | None = None,
        **runtime_kwargs,
    ) -> dict[str, Any]:
        """Execute a structured completion and return parsed and raw artifacts."""
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

        self._require_single_choice(runtime_kwargs)
        # Prepare messages
        messages = self._prepare_input(input_data)

        # Merge runtime kwargs with default model kwargs (runtime takes precedence)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name, api_kwargs = self._build_api_kwargs(
            effective_kwargs,
            enable_thinking=enable_thinking,
        )

        pydantic_model_to_use: type[BaseModel] = cast(type[BaseModel], response_model)
        with self._borrow_client() as client:
            self._set_cache(cache, client=client)
            try:
                completion = client.chat.completions.parse(
                    model=model_name,
                    messages=messages,
                    response_format=pydantic_model_to_use,
                    **api_kwargs,
                )
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
        message = self._get_choice_message(choice)
        if message is None:
            raise ValueError("No message returned from completion.")
        conversation_messages = cast(
            Messages,
            messages + [{"role": "assistant", "content": message.content}],
        )

        # Ensure consistent Pydantic model output for both fresh and cached responses
        parsed_content = message.parsed  # type: ignore[attr-defined]
        if isinstance(parsed_content, dict):
            # Cached response: validate dict back to Pydantic model
            parsed_content = pydantic_model_to_use.model_validate(parsed_content)
        elif isinstance(parsed_content, str):
            parsed_content = pydantic_model_to_use.model_validate_json(parsed_content)
        elif not isinstance(parsed_content, pydantic_model_to_use):
            # Fallback: ensure it's the correct type
            parsed_content = pydantic_model_to_use.model_validate(parsed_content)

        reasoning_content = self._extract_reasoning_content(message)
        if reasoning_content:
            conversation_messages[-1]["reasoning_content"] = reasoning_content
        self._record_history(conversation_messages)
        result = {
            "parsed": cast(BaseModel, parsed_content),
            "messages": conversation_messages,
            "completion": completion,
            "message": message,
        }
        if reasoning_content:
            result["reasoning_content"] = reasoning_content
        return result

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
        result = self._pydantic_completion(
            input_data,
            response_model=response_model,
            cache=cache,
            enable_thinking=enable_thinking,
            **runtime_kwargs,
        )
        return cast(BaseModel, result["parsed"])

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
        """Convenience wrapper around the explicit chat and completion methods."""

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
            messages = self._prepare_input(input_data)
            effective_kwargs = {**self.model_kwargs, **openai_client_kwargs}
            model_name, api_kwargs = self._build_api_kwargs(
                effective_kwargs,
                enable_thinking=enable_thinking,
                drop_keys=("stream",),
            )
            with self._borrow_client() as client:
                # Explicitly disable caching on the selected client to prevent pickle errors
                self._set_cache(False, client=client)
                return client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    stream=True,
                    **api_kwargs,
                )

        pydantic_model = response_model
        if return_dict:
            if pydantic_model in (str, None):
                result = self._chat_completion_result(
                    input_data,
                    cache=cache,
                    enable_thinking=enable_thinking,
                    **openai_client_kwargs,
                )[0]
            else:
                result = self._pydantic_completion(
                    input_data,
                    response_model=cast(type[BaseModel], pydantic_model),
                    cache=cache,
                    enable_thinking=enable_thinking,
                    **openai_client_kwargs,
                )
                return result
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

    def inspect_history(
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
