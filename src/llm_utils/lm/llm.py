# type: ignore

"""Runtime LLM wrapper for OpenAI-compatible chat and text completions."""

# Typing imports
import json
import os
import random
import threading
import time
import weakref
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING, Any, cast

from httpx import Timeout
from loguru import logger
from pydantic import BaseModel

from .._traceback import clean_traceback
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


_CLIENT_BOOTSTRAP_TTL_SECONDS = 60.0
_client_bootstrap_cache_lock = threading.Lock()
_client_bootstrap_cache: dict[tuple[Any, ...], tuple[float, tuple[int, ...], Any]] = {}


def _normalize_client_bootstrap_key(client: Any) -> tuple[Any, ...]:
    """Build a stable cache key for one client spec."""
    if client is None:
        return ("none",)
    if isinstance(client, int):
        return ("int", client)
    if isinstance(client, str):
        return ("str", client)
    if isinstance(client, list):
        return (
            "list",
            tuple(_normalize_client_bootstrap_key(item) for item in client),
        )

    base_url = getattr(client, "base_url", None)
    if base_url is not None:
        return ("base_url", str(base_url))
    return ("object", id(client))


def _client_bootstrap_cache_key(client: Any, api_key: str) -> tuple[Any, ...]:
    """Build the process-local cache key for LLM bootstrap metadata."""
    return (_normalize_client_bootstrap_key(client), api_key)


def _get_or_create_client_bootstrap(
    cache_key: tuple[Any, ...],
    bootstrap_fn: Callable[[], tuple[list[int], Any | None]],
) -> tuple[list[int], Any | None]:
    """Reuse recent client bootstrap results across LLM instances."""
    now = time.monotonic()
    with _client_bootstrap_cache_lock:
        cached = _client_bootstrap_cache.get(cache_key)
        if cached is not None:
            expires_at, cached_alive_indices, cached_primary_models_response = cached
            if expires_at > now:
                return list(cached_alive_indices), cached_primary_models_response
            _client_bootstrap_cache.pop(cache_key, None)

        alive_indices, primary_models_response = bootstrap_fn()
        if alive_indices and primary_models_response is not None:
            _client_bootstrap_cache[cache_key] = (
                now + _CLIENT_BOOTSTRAP_TTL_SECONDS,
                tuple(alive_indices),
                primary_models_response,
            )
        return alive_indices, primary_models_response


_DEFAULT_WAIT_POLL_INTERVAL = 1.0


# Optional bootstrap wait helper. Older builds / partial source trees may
# not define this; default to None so the call sites below can fall back to
# calling bootstrap_fn() directly.
_wait_for_endpoint_bootstrap = None  # type: ignore[assignment]


def _wait_for_endpoint_bootstrap(  # type: ignore[no-redef]
    llm: "LLM",
    cache_key: tuple[Any, ...],
    bootstrap_fn: Callable[[], tuple[list[int], Any | None]],
    *,
    wait_for_endpoint: float,
) -> tuple[list[int], Any | None]:
    """Run ``bootstrap_fn``, waiting up to ``wait_for_endpoint`` seconds.

    The client/endpoint is expected to come up shortly after the LLM is
    constructed (e.g. a vLLM worker is starting). When the bootstrap fails
    on the first try, we log a progress message like
    ``"waiting for 10/600..."`` every poll interval and retry until either
    the endpoint responds or the budget is exhausted.

    A ``wait_for_endpoint`` value of ``0`` (or any non-positive number)
    disables the wait and surfaces the bootstrap error immediately,
    preserving the original fast-fail behaviour.
    """
    wait_budget = max(0.0, float(wait_for_endpoint))
    poll_interval = _DEFAULT_WAIT_POLL_INTERVAL

    # While the constructor is intentionally waiting, suppress per-attempt
    # error logs from the bootstrap helper so the terminal isn't spammed
    # with the same "Connection error" line every second for the full wait
    # budget. The progress line and the final "did not become ready"
    # error are still emitted.
    llm._waiting_for_endpoint = True
    try:
        # Fast path: try the regular bootstrap (which may also serve a
        # cached recent result) without waiting.
        try:
            result = _get_or_create_client_bootstrap(cache_key, bootstrap_fn)
        except Exception:
            result = ([], None)

        if result[0] and result[1] is not None:
            return result

        if wait_budget <= 0:
            # Re-raise the original bootstrap error so the caller sees the
            # same failure as before the wait feature was introduced.
            return _get_or_create_client_bootstrap(cache_key, bootstrap_fn)

        deadline = time.monotonic() + wait_budget
        elapsed = 0.0
        client_label = _describe_bootstrap_target(llm)
        print(
            f"LLM endpoint {client_label} is not ready; waiting (budget={wait_budget:.0f}s)...",
            flush=True,
        )
        while True:
            time.sleep(poll_interval)
            elapsed = min(wait_budget, elapsed + poll_interval)
            try:
                result = _get_or_create_client_bootstrap(cache_key, bootstrap_fn)
            except Exception:  # noqa: BLE001
                result = ([], None)
            if result[0] and result[1] is not None:
                print(
                    f"\rLLM endpoint {client_label} became ready after {elapsed:.0f}s",
                    flush=True,
                )
                return result
            if time.monotonic() >= deadline:
                logger.error(
                    "LLM endpoint {} did not become ready within {:.0f}s",
                    client_label,
                    wait_budget,
                )
                return _get_or_create_client_bootstrap(cache_key, bootstrap_fn)
            print(
                f"\rwaiting for {elapsed:.0f}/{wait_budget:.0f}...",
                end="",
                flush=True,
            )
    finally:
        llm._waiting_for_endpoint = False


def _describe_bootstrap_target(llm: "LLM") -> str:
    """Return a short human-readable label for the LLM's bootstrap target."""
    clients = getattr(llm, "_clients", None) or []
    if not clients:
        return "endpoint"
    if len(clients) == 1:
        return llm._client_label(clients[0])
    return f"{len(clients)} endpoints"


def _parse_with_warnings(
    client: Any,
    model_name: str,
    messages: list[dict],
    pydantic_model_to_use: type,
    api_kwargs: dict[str, Any],
) -> Any:
    """Call chat completions parse, suppressing Pydantic serializer warnings."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Pydantic serializer warnings",
            category=UserWarning,
        )
        return client.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=pydantic_model_to_use,
            **api_kwargs,
        )


def _response_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Read fields from dicts, OpenAI models, or simple test doubles."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    value = getattr(obj, key, default)
    if value is not default:
        return value
    model_extra = getattr(obj, "model_extra", None)
    if isinstance(model_extra, dict):
        return model_extra.get(key, default)
    return default


def _model_id(model: Any) -> str | None:
    model_id = _response_attr(model, "id")
    if isinstance(model_id, str) and model_id:
        return model_id
    return None


def _select_default_model_id(models_response: Any) -> str:
    """Choose the model callers should use when the endpoint exposes metadata."""
    current_inference_model = _response_attr(models_response, "current_inference_model")
    current_model = _response_attr(current_inference_model, "model")
    if isinstance(current_model, str) and current_model:
        return current_model

    models = list(_response_attr(models_response, "data", []) or [])
    for preferred_key in ("active", "loaded"):
        for model in models:
            if _response_attr(model, preferred_key) is True:
                model_id = _model_id(model)
                if model_id:
                    return model_id

    for model in models:
        model_id = _model_id(model)
        if model_id:
            return model_id

    raise RuntimeError("No models available.")


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
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | tuple[str, ...] | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        wait_for_endpoint: float = 600.0,
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
        common_model_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        for key, value in common_model_kwargs.items():
            if value is not None:
                model_kwargs[key] = value
        self.model_kwargs = model_kwargs
        self.timeout = timeout
        self.enable_thinking = enable_thinking
        self.cache = cache
        self._verbose = verbose
        self.wait_for_endpoint = wait_for_endpoint
        # Set while the constructor is waiting for the endpoint to come up.
        # Bootstrap helpers use it to downgrade per-attempt ERROR logs to DEBUG
        # so the terminal isn't spammed during a long intentional wait.
        self._waiting_for_endpoint = False
        # Avoid importing OpenAI client class at module import time.
        # If a client object provides an api_key attribute, use it.
        self.api_key = os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY", "abc")
        if client is not None and not isinstance(client, list):
            client_api_key = getattr(client, "api_key", None)
            if isinstance(client_api_key, str) and client_api_key:
                self.api_key = client_api_key

        raw_clients = get_base_client(
            client,
            cache=cache,
            api_key=self.api_key,
        )
        primary_models_response: Any | None = None
        bootstrap_cache_key = _client_bootstrap_cache_key(raw_clients, self.api_key)

        # Handle list of clients for load balancing
        if isinstance(raw_clients, list):
            self._clients = raw_clients
            # If the bootstrap wait helper is unavailable, call directly.
            if _wait_for_endpoint_bootstrap is None:
                alive_indices, primary_models_response = self._check_clients_health()
            else:
                alive_indices, primary_models_response = _wait_for_endpoint_bootstrap(
                    self,
                    bootstrap_cache_key,
                    self._check_clients_health,
                    wait_for_endpoint=wait_for_endpoint,
                )
            self._alive_clients = [
                raw_clients[idx] for idx in alive_indices if 0 <= idx < len(raw_clients)
            ]
            if not self._alive_clients:
                raise RuntimeError(
                    f"None of the {len(self._clients)} provided clients are alive."
                )
        else:
            # Single client mode
            self._clients = [raw_clients]
            self._alive_clients = []

            def _bootstrap_single_client() -> tuple[list[int], Any | None]:
                try:
                    models_response = raw_clients.models.list()
                    self._alive_clients = [raw_clients]
                    return [0], models_response
                except Exception as e:
                    # While the constructor is intentionally waiting for the
                    # endpoint to come up we stay silent — the
                    # "waiting for N/600..." progress line tells the user
                    # what's going on, and the bootstrap error is only
                    # surfaced (and logged) once the wait budget is
                    # exhausted.
                    if not self._waiting_for_endpoint:
                        logger.error(
                            f"Failed to connect to OpenAI client: {str(e)}, base_url={raw_clients.base_url}"
                        )
                    raise e

            if _wait_for_endpoint_bootstrap is None:
                _, primary_models_response = _bootstrap_single_client()
            else:
                _, primary_models_response = _wait_for_endpoint_bootstrap(
                    self,
                    bootstrap_cache_key,
                    _bootstrap_single_client,
                    wait_for_endpoint=wait_for_endpoint,
                )
            if primary_models_response is not None:
                self._alive_clients = [raw_clients]

        if primary_models_response is None:
            raise RuntimeError("No alive clients available.")
        primary_models = primary_models_response.data

        if verbose:
            available_models = [_model_id(model) for model in primary_models]
            logger.info(f"Available models: {available_models}")

        if not self.model_kwargs.get("model", ""):
            self.model_kwargs["model"] = _select_default_model_id(
                primary_models_response
            )

        self._multiple_clients = len(self._alive_clients) > 1
        self._client_balance_lock = threading.Lock()
        self._client_inflight_counts = [0 for _ in self._alive_clients]
        self._client_index_by_id = {
            id(client): idx for idx, client in enumerate(self._alive_clients)
        }
        self._load_balance_report_interval = 5.0
        self._load_balance_report_stop = threading.Event()
        self._load_balance_report_thread: threading.Thread | None = None
        if self._multiple_clients and self._verbose:
            self._start_load_balance_reporter()

    def _check_clients_health(self) -> tuple[list[int], Any | None]:
        """Check health of all clients and return alive indices and models."""
        self._alive_clients = []
        alive_indices: list[int] = []
        dead_urls = []
        primary_models_response: Any | None = None

        for idx, client in enumerate(self._clients):
            try:
                models_response = client.models.list()
                self._alive_clients.append(client)
                alive_indices.append(idx)
                if primary_models_response is None:
                    primary_models_response = models_response
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

        return alive_indices, primary_models_response

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

    @property
    def temperature(self) -> float | None:
        """Return the temperature from model_kwargs."""
        return self.model_kwargs.get("temperature")

    @property
    def top_p(self) -> float | None:
        """Return the top_p from model_kwargs."""
        return self.model_kwargs.get("top_p")

    @property
    def max_tokens(self) -> int | None:
        """Return the max_tokens from model_kwargs."""
        return self.model_kwargs.get("max_tokens")

    @property
    def stop(self) -> str | list[str] | tuple[str, ...] | None:
        """Return the stop sequences from model_kwargs."""
        return self.model_kwargs.get("stop")

    @property
    def presence_penalty(self) -> float | None:
        """Return the presence_penalty from model_kwargs."""
        return self.model_kwargs.get("presence_penalty")

    @property
    def frequency_penalty(self) -> float | None:
        """Return the frequency_penalty from model_kwargs."""
        return self.model_kwargs.get("frequency_penalty")

    @staticmethod
    def _prepare_input(input_data: str | BaseModel | list[dict]) -> Messages:
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
    def _extract_reasoning(message: Any) -> str | None:
        """Extract canonical reasoning from a response message when present."""
        reasoning = getattr(message, "reasoning", None)
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
                choice_data = dict(choice.model_dump(mode="python", round_trip=True))
            except TypeError:
                choice_data = dict(choice.model_dump())
        elif isinstance(choice, dict):
            choice_data = dict(choice)
        elif hasattr(choice, "__dict__"):
            choice_data = dict(vars(choice))
        else:
            choice_data = {}

        choice_data.setdefault("finish_reason", getattr(choice, "finish_reason", None))
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

    @staticmethod
    def _format_bad_request_body(exc: "BadRequestError") -> str:
        """Format BadRequestError.body for readable logging."""
        body = exc.body
        if body is None:
            return ""
        if isinstance(body, dict):
            # OpenAI-style: {"error": {"message": "...", "code": "..."}}
            err = body.get("error", body)
            if isinstance(err, dict):
                msg = err.get("message", "") or ""
                code = err.get("code", "") or ""
                return f" [{code}] {msg}" if code else f" {msg}"
            return f" {err}"
        # String body (common with vLLM overload): just the message itself
        if "invalid http request" in str(body).lower():
            return " body=Invalid HTTP request received (vLLM connection overload — reduce concurrency or increase vLLM capacity)"
        return f" body={body}"

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Check if the exception is transient and worth retrying."""
        from openai import APITimeoutError, BadRequestError, RateLimitError

        if isinstance(exc, (APITimeoutError, RateLimitError)):
            return True
        # BadRequestError can be transient when vLLM is overloaded
        if isinstance(exc, BadRequestError):
            body = exc.body
            if isinstance(body, dict):
                err = body.get("error", body) if isinstance(body, dict) else body
                code = (err.get("code") or "") if isinstance(err, dict) else ""
                # Non-retryable: input validation, auth-style codes
                if code in ("context_length_exceeded", "content_filter"):
                    return False
            return True
        return False

    @staticmethod
    def _handle_completion_exception(
        exc: Exception,
        *,
        api_kwargs: dict[str, Any],
        model_name: str,
    ) -> None:
        """Normalize completion-style API exceptions. Raises always (caller retries)."""
        from openai import (
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            RateLimitError,
        )

        if isinstance(exc, APITimeoutError):
            logger.warning(
                f"OpenAI API timeout ({api_kwargs['timeout']}) for {model_name}: {exc}"
            )
            raise exc
        if isinstance(exc, (AuthenticationError, RateLimitError, BadRequestError)):
            body_detail = ""
            if isinstance(exc, BadRequestError):
                body_detail = LLM._format_bad_request_body(exc)
            # RateLimit: warn, not error (transient)
            logger_func = (
                logger.warning if isinstance(exc, RateLimitError) else logger.error
            )
            logger_func(f"OpenAI API error ({type(exc).__name__}): {exc}{body_detail}")
            raise exc
        if isinstance(exc, ValueError):
            logger.error(f"ValueError during API call: {exc}")
            raise exc
        is_length_error = "Length" in str(exc) or "maximum context length" in str(exc)
        if is_length_error:
            raise ValueError(
                f"Input too long for model {model_name}. Error: {str(exc)[:100]}..."
            ) from exc
        raise exc

    @staticmethod
    def _call_with_retry(
        fn: Callable[[], Any],
        *,
        max_retries: int = 5,
        model_name: str = "",
    ) -> Any:
        """Call fn, retrying on transient errors with random sleep 0.1--2s."""
        from openai import APITimeoutError, BadRequestError, RateLimitError

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except (APITimeoutError, RateLimitError, BadRequestError) as exc:
                last_exc = exc
                if not LLM._is_retryable(exc):
                    raise
                if attempt >= max_retries:
                    logger.error(
                        f"Giving up after {max_retries} retries for {model_name}: {exc}"
                    )
                    raise
                sleep_dur = random.uniform(0.1, 2.0)
                hint = ""
                if isinstance(exc, BadRequestError):
                    b = exc.body
                    if isinstance(b, str) and "invalid http" in b.lower():
                        hint = " vLLM overload"
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} {model_name}"
                    f" sleep={sleep_dur:.2f}s{hint}"
                )
                time.sleep(sleep_dur)
            except Exception as exc:
                last_exc = exc
                raise
        # Should not reach here, but satisfy type narrowing
        raise last_exc  # type: ignore[misc]

    def _raw_completion_step(
        self,
        prompt: str,
        *,
        cache: bool | None = None,
        client_idx: int | None = None,
        return_client_idx: bool = False,
        enable_thinking: bool | None = None,
        drop_keys: tuple[str, ...] = (),
        **runtime_kwargs,
    ) -> "CompletionChoice | tuple[CompletionChoice, int]":
        """Run one text completion request against a fully-built prompt."""
        if not isinstance(prompt, str):
            raise TypeError("_raw_completion_step expects `prompt` to be a string")

        call_kwargs = dict(runtime_kwargs)
        self._require_single_choice(call_kwargs)

        effective_kwargs = {**self.model_kwargs, **call_kwargs}
        if effective_kwargs.get("max_tokens") is None:
            effective_kwargs["max_tokens"] = 1
        model_name, api_kwargs = self._build_api_kwargs(
            effective_kwargs,
            enable_thinking=enable_thinking,
            drop_keys=drop_keys,
        )
        for key in drop_keys:
            api_kwargs.pop(key, None)

        borrow_client = (
            self._borrow_client()
            if client_idx is None
            else self._borrow_client_by_index(client_idx)
        )
        with borrow_client as client:
            self._set_cache(cache, client=client)
            completion = LLM._call_with_retry(
                lambda: client.completions.create(
                    model=model_name,
                    prompt=prompt,
                    **api_kwargs,
                ),
                model_name=model_name,
            )

        choice = self._get_completion_choice(completion)
        usage = self._get_completion_usage(completion)
        if usage is not None:
            choice.usage = usage

        if return_client_idx:
            return choice, self._client_index_by_id[id(client)]
        return choice

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
            completion = LLM._call_with_retry(
                lambda: client.chat.completions.create(
                    model=model_name, messages=messages, **api_kwargs
                ),
                model_name=model_name,
            )
        # print(completion)

        choices = getattr(completion, "choices", None)
        if not choices:
            raise ValueError("No choices returned from completion.")

        choice = choices[0]
        message = self._get_choice_message(choice)
        if message is None:
            raise ValueError("No message returned from completion.")

        assistant_message = [{"role": "assistant", "content": message.content}]
        reasoning = self._extract_reasoning(message)
        if reasoning:
            assistant_message[0]["reasoning"] = reasoning

        conversation_messages = cast(Messages, messages + assistant_message)
        result = {
            "parsed": message.content,
            "messages": conversation_messages,
            "completion": completion,
            "message": message,
        }
        if reasoning:
            result["reasoning"] = reasoning

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
        choice = self._raw_completion_step(
            prompt,
            cache=cache,
            enable_thinking=enable_thinking,
            **runtime_kwargs,
        )
        if not hasattr(choice, "text"):
            raise ValueError("No text returned from completion.")

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
            completion = LLM._call_with_retry(
                lambda: _parse_with_warnings(
                    client, model_name, messages, pydantic_model_to_use, api_kwargs
                ),
                model_name=model_name,
            )

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

        reasoning = self._extract_reasoning(message)
        if reasoning:
            conversation_messages[-1]["reasoning"] = reasoning
        self._record_history(conversation_messages)
        result = {
            "parsed": cast(BaseModel, parsed_content),
            "messages": conversation_messages,
            "completion": completion,
            "message": message,
        }
        if reasoning:
            result["reasoning"] = reasoning
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

    def _stream_chat(
        self,
        input_data: str | BaseModel | list[dict],
        *,
        response_model: type[BaseModel] | type[str] | None,
        return_dict: bool,
        cache: bool | None,
        enable_thinking: bool | None,
        **runtime_kwargs,
    ) -> Any:
        """Stream a chat completion directly from the API (no caching)."""
        if return_dict:
            raise ValueError(
                "Streaming is only supported with the default return value."
            )
        if response_model not in (str, None):
            raise ValueError(
                "Streaming is only supported for text completions, not structured outputs. "
                "Set response_model=None or response_model=str to use streaming."
            )
        if cache is not False and self.cache:
            logger.warning(
                "Caching is disabled when streaming. "
                "Responses will be streamed directly from the API without caching."
            )

        messages = self._prepare_input(input_data)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name, api_kwargs = self._build_api_kwargs(
            effective_kwargs,
            enable_thinking=enable_thinking,
            drop_keys=("stream",),
        )
        with self._borrow_client() as client:
            # Disable caching on the selected client to prevent pickle errors.
            self._set_cache(False, client=client)
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                **api_kwargs,
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
        """Route one request to the right chat/completion/streaming path."""
        if n != 1:
            raise ValueError("LLM only supports n=1")

        if stream:
            return self._stream_chat(
                input_data,
                response_model=response_model,
                return_dict=return_dict,
                cache=cache,
                enable_thinking=enable_thinking,
                **openai_client_kwargs,
            )

        if return_dict:
            if response_model in (str, None):
                return self._chat_completion_result(
                    input_data,
                    cache=cache,
                    enable_thinking=enable_thinking,
                    **openai_client_kwargs,
                )[0]
            return self._pydantic_completion(
                input_data,
                response_model=cast(type[BaseModel], response_model),
                cache=cache,
                enable_thinking=enable_thinking,
                **openai_client_kwargs,
            )

        if response_model in (str, None):
            return self.chat_completion(
                input_data,
                cache=cache,
                enable_thinking=enable_thinking,
                **openai_client_kwargs,
            )
        return self.pydantic_parse(
            input_data,
            response_model=cast(type[BaseModel], response_model),
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


__all__ = ["LLM", "Qwen3LLM"]
