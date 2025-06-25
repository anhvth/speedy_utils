"""
# ============================================================================= #
# SYNCHRONOUS LANGUAGE MODEL WRAPPER WITH OPENAI COMPATIBILITY
# ============================================================================= #
#
# Title & Intent:
# Unified synchronous language model interface with caching, type safety, and OpenAI API compatibility
#
# High-level Summary:
# This module provides a comprehensive synchronous wrapper for language models that supports both
# string prompts and structured Pydantic model responses. It includes intelligent caching with
# content-based hashing, automatic retry logic for rate limits, and seamless integration with
# OpenAI-compatible APIs. The LM class handles message formatting, response parsing, token counting,
# and provides detailed logging and debugging capabilities for production use.
#
# Public API / Data Contracts:
# • LM(model, temperature=0.0, max_tokens=2000, host="localhost", port=None, **kwargs) - Main wrapper class
# • LM.__call__(prompt=None, messages=None, response_format=str, cache=None, **kwargs) -> str | BaseModel
# • LM.list_models(port=None) -> List[str] - Enumerate available models
# • LM.count_tokens(messages, model=None) -> int - Token counting utility
# • LM.price(messages, model=None, response_tokens=0) -> float - Cost estimation
# • LM.set_model(model_name) -> None - Runtime model switching
# • TModel = TypeVar("TModel", bound=BaseModel) - Generic type for structured responses
# • Messages = List[ChatCompletionMessageParam] - Typed message format
# • RawMsgs = Union[Messages, LegacyMsgs] - Flexible input format
#
# Invariants / Constraints:
# • MUST provide either 'prompt' or 'messages' parameter, but not both
# • MUST set model name before making API calls (auto-detection available)
# • response_format=str MUST return string; response_format=PydanticModel MUST return model instance
# • Caching MUST use content-based hashing for reproducible results
# • MUST handle OpenAI rate limits with exponential backoff (up to 3 retries)
# • MUST preserve message order and format during transformations
# • Token counting SHOULD use tiktoken when available, fall back to character estimation
# • MUST validate Pydantic responses and retry on parsing failures
#
# Usage Example:
# ```python
# from llm_utils.lm.sync_lm import LM
# from pydantic import BaseModel
#
# class CodeResponse(BaseModel):
#     language: str
#     code: str
#     explanation: str
#
# # String response
# lm = LM(model="gpt-4o-mini", temperature=0.1)
# response = lm(prompt="Write a Python hello world")
# print(response)  # Returns string
#
# # Structured response
# code_response = lm(
#     prompt="Write a Python function to calculate fibonacci",
#     response_format=CodeResponse
# )
# print(f"Language: {code_response.language}")  # Returns CodeResponse instance
#
# # Message-based conversation
# messages = [
#     {"role": "system", "content": "You are a helpful coding assistant"},
#     {"role": "user", "content": "Explain async/await in Python"}
# ]
# response = lm(messages=messages, max_tokens=1000)
# ```
#
# TODO & Future Work:
# • Add streaming response support for long-form generation
# • Implement fine-grained token usage tracking per conversation
# • Add support for function calling and tool use
# • Optimize caching strategy for conversation contexts
# • Add async context manager support for resource cleanup
#
# ============================================================================= #
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from abc import ABC
from functools import lru_cache
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from loguru import logger
from openai import AuthenticationError, OpenAI, RateLimitError
from openai.pagination import SyncPage
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.model import Model
from pydantic import BaseModel

from llm_utils.chat_format.display import get_conversation_one_turn
from speedy_utils.common.utils_io import jdumps

# --------------------------------------------------------------------------- #
# type helpers
# --------------------------------------------------------------------------- #
TModel = TypeVar("TModel", bound=BaseModel)
Messages = List[ChatCompletionMessageParam]  # final, already-typed messages
LegacyMsgs = List[Dict[str, str]]  # old “…role/content…” dicts
RawMsgs = Union[Messages, LegacyMsgs]  # what __call__ accepts


# --------------------------------------------------------------------------- #
# color formatting helpers
# --------------------------------------------------------------------------- #
def _red(text: str) -> str:
    """Format text with red color."""
    return f"\x1b[31m{text}\x1b[0m"


def _green(text: str) -> str:
    """Format text with green color."""
    return f"\x1b[32m{text}\x1b[0m"


def _blue(text: str) -> str:
    """Format text with blue color."""
    return f"\x1b[34m{text}\x1b[0m"


def _yellow(text: str) -> str:
    """Format text with yellow color."""
    return f"\x1b[33m{text}\x1b[0m"


# from functools import lru_cache


# @lru_cache(maxsize=10)
# def get_tok(tokenizer_name):
#     from transformers import AutoTokenizer

#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
#     return tokenizer


class LM:
    """
    Unified language-model wrapper.

    • `response_format=str`               → returns `str`
    • `response_format=YourPydanticModel` → returns that model instance
    """

    # --------------------------------------------------------------------- #
    # ctor / plumbing
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2_000,
        host: str = "localhost",
        port: Optional[int | str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: bool = True,
        **openai_kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url or (f"http://{host}:{port}/v1" if port else None)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "abc")
        self.openai_kwargs = openai_kwargs
        self.do_cache = cache
        self._init_port = port  # <-- store the port provided at init

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def set_model(self, model: str) -> None:
        """Set the model name after initialization."""
        self.model = model

    # --------------------------------------------------------------------- #
    # public API – typed overloads
    # --------------------------------------------------------------------- #
    @overload
    def __call__(
        self,
        *,
        prompt: str | None = ...,
        messages: RawMsgs | None = ...,
        response_format: type[str] = str,
        return_openai_response: bool = ...,
        **kwargs: Any,
    ) -> str: ...

    @overload
    def __call__(
        self,
        *,
        prompt: str | None = ...,
        messages: RawMsgs | None = ...,
        response_format: Type[TModel],
        return_openai_response: bool = ...,
        **kwargs: Any,
    ) -> TModel: ...

    # single implementation
    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[RawMsgs] = None,
        response_format: Union[type[str], Type[BaseModel]] = str,
        cache: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        return_openai_response: bool = False,
        **kwargs: Any,
    ):
        # argument validation ------------------------------------------------
        if (prompt is None) == (messages is None):
            raise ValueError("Provide *either* `prompt` or `messages` (but not both).")

        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        assert messages is not None  # for type-checker

        # If model is not specified, but port is provided, use the first available model
        if self.model is None:
            port = self._init_port
            if port:
                available_models = self.list_models(port=port)
                if available_models:
                    self.model = available_models[0]
                    logger.debug(f"Auto-selected model: {self.model}")
                else:
                    raise ValueError("No models available to select from.")
            else:
                raise AssertionError("Model must be set before calling.")

        openai_msgs: Messages = (
            self._convert_messages(cast(LegacyMsgs, messages))
            if isinstance(messages[0], dict)  # legacy style
            else cast(Messages, messages)  # already typed
        )

        kw = dict(
            self.openai_kwargs,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        kw.update(kwargs)
        use_cache = self.do_cache if cache is None else cache

        raw_response = self._call_raw(
            openai_msgs,
            response_format=response_format,
            use_cache=use_cache,
            **kw,
        )

        if return_openai_response:
            response = raw_response
        else:
            response = self._parse_output(raw_response, response_format)

        self.last_log = [prompt, messages, raw_response]
        return response

    def inspect_history(self) -> None:
        if not hasattr(self, "last_log"):
            raise ValueError("No history available. Please call the model first.")

        prompt, messages, response = self.last_log
        # Ensure response is a dictionary
        if hasattr(response, "model_dump"):
            response = response.model_dump()

        if not messages:
            messages = [{"role": "user", "content": prompt}]

        print("\n\n")
        print(_blue("[Conversation History]") + "\n")

        # Print all messages in the conversation
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            print(_red(f"{role.capitalize()}:"))

            if isinstance(content, str):
                print(content.strip())
            elif isinstance(content, list):
                # Handle multimodal content
                for item in content:
                    if item.get("type") == "text":
                        print(item["text"].strip())
                    elif item.get("type") == "image_url":
                        image_url = item["image_url"]["url"]
                        if "base64" in image_url:
                            len_base64 = len(image_url.split("base64,")[1])
                            print(_blue(f"<IMAGE BASE64 ENCODED({len_base64})>"))
                        else:
                            print(_blue(f"<image_url: {image_url}>"))
            print("\n")

        # Print the response - now always an OpenAI completion
        print(_red("Response:"))

        # Handle OpenAI response object
        if isinstance(response, dict) and "choices" in response and response["choices"]:
            message = response["choices"][0].get("message", {})

            # Check for reasoning content (if available)
            reasoning = message.get("reasoning_content")

            # Check for parsed content (structured mode)
            parsed = message.get("parsed")

            # Get regular content
            content = message.get("content")

            # Display reasoning if available
            if reasoning:
                print(_yellow("<think>"))
                print(reasoning.strip())
                print(_yellow("</think>"))
                print()

            # Display parsed content for structured responses
            if parsed:
                # print(_green('<Parsed Structure>'))
                if hasattr(parsed, "model_dump"):
                    print(jdumps(parsed.model_dump(), indent=2))
                else:
                    print(jdumps(parsed, indent=2))
                # print(_green('</Parsed Structure>'))
                print()

            else:
                if content:
                    # print(_green("<Content>"))
                    print(content.strip())
                    # print(_green("</Content>"))
                else:
                    print(_green("[No content]"))

            # Show if there were multiple completions
            if len(response["choices"]) > 1:
                print(
                    _blue(f"\n(Plus {len(response['choices']) - 1} other completions)")
                )
        else:
            # Fallback for non-standard response objects or cached responses
            print(_yellow("Warning: Not a standard OpenAI response object"))
            if isinstance(response, str):
                print(_green(response.strip()))
            elif isinstance(response, dict):
                print(_green(json.dumps(response, indent=2)))
            else:
                print(_green(str(response)))

        # print("\n\n")

    # --------------------------------------------------------------------- #
    # low-level OpenAI call
    # --------------------------------------------------------------------- #
    def _call_raw(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        response_format: Union[type[str], Type[BaseModel]],
        use_cache: bool,
        **kw: Any,
    ):
        assert self.model is not None, "Model must be set before making a call."
        model: str = self.model

        cache_key = (
            self._cache_key(messages, kw, response_format) if use_cache else None
        )
        if cache_key and (hit := self._load_cache(cache_key)) is not None:
            return hit

        try:
            # structured mode
            if response_format is not str and issubclass(response_format, BaseModel):
                openai_response = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=list(messages),
                    response_format=response_format,  # type: ignore[arg-type]
                    **kw,
                )
            # plain-text mode
            else:
                openai_response = self.client.chat.completions.create(
                    model=model,
                    messages=list(messages),
                    **kw,
                )

        except (AuthenticationError, RateLimitError) as exc:  # pragma: no cover
            logger.error(exc)
            raise

        if cache_key:
            self._dump_cache(cache_key, openai_response)

        return openai_response

    # --------------------------------------------------------------------- #
    # legacy → typed messages
    # --------------------------------------------------------------------- #
    @staticmethod
    def _convert_messages(msgs: LegacyMsgs) -> Messages:
        converted: Messages = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                converted.append(
                    ChatCompletionUserMessageParam(role="user", content=content)
                )
            elif role == "assistant":
                converted.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant", content=content
                    )
                )
            elif role == "system":
                converted.append(
                    ChatCompletionSystemMessageParam(role="system", content=content)
                )
            elif role == "tool":
                converted.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        content=content,
                        tool_call_id=msg.get("tool_call_id") or "",  # str, never None
                    )
                )
            else:
                # fall back to raw dict for unknown roles
                converted.append({"role": role, "content": content})  # type: ignore[arg-type]
        return converted

    # --------------------------------------------------------------------- #
    # final parse (needed for plain-text or cache hits only)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _parse_output(
        raw_response: Any,
        response_format: Union[type[str], Type[BaseModel]],
    ) -> str | BaseModel:
        # Convert any object to dict if needed
        if hasattr(raw_response, "model_dump"):
            raw_response = raw_response.model_dump()

        if response_format is str:
            # Extract the content from OpenAI response dict
            if isinstance(raw_response, dict) and "choices" in raw_response:
                message = raw_response["choices"][0]["message"]
                return message.get("content", "") or ""
            return cast(str, raw_response)

        # For the type-checker: we *know* it's a BaseModel subclass here.
        model_cls = cast(Type[BaseModel], response_format)

        # Handle structured response
        if isinstance(raw_response, dict) and "choices" in raw_response:
            message = raw_response["choices"][0]["message"]

            # Check if already parsed by OpenAI client
            if "parsed" in message:
                return model_cls.model_validate(message["parsed"])

            # Need to parse the content
            content = message.get("content")
            if content is None:
                raise ValueError("Model returned empty content")

            try:
                data = json.loads(content)
                return model_cls.model_validate(data)
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse model output as JSON:\n{content}"
                ) from exc

        # Handle cached response or other formats
        if isinstance(raw_response, model_cls):
            return raw_response
        if isinstance(raw_response, dict):
            return model_cls.model_validate(raw_response)

        # Try parsing as JSON string
        try:
            data = json.loads(raw_response)
            return model_cls.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"Model did not return valid JSON:\n---\n{raw_response}"
            ) from exc

    # --------------------------------------------------------------------- #
    # tiny disk cache
    # --------------------------------------------------------------------- #
    @staticmethod
    def _cache_key(
        messages: Any,
        kw: Any,
        response_format: Union[type[str], Type[BaseModel]],
    ) -> str:
        tag = response_format.__name__ if response_format is not str else "text"
        blob = json.dumps([messages, kw, tag], sort_keys=True).encode()
        return base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).decode()[:22]

    @staticmethod
    def _cache_path(key: str) -> str:
        return os.path.expanduser(f"~/.cache/lm/{key}.json")

    def _dump_cache(self, key: str, val: Any) -> None:
        try:
            path = self._cache_path(key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                if isinstance(val, BaseModel):
                    json.dump(val.model_dump(mode="json"), fh)
                else:
                    json.dump(val, fh)
        except Exception as exc:  # pragma: no cover
            logger.debug(f"cache write skipped: {exc}")

    def _load_cache(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:  # pragma: no cover
            return None

    @staticmethod
    def list_models(
        port=None, host="localhost", base_url: Optional[str] = None
    ) -> List[str]:
        """List available models from OpenAI-compatible API server."""
        try:
            client: OpenAI = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY", "abc"),
                base_url=f"http://{host}:{port}/v1" if port else base_url or None,
            )
            models: SyncPage[Model] = client.models.list()
            return [model.id for model in models.data]
        except Exception as exc:
            endpoint = f"http://{host}:{port}/v1" if port else base_url
            error_msg = str(exc)

            if "404" in error_msg or "Not Found" in error_msg:
                raise ValueError(
                    f"No OpenAI-compatible API found at {endpoint}. "
                    f"The endpoint appears to be running a different service "
                    f"(possibly Jupyter Server). Please check the port number."
                ) from exc
            elif "Connection" in error_msg:
                raise ValueError(
                    f"Cannot connect to {endpoint}. "
                    f"Please verify the service is running and accessible."
                ) from exc
            else:
                raise ValueError(
                    f"Failed to list models from {endpoint}: {error_msg}"
                ) from exc

    def parse(
        self,
        response_model: Type[BaseModel],
        instruction: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[RawMsgs] = None,
        think: Literal[True, False, None] = None,
        add_json_schema_to_instruction: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_openai_response: bool = False,
        cache: Optional[bool] = True,
        **kwargs,
    ):
        if messages is None:
            assert instruction is not None, "Instruction must be provided."
            assert prompt is not None, "Prompt must be provided."
            messages = [
                {
                    "role": "system",
                    "content": instruction,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]  # type: ignore

        post_fix = ""
        json_schema = response_model.model_json_schema()
        if add_json_schema_to_instruction and response_model:
            _schema = f"\n\n<output_json_schema>\n{json.dumps(json_schema, indent=2)}\n</output_json_schema>"
            post_fix += _schema

        if think:
            post_fix += "\n\n/think"
        elif not think:
            post_fix += "\n\n/no_think"

        assert isinstance(messages, list), "Messages must be a list."
        assert len(messages) > 0, "Messages cannot be empty."
        assert messages[0]["role"] == "system", (
            "First message must be a system message with instruction."
        )
        messages[0]["content"] += post_fix  # type: ignore

        model_kwargs = {}
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        model_kwargs.update(kwargs)

        use_cache = self.do_cache if cache is None else cache
        cache_key = None
        if use_cache:
            cache_data = {
                "messages": messages,
                "model_kwargs": model_kwargs,
                "guided_json": json_schema,
                "response_format": response_model.__name__,
            }
            cache_key = self._cache_key(cache_data, {}, response_model)
            cached_response = self._load_cache(cache_key)
            self.last_log = [prompt, messages, cached_response]
            if cached_response is not None:
                if return_openai_response:
                    return cached_response
                return self._parse_complete_output(cached_response, response_model)

        completion = self.client.chat.completions.create(
            model=self.model,  # type: ignore
            messages=messages,  # type: ignore
            extra_body={"guided_json": json_schema},
            **model_kwargs,
        )

        if cache_key:
            self._dump_cache(cache_key, completion)

        self.last_log = [prompt, messages, completion]
        if return_openai_response:
            return completion
        return self._parse_complete_output(completion, response_model)

    def _parse_complete_output(
        self, completion: Any, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Parse completion output to response model."""
        if hasattr(completion, "model_dump"):
            completion = completion.model_dump()

        if "choices" not in completion or not completion["choices"]:
            raise ValueError("No choices in OpenAI response")

        content = completion["choices"][0]["message"]["content"]
        if not content:
            raise ValueError("Empty content in response")

        try:
            data = json.loads(content)
            return response_model.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse response as {response_model.__name__}: {content}"
            ) from exc

    def inspect_word_probs(
        self,
        messages: Optional[List[Dict[str, Any]]] = None,
        tokenizer: Optional[Any] = None,
        do_print=True,
        add_think: bool = True,
    ) -> tuple[List[Dict[str, Any]], Any, str]:
        """
        Inspect word probabilities in a language model response.

        Args:
            tokenizer: Tokenizer instance to encode words.
            messages: List of messages to analyze.

        Returns:
            A tuple containing:
            - List of word probabilities with their log probabilities.
            - Token log probability dictionaries.
            - Rendered string with colored word probabilities.
        """
        if messages is None:
            messages = self.last_messages(add_think=add_think)
            if messages is None:
                raise ValueError("No messages provided and no last messages available.")

        if tokenizer is None:
            tokenizer = get_tokenizer(self.model)

        ret = inspect_word_probs(self, tokenizer, messages)
        if do_print:
            print(ret[-1])
        return ret

    def last_messages(self, add_think: bool = True) -> Optional[List[Dict[str, str]]]:
        last_conv = self.last_log
        messages = last_conv[1] if len(last_conv) > 1 else None
        last_msg = last_conv[2]
        if not isinstance(last_msg, dict):
            last_conv[2] = last_conv[2].model_dump()  # type: ignore
        msg = last_conv[2]
        # Ensure msg is a dict
        if hasattr(msg, "model_dump"):
            msg = msg.model_dump()
        message = msg["choices"][0]["message"]
        reasoning = message.get("reasoning_content")
        answer = message.get("content")
        if reasoning and add_think:
            final_answer = f"<think>{reasoning}</think>\n{answer}"
        else:
            final_answer = f"<think>\n\n</think>\n{answer}"
        assistant = {"role": "assistant", "content": final_answer}
        messages = messages + [assistant]  # type: ignore
        return messages if messages else None


@lru_cache(maxsize=10)
def get_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer


def inspect_word_probs(lm, tokenizer, messages):
    import numpy as np

    def compute_word_log_probs(
        tokenizer: Any,
        lm_client: Any,
    ) -> tuple[List[Dict[str, Any]], Any]:
        # Build a prompt that preserves literal newlines
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,  # Don't tokenize yet, we need raw text
            add_generation_prompt=False,  # No generation prompt needed
        )

        # Request token logprobs
        response = lm_client.client.completions.create(
            model=lm_client.model,  # type: ignore
            prompt=prompt,
            max_tokens=1,
            logprobs=1,
            extra_body={"prompt_logprobs": 0},
        )
        token_logprob_dicts = response.choices[0].prompt_logprobs  # type: ignore

        # Override first token to known start marker
        start_id = tokenizer.encode("<|im_start|>")[0]
        token_logprob_dicts[0] = {
            str(start_id): {
                "logprob": -1,
                "rank": 1,
                "decoded_token": "<|im_start|>",
            }
        }

        # Flatten tokens
        tokens: List[Dict[str, Any]] = [
            {"id": int(tid), **tdata}
            for td in token_logprob_dicts
            for tid, tdata in td.items()
        ]

        # Validate tokenization
        tokenized = tokenizer.tokenize(prompt)
        if len(tokenized) != len(tokens):
            raise ValueError(f"Token count mismatch: {len(tokenized)} vs {len(tokens)}")
        for idx, tok in enumerate(tokens):
            if tokenized[idx] != tok["decoded_token"]:
                raise AssertionError(
                    f"Token mismatch at {idx}: "
                    f"{tokenized[idx]} != {tok['decoded_token']}"
                )

        # Split on newline sentinel
        split_prompt = prompt.replace("\n", " <NL> ")
        words = split_prompt.split()

        word_log_probs: List[Dict[str, Any]] = []
        token_idx = 0

        for word in words:
            # Map sentinel back to actual newline for encoding
            target = "\n" if word == "<NL>" else word
            sub_ids = tokenizer.encode(target, add_special_tokens=False)
            count = len(sub_ids)
            if count == 0:
                continue

            subs = tokens[token_idx : token_idx + count]
            avg_logprob = sum(s["logprob"] for s in subs) / count
            prob = float(np.exp(avg_logprob))
            word_log_probs.append({"word": target, "probability": prob})
            token_idx += count

        return word_log_probs, token_logprob_dicts  # type: ignore

    def render_by_logprob(word_log_probs: List[Dict[str, Any]]) -> str:
        """
        Return an ANSI-colored string for word probabilities (red → green).
        """
        if not word_log_probs:
            return ""

        probs = [entry["probability"] for entry in word_log_probs]
        min_p, max_p = min(probs), max(probs)
        parts: List[str] = []

        for entry in word_log_probs:
            word = entry["word"]
            # Preserve actual line breaks
            if word == "\n":
                parts.append("\n")
                continue

            p = entry["probability"]
            norm = (p - min_p) / (max_p - min_p or 1.0)
            r = int(255 * (1 - norm))  # red component (high when prob is low)
            g = int(255 * norm)  # green component (high when prob is high)
            b = 0  # no blue for red-green gradient
            colored = f"\x1b[38;2;{r};{g};{b}m{word}\x1b[0m"
            parts.append(colored + " ")

        return "".join(parts).rstrip()

    word_probs, token_logprob_dicts = compute_word_log_probs(tokenizer, lm)
    return word_probs, token_logprob_dicts, render_by_logprob(word_probs)


class LLMTask(ABC):
    """
    Callable wrapper around an LM endpoint.

    Sub-classes must set:
      • lm              – the language-model instance
      • InputModel      – a Pydantic input class
      • OutputModel     – a Pydantic output class

    Optional flags:
      • temperature     – float (default 0.6)
      • think           – bool  (if the backend supports “chain-of-thought”)
      • add_json_schema – bool  (include schema in the instruction)

    The **docstring** of each sub-class is sent as the LM instruction.
    Example
    ```python
        class DemoTask(LLMTask):
            "TODO: SYSTEM_PROMPT_INSTURCTION HERE"

            lm = LM(port=8130, cache=False, model="gpt-3.5-turbo")

            class InputModel(BaseModel):
                text_to_translate:str

            class OutputModel(BaseModel):
                translation:str
                glossary_use:str

            temperature = 0.6
            think=False

        demo_task = DemoTask()
        demo_task({'text_to_translate': 'Translate from english to vietnamese: Hello how are you'})
    ```
    """

    lm: "LM"
    InputModel: Type[BaseModel]
    OutputModel: Type[BaseModel]

    temperature: float = 0.6
    think: bool = False
    add_json_schema: bool = False

    def __call__(self, data: BaseModel | dict) -> BaseModel:
        if (
            not hasattr(self, "InputModel")
            or not hasattr(self, "OutputModel")
            or not hasattr(self, "lm")
        ):
            raise NotImplementedError(
                f"{self.__class__.__name__} must define lm, InputModel, and OutputModel as class attributes."
            )

        item = data if isinstance(data, BaseModel) else self.InputModel(**data)

        return self.lm.parse(
            prompt=item.model_dump_json(),
            instruction=self.__doc__ or "",
            response_model=self.OutputModel,
            temperature=self.temperature,
            think=self.think,
            add_json_schema_to_instruction=self.add_json_schema,
        )

    def generate_training_data(
        self, input_dict: Dict[str, Any], output: Dict[str, Any]
    ):
        "Return share gpt like format"
        system_prompt = self.__doc__ or ""
        user_msg = self.InputModel(**input_dict).model_dump_json()  # type: ignore[attr-defined]
        assistant_msg = self.OutputModel(**output).model_dump_json()  # type: ignore[attr-defined]
        return get_conversation_one_turn(
            system_msg=system_prompt, user_msg=user_msg, assistant_msg=assistant_msg
        )

    run = __call__  # alias for compatibility with other LLMTask implementations