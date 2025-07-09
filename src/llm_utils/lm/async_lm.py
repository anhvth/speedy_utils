"""
# ============================================================================= #
# ASYNCHRONOUS LANGUAGE MODEL WRAPPER WITH CONCURRENT EXECUTION SUPPORT
# ============================================================================= #
#
# Title & Intent:
# High-performance asynchronous language model interface for concurrent LLM operations
#
# High-level Summary:
# This module provides an async drop-in replacement for the synchronous LM class, designed
# for high-throughput applications requiring concurrent language model operations. It maintains
# full API compatibility while adding async/await semantics, connection pooling, and efficient
# resource management. The AsyncLM class supports batch processing, concurrent request handling,
# and maintains the same caching and type safety guarantees as the synchronous version.
#
# Public API / Data Contracts:
# • AsyncLM(model, temperature=0.0, max_tokens=2000, host="localhost", port=None, **kwargs) - Async wrapper class
# • async AsyncLM.__call__(prompt=None, messages=None, response_format=str, cache=None, **kwargs) -> str | BaseModel
# • async AsyncLM.list_models(port=None) -> List[str] - Enumerate available models
# • async AsyncLM.count_tokens(messages, model=None) -> int - Token counting utility
# • async AsyncLM.price(messages, model=None, response_tokens=0) -> float - Cost estimation
# • AsyncLM.set_model(model_name) -> None - Runtime model switching (sync method)
# • async AsyncLM.batch_call(requests) -> List[Union[str, BaseModel]] - Concurrent batch processing
# • TModel = TypeVar("TModel", bound=BaseModel) - Generic type for structured responses
# • Messages = List[ChatCompletionMessageParam] - Typed message format
#
# Invariants / Constraints:
# • MUST be used within async context (asyncio event loop required)
# • MUST provide either 'prompt' or 'messages' parameter, but not both
# • MUST properly await all async method calls
# • Connection pooling MUST handle concurrent requests efficiently
# • MUST maintain thread safety across concurrent operations
# • Rate limit handling MUST use async backoff without blocking event loop
# • MUST preserve all synchronous LM class behaviors and constraints
# • Resource cleanup MUST occur on context manager exit or explicit close
#
# Usage Example:
# ```python
# import asyncio
# from llm_utils.lm.async_lm import AsyncLM
# from pydantic import BaseModel
#
# class SummaryResponse(BaseModel):
#     summary: str
#     key_points: List[str]
#     confidence: float
#
# async def main():
#     # Single async call
#     lm = AsyncLM(model="gpt-4o-mini", temperature=0.1)
#     response = await lm(prompt="Summarize quantum computing")
#     print(response)
#
#     # Concurrent batch processing
#     texts = ["Text 1 to summarize", "Text 2 to summarize", "Text 3 to summarize"]
#     tasks = [lm(prompt=f"Summarize: {text}", response_format=SummaryResponse) for text in texts]
#     summaries = await asyncio.gather(*tasks)
#
#     for summary in summaries:
#         print(f"Summary: {summary.summary}")
#         print(f"Key points: {summary.key_points}")
#
# asyncio.run(main())
# ```
#
# TODO & Future Work:
# • Add async context manager support for automatic resource cleanup
# • Implement connection pool size optimization based on usage patterns
# • Add async streaming response support with async generators
# • Optimize memory usage for large-scale concurrent operations
# • Add async rate limiting with priority queuing
#
# ============================================================================= #
"""

import base64
import hashlib
import json
import os
from abc import ABC
from functools import cache, lru_cache
from typing import (
    Any,
    Dict,
    Generic,
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
from typing_extensions import TypedDict
from httpx import URL
from loguru import logger
from numpy import isin
from openai import AsyncOpenAI, AuthenticationError, BadRequestError, RateLimitError
from openai.pagination import AsyncPage as AsyncSyncPage

# from openai.pagination import AsyncSyncPage
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

# --------------------------------------------------------------------------- #
# type helpers
# --------------------------------------------------------------------------- #
TModel = TypeVar("TModel", bound=BaseModel)
Messages = List[ChatCompletionMessageParam]
LegacyMsgs = List[Dict[str, str]]
RawMsgs = Union[Messages, LegacyMsgs]

# --------------------------------------------------------------------------- #
# color helpers (unchanged)
# --------------------------------------------------------------------------- #


def _color(code: int, text: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m"


def _red(t):
    return _color(31, t)


def _green(t):
    return _color(32, t)


def _blue(t):
    return _color(34, t)


def _yellow(t):
    return _color(33, t)



TParsed = TypeVar('TParsed', bound=BaseModel)

class ParsedOutput(TypedDict, Generic[TParsed]):
    messages: List
    completion: Any
    parsed: TParsed


class AsyncLM:
    """Unified **async** language‑model wrapper with optional JSON parsing."""

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
        ports: Optional[List[int]] = None,
        **openai_kwargs: Any,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.port = port
        self.host = host
        self.base_url = base_url or (f"http://{host}:{port}/v1" if port else None)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "abc")
        self.openai_kwargs = openai_kwargs
        self.do_cache = cache
        self.ports = ports
        self._init_port = port  # <-- store the port provided at init

        # Async client

    @property
    def client(self) -> AsyncOpenAI:
        # if have multiple ports
        if self.ports:
            import random

            port = random.choice(self.ports)
            api_base = f"http://{self.host}:{port}/v1"
            logger.debug(f"Using port: {port}")
        else:
            api_base = self.base_url or f"http://{self.host}:{self.port}/v1"
        client = AsyncOpenAI(
            api_key=self.api_key, base_url=api_base, **self.openai_kwargs
        )
        return client

    # ------------------------------------------------------------------ #
    # Public API – typed overloads
    # ------------------------------------------------------------------ #
    @overload
    async def __call__(
        self,
        *,
        prompt: str | None = ...,
        messages: RawMsgs | None = ...,
        response_format: type[str] = str,
        return_openai_response: bool = ...,
        **kwargs: Any,
    ) -> str: ...

    @overload
    async def __call__(
        self,
        *,
        prompt: str | None = ...,
        messages: RawMsgs | None = ...,
        response_format: Type[TModel],
        return_openai_response: bool = ...,
        **kwargs: Any,
    ) -> TModel: ...

    async def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[RawMsgs] = None,
        response_format: Union[type[str], Type[BaseModel]] = str,
        cache: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        return_openai_response: bool = False,
        **kwargs: Any,
    ):
        if (prompt is None) == (messages is None):
            raise ValueError("Provide *either* `prompt` or `messages` (but not both).")

        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        assert messages is not None
        # assert self.model is not None, "Model must be set before calling."
        if not self.model:
            models = await self.list_models(port=self.port, host=self.host)
            self.model = models[0] if models else None
            logger.info(
                f"No model specified. Using the first available model. {self.model}"
            )
        openai_msgs: Messages = (
            self._convert_messages(cast(LegacyMsgs, messages))
            if isinstance(messages[0], dict)
            else cast(Messages, messages)
        )

        kw = dict(
            self.openai_kwargs,
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        kw.update(kwargs)
        use_cache = self.do_cache if cache is None else cache

        raw_response = await self._call_raw(
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

    # ------------------------------------------------------------------ #
    # Model invocation (async)
    # ------------------------------------------------------------------ #
    async def _call_raw(
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
            # Check if cached value is an error
            if isinstance(hit, dict) and hit.get("error"):
                error_type = hit.get("error_type", "Unknown")
                error_msg = hit.get("error_message", "Cached error")
                logger.warning(f"Found cached error ({error_type}): {error_msg}")
                # Re-raise as a ValueError with meaningful message
                raise ValueError(f"Cached {error_type}: {error_msg}")
            return hit

        try:
            if response_format is not str and issubclass(response_format, BaseModel):
                openai_response = await self.client.beta.chat.completions.parse(
                    model=model,
                    messages=list(messages),
                    response_format=response_format,  # type: ignore[arg-type]
                    **kw,
                )
            else:
                openai_response = await self.client.chat.completions.create(
                    model=model,
                    messages=list(messages),
                    **kw,
                )

        except (AuthenticationError, RateLimitError, BadRequestError) as exc:
            error_msg = f"OpenAI API error ({type(exc).__name__}): {exc}"
            logger.error(error_msg)

            # Cache the error if it's a BadRequestError to avoid repeated calls
            if isinstance(exc, BadRequestError) and cache_key:
                error_response = {
                    "error": True,
                    "error_type": "BadRequestError",
                    "error_message": str(exc),
                    "choices": [],
                }
                self._dump_cache(cache_key, error_response)
                logger.debug(f"Cached BadRequestError for key: {cache_key}")

            raise

        if cache_key:
            self._dump_cache(cache_key, openai_response)

        return openai_response

    # ------------------------------------------------------------------ #
    # Utilities below are unchanged (sync I/O is acceptable)
    # ------------------------------------------------------------------ #
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
                        tool_call_id=msg.get("tool_call_id") or "",
                    )
                )
            else:
                converted.append({"role": role, "content": content})  # type: ignore[arg-type]
        return converted

    @staticmethod
    def _parse_output(
        raw_response: Any, response_format: Union[type[str], Type[BaseModel]]
    ) -> str | BaseModel:
        if hasattr(raw_response, "model_dump"):
            raw_response = raw_response.model_dump()

        if response_format is str:
            if isinstance(raw_response, dict) and "choices" in raw_response:
                message = raw_response["choices"][0]["message"]
                return message.get("content", "") or ""
            return cast(str, raw_response)

        model_cls = cast(Type[BaseModel], response_format)

        if isinstance(raw_response, dict) and "choices" in raw_response:
            message = raw_response["choices"][0]["message"]
            if "parsed" in message:
                return model_cls.model_validate(message["parsed"])
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

        if isinstance(raw_response, model_cls):
            return raw_response
        if isinstance(raw_response, dict):
            return model_cls.model_validate(raw_response)

        try:
            data = json.loads(raw_response)
            return model_cls.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"Model did not return valid JSON:\n---\n{raw_response}"
            ) from exc

    # ------------------------------------------------------------------ #
    # Simple disk cache (sync)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _cache_key(
        messages: Any, kw: Any, response_format: Union[type[str], Type[BaseModel]]
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
        except Exception as exc:
            logger.debug(f"cache write skipped: {exc}")

    def _load_cache(self, key: str) -> Any | None:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Missing methods from LM class
    # ------------------------------------------------------------------ #
    async def parse(
        self,
        response_model: Type[TParsed],
        instruction: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[RawMsgs] = None,
        think: Literal[True, False, None] = None,
        add_json_schema_to_instruction: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        cache: Optional[bool] = True,
        **kwargs,
    ) -> ParsedOutput[TParsed]:
        """Parse response using guided JSON generation."""
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
        completion = None
        if use_cache:
            cache_data = {
                "messages": messages,
                "model_kwargs": model_kwargs,
                "guided_json": json_schema,
                "response_format": response_model.__name__,
            }
            cache_key = self._cache_key(cache_data, {}, response_model)
            completion = self._load_cache(cache_key)  # dict
        if not completion:
            completion = await self.client.chat.completions.create(
                model=self.model,  # type: ignore
                messages=messages,  # type: ignore
                extra_body={"guided_json": json_schema},
                **model_kwargs,
            )
            completion = completion.model_dump()
            if cache_key:
                self._dump_cache(cache_key, completion)
        assert isinstance(completion, dict), (
            "Completion must be a dictionary with OpenAI response format."
        )
        self.last_log = [prompt, messages, completion]

        output = cast(TParsed, self._parse_complete_output(completion, response_model))
        full_messages = messages + [completion]
        return ParsedOutput(
            messages=full_messages,
            completion=completion,
            parsed=output,
        )

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

    async def inspect_word_probs(
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
            messages = await self.last_messages(add_think=add_think)
            if messages is None:
                raise ValueError("No messages provided and no last messages available.")

        if tokenizer is None:
            tokenizer = get_tokenizer(self.model)

        ret = await inspect_word_probs_async(self, tokenizer, messages)
        if do_print:
            print(ret[-1])
        return ret

    async def last_messages(
        self, add_think: bool = True
    ) -> Optional[List[Dict[str, str]]]:
        """Get the last conversation messages including assistant response."""
        if not hasattr(self, "last_log"):
            return None

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

    # ------------------------------------------------------------------ #
    # Utility helpers
    # ------------------------------------------------------------------ #
    async def inspect_history(self) -> None:
        """Inspect the conversation history with proper formatting."""
        if not hasattr(self, "last_log"):
            raise ValueError("No history available. Please call the model first.")

        prompt, messages, response = self.last_log
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        if not messages:
            messages = [{"role": "user", "content": prompt}]

        print("\n\n")
        print(_blue("[Conversation History]") + "\n")

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            print(_red(f"{role.capitalize()}:"))
            if isinstance(content, str):
                print(content.strip())
            elif isinstance(content, list):
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

        print(_red("Response:"))
        if isinstance(response, dict) and response.get("choices"):
            message = response["choices"][0].get("message", {})
            reasoning = message.get("reasoning_content")
            parsed = message.get("parsed")
            content = message.get("content")
            if reasoning:
                print(_yellow("<think>"))
                print(reasoning.strip())
                print(_yellow("</think>\n"))
            if parsed:
                print(
                    json.dumps(
                        (
                            parsed.model_dump()
                            if hasattr(parsed, "model_dump")
                            else parsed
                        ),
                        indent=2,
                    )
                    + "\n"
                )
            elif content:
                print(content.strip())
            else:
                print(_green("[No content]"))
            if len(response["choices"]) > 1:
                print(
                    _blue(f"\n(Plus {len(response['choices']) - 1} other completions)")
                )
        else:
            print(_yellow("Warning: Not a standard OpenAI response object"))
            if isinstance(response, str):
                print(_green(response.strip()))
            elif isinstance(response, dict):
                print(_green(json.dumps(response, indent=2)))
            else:
                print(_green(str(response)))

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #
    def set_model(self, model: str) -> None:
        self.model = model

    @staticmethod
    async def list_models(port=None, host="localhost") -> List[str]:
        try:
            client: AsyncOpenAI = AsyncLM(port=port, host=host).client  # type: ignore[arg-type]
            base_url: URL = client.base_url
            logger.debug(f"Base URL: {base_url}")
            models: AsyncSyncPage[Model] = await client.models.list()  # type: ignore[assignment]
            return [model.id for model in models.data]
        except Exception as exc:
            logger.error(f"Failed to list models: {exc}")
            return []


# --------------------------------------------------------------------------- #
# Module-level utility functions (async versions)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=10)
def get_tokenizer(model_name: str) -> Any:
    """Get tokenizer for the given model."""
    from transformers import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return tokenizer


async def inspect_word_probs_async(lm, tokenizer, messages):
    """Async version of inspect_word_probs."""

    import numpy as np

    async def compute_word_log_probs(
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
        response = await lm_client.client.completions.create(
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

    word_probs, token_logprob_dicts = await compute_word_log_probs(tokenizer, lm)
    return word_probs, token_logprob_dicts, render_by_logprob(word_probs)


# --------------------------------------------------------------------------- #
# Async LLMTask class
# --------------------------------------------------------------------------- #

InputModelType = TypeVar("InputModelType", bound=BaseModel)
OutputModelType = TypeVar("OutputModelType", bound=BaseModel)


class AsyncLLMTask(ABC, Generic[InputModelType, OutputModelType]):
    """
    Async callable wrapper around an AsyncLM endpoint.

    Sub-classes must set:
      • lm              – the async language-model instance
      • InputModel      – a Pydantic input class
      • OutputModel     – a Pydantic output class

    Optional flags:
      • temperature     – float (default 0.6)
      • think           – bool  (if the backend supports "chain-of-thought")
      • add_json_schema – bool  (include schema in the instruction)

    The **docstring** of each sub-class is sent as the LM instruction.
    Example
    ```python
        class DemoTask(AsyncLLMTask):
            "TODO: SYSTEM_PROMPT_INSTURCTION HERE"

            lm = AsyncLM(port=8130, cache=False, model="gpt-3.5-turbo")

            class InputModel(BaseModel):
                text_to_translate:str

            class OutputModel(BaseModel):
                translation:str
                glossary_use:str

            temperature = 0.6
            think=False

        demo_task = DemoTask()
        result = await demo_task({'text_to_translate': 'Translate from english to vietnamese: Hello how are you'})
    ```
    """

    lm: "AsyncLM"
    InputModel: InputModelType
    OutputModel: OutputModelType

    temperature: float = 0.6
    think: bool = False
    add_json_schema: bool = False
    cache: bool = False

    async def __call__(
        self,
        data: BaseModel | dict,
        temperature: float = 0.1,
        cache: bool = False,
    ) -> tuple[OutputModelType, List[Dict[str, Any]]]:
        # Get the input and output model types from the generic parameters
        type_args = getattr(self.__class__, "__orig_bases__", None)
        if (
            type_args
            and hasattr(type_args[0], "__args__")
            and len(type_args[0].__args__) >= 2
        ):
            input_model = type_args[0].__args__[0]
            output_model = type_args[0].__args__[1]
        else:
            # Fallback to the old way if type introspection fails
            if (
                not hasattr(self, "InputModel")
                or not hasattr(self, "OutputModel")
                or not hasattr(self, "lm")
            ):
                raise NotImplementedError(
                    f"{self.__class__.__name__} must define lm, InputModel, and OutputModel as class attributes or use proper generic typing."
                )
            input_model = self.InputModel
            output_model = self.OutputModel

        # Ensure input_model is a class before calling
        if isinstance(data, BaseModel):
            item = data
        elif isinstance(input_model, type) and issubclass(input_model, BaseModel):
            item = input_model(**data)
        else:
            raise TypeError("InputModel must be a subclass of BaseModel")

        assert isinstance(output_model, type) and issubclass(output_model, BaseModel), (
            "OutputModel must be a subclass of BaseModel"
        )

        result = await self.lm.parse(
            prompt=item.model_dump_json(),
            instruction=self.__doc__ or "",
            response_model=output_model,
            temperature=temperature or self.temperature,
            think=self.think,
            add_json_schema_to_instruction=self.add_json_schema,
            cache=self.cache or cache,
        )

        return (
            cast(OutputModelType, result["parsed"]),  # type: ignore
            cast(List[dict], result["messages"]),  # type: ignore
        )

    def generate_training_data(
        self, input_dict: Dict[str, Any], output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return share gpt like format"""
        system_prompt = self.__doc__ or ""
        user_msg = self.InputModel(**input_dict).model_dump_json()  # type: ignore[attr-defined]
        assistant_msg = self.OutputModel(**output).model_dump_json()  # type: ignore[attr-defined]
        messages = get_conversation_one_turn(
            system_msg=system_prompt, user_msg=user_msg, assistant_msg=assistant_msg
        )
        return {"messages": messages}

    arun = __call__  # alias for compatibility with other LLMTask implementations
