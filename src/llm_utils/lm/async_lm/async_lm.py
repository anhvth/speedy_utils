# from ._utils import *
import json
import os
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    overload,
)

from click import Option
from loguru import logger
from openai import AsyncOpenAI, AuthenticationError, BadRequestError, RateLimitError
from pydantic import BaseModel
from traitlets import Bool

# from llm_utils.lm.async_lm.async_llm_task import OutputModelType
from llm_utils.lm.async_lm.async_lm_base import AsyncLMBase
from speedy_utils import jloads

from ._utils import (
    LegacyMsgs,
    Messages,
    OutputModelType,
    ParsedOutput,
    RawMsgs,
    TModel,
    get_tokenizer,
)


class AsyncLM(AsyncLMBase):
    """Unified **async** language‑model wrapper with optional JSON parsing."""

    def __init__(
        self,
        model: str,
        *,
        response_model: Optional[type[BaseModel]] = None,
        temperature: float = 0.0,
        max_tokens: int = 2_000,
        host: str = "localhost",
        port: Optional[int | str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: bool = True,
        think: Literal[True, False, None] = None,
        add_json_schema_to_instruction: Optional[bool] = None,
        use_beta: bool = False,
        ports: Optional[List[int]] = None,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.port = port
        self.host = host
        self.base_url = base_url or (f"http://{host}:{port}/v1" if port else None)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "abc")
        # self.openai_kwargs = openai_kwargs
        self.do_cache = cache
        self.ports = ports
        self._init_port = port  # <-- store the port provided at init

        # Model behavior options
        self.response_model = response_model
        self.think = think
        self.use_beta = use_beta
        self.add_json_schema_to_instruction = add_json_schema_to_instruction
        if not use_beta:
            self.add_json_schema_to_instruction = True

        # Set default parameter values explicitly
        self.top_p = 1.0
        self.presence_penalty = 0.0
        self.top_k = 1
        self.repetition_penalty = 1.0

        self.model_kwargs = dict(
            model=model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
        )
        self.extra_body = dict(
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )

    async def _call_and_parse(
        self,
        messages: list[dict],
        response_model: Type[OutputModelType],
        json_schema: dict,
        use_beta: bool,
        cache: Optional[bool] = None,
    ) -> tuple[dict, dict, OutputModelType]:
        """Unified call and parse with cache and error handling."""
        converted_messages = self._convert_messages(messages)
        use_cache = self.do_cache if cache is None else cache
        cache_key = None
        completion = None
        choice = None
        parsed = None

        if use_cache:
            cache_data = {
                "messages": converted_messages,
                "model_kwargs": self.model_kwargs,
                "guided_json": json_schema,
                "response_format": response_model.__name__,
                "use_beta": use_beta,
            }
            cache_key = self._cache_key(cache_data, {}, response_model)
            completion = self._load_cache(cache_key)

        if not completion:
            try:
                if use_beta:
                    completion = await self.client.chat.completions.create(
                        messages=converted_messages,
                        extra_body={"guided_json": json_schema, **self.extra_body},
                        **self.model_kwargs,  # type: ignore
                    )  # type: ignore
                else:
                    completion = await self.client.chat.completions.create(
                        messages=converted_messages,
                        extra_body={
                            **self.extra_body,
                        },
                        **self.model_kwargs,  # type: ignore
                    )  # type: ignore
            except (AuthenticationError, RateLimitError, BadRequestError) as exc:
                error_msg = f"OpenAI API error ({type(exc).__name__}): {exc}"
                logger.error(error_msg)
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
            if hasattr(completion, "model_dump"):
                completion = completion.model_dump()
            if cache_key:
                self._dump_cache(cache_key, completion)

        # Extract choice and parse
        choice = completion["choices"][0]["message"]
        try:
            if use_beta:
                parsed = self._parse_complete_output(completion, response_model)
            else:
                if "content" in choice:
                    content = choice["content"]
                    if not content:
                        raise ValueError("Response content is empty")
                else:
                    raise ValueError("Response has no content field")
                parsed = response_model.model_validate(jloads(content))
        except Exception as e:
            content = choice.get(
                "content", ""
            )  # Keep .get() here for error message only
            raise ValueError(
                f"Failed to parse model response: {e}\nRaw: {content}"
            ) from e

        return completion, choice, cast(OutputModelType, parsed)

    async def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[RawMsgs] = None,
    ):  # -> tuple[Any | dict[Any, Any], list[ChatCompletionMessagePar...:# -> tuple[Any | dict[Any, Any], list[ChatCompletionMessagePar...:
        """Unified async call for language model, returns (assistant_message.model_dump(), messages)."""
        if (prompt is None) == (messages is None):
            raise ValueError("Provide *either* `prompt` or `messages` (but not both).")

        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        assert messages is not None

        openai_msgs: Messages = (
            self._convert_messages(cast(LegacyMsgs, messages))
            if isinstance(messages[0], dict)
            else cast(Messages, messages)
        )

        use_cache = self.do_cache

        assert self.model is not None, "Model must be set before making a call."
        model: str = self.model

        cache_key = (
            self._cache_key(openai_msgs, self.model_kwargs, str) if use_cache else None
        )
        if cache_key and (hit := self._load_cache(cache_key)) is not None:
            if isinstance(hit, dict) and "error" in hit and hit["error"]:
                # Handle error response with explicit key checking
                error_type = "Unknown"
                error_message = "Cached error"

                if "error_type" in hit:
                    error_type = hit["error_type"]
                if "error_message" in hit:
                    error_message = hit["error_message"]

                logger.warning(f"Found cached error ({error_type}): {error_message}")
                raise ValueError(f"Cached {error_type}: {error_message}")
            raw_response = hit
        else:
            try:
                raw_response = await self.client.chat.completions.create(
                    model=model,
                    messages=list(openai_msgs),
                    **self.model_kwargs,  # type: ignore
                )  # type: ignore
            except (AuthenticationError, RateLimitError, BadRequestError) as exc:
                error_msg = f"OpenAI API error ({type(exc).__name__}): {exc}"
                logger.error(error_msg)
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
                self._dump_cache(cache_key, raw_response)

        # Extract the assistant's message
        assistant_msg = raw_response["choices"][0]["message"]
        # Build the full messages list (input + assistant reply)
        full_messages = list(messages) + [
            {"role": assistant_msg["role"], "content": assistant_msg["content"]}
        ]
        # Return the OpenAI message as model_dump (if available) and the messages list
        if hasattr(assistant_msg, "model_dump"):
            msg_dump = assistant_msg.model_dump()
        else:
            msg_dump = dict(assistant_msg)
        return msg_dump, full_messages

    async def parse(
        self,
        instruction,
        prompt,
    ) -> ParsedOutput[BaseModel]:
        """Parse response using guided JSON generation. Returns (parsed.model_dump(), messages)."""
        if not self.use_beta:
            assert self.add_json_schema_to_instruction, (
                "add_json_schema_to_instruction must be True when use_beta is False. otherwise model will not be able to parse the response."
            )

        assert self.response_model is not None, "response_model must be set at init."
        json_schema = self.response_model.model_json_schema()

        # Build system message content in a single, clear block
        assert instruction is not None, "Instruction must be provided."
        assert prompt is not None, "Prompt must be provided."
        system_content = instruction

        # Add schema if needed
        system_content = self.build_system_prompt(
            self.response_model,
            self.add_json_schema_to_instruction,
            json_schema,
            system_content,
            think=self.think,
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]  # type: ignore

        completion, choice, parsed = await self._call_and_parse(
            messages,
            self.response_model,
            json_schema,
            use_beta=self.use_beta,
        )

        # Build the full messages list (input + assistant reply)
        # Handle reasoning_content with explicit checking
        if "reasoning_content" in choice:
            reasoning_content = choice["reasoning_content"].strip()
        else:
            reasoning_content = ""

        # Handle content with explicit checking
        if "content" in choice:
            content = choice["content"]
            if not content:
                logger.warning("Assistant response content is empty")
                content = ""
        else:
            logger.warning("Assistant response has no content field")
            content = ""

        _content = content.lstrip("\n")
        assistant_msg = {
            "role": "assistant",
            "content": f"<think>\n{reasoning_content}\n</think>\n\n{_content}",
        }
        full_messages = messages + [assistant_msg]

        # Return the parsed model's model_dump and the messages list
        return ParsedOutput(
            messages=full_messages,
            parsed=cast(BaseModel, parsed),
            completion=completion,
            model_kwargs=self.model_kwargs,
        )

    async def inspect_history(self):
        """Inspect the history of the LLM calls."""
        pass
