# from ._utils import *
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Type,
    Union,
    cast,
)

from loguru import logger
from openai import AuthenticationError, BadRequestError, OpenAI, RateLimitError
from pydantic import BaseModel

# from llm_utils.lm.async_lm.async_llm_task import OutputModelType
from llm_utils.lm.async_lm.async_lm_base import AsyncLMBase
from speedy_utils import jloads

from ._utils import (
    LegacyMsgs,
    Messages,
    OutputModelType,
    ParsedOutput,
    RawMsgs,
)


def jloads_safe(content: str) -> Any:
    if "```json" in content:
        content = content.split("```json")[1].strip().split("```")[0].strip()
    try:
        return jloads(content)
    except Exception as e:
        logger.error(
            f"Failed to parse JSON content: {content[:100]}... with error: {e}"
        )
        raise ValueError(f"Invalid JSON content: {content}") from e


class AsyncLM(AsyncLMBase):
    """Unified **async** language‑model wrapper with optional JSON parsing."""

    def __init__(
        self,
        *,
        model: str | None = None,
        response_model: type[BaseModel] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2_000,
        host: str = "localhost",
        port: int | str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        cache: bool = True,
        think: Literal[True, False, None] = None,
        add_json_schema_to_instruction: bool | None = None,
        use_beta: bool = False,
        ports: list[int] | None = None,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        top_k: int = 1,
        repetition_penalty: float = 1.0,
        frequency_penalty: float | None = None,
    ) -> None:

        if model is None:
            models = (
                OpenAI(base_url=f"http://{host}:{port}/v1", api_key="abc")
                .models.list()
                .data
            )
            assert len(models) == 1, f"Found {len(models)} models, please specify one."
            model = models[0].id
            print(f"Using model: {model}")

        super().__init__(
            ports=ports,
            base_url=base_url,
            cache=cache,
            api_key=api_key,
        )

        # Model behavior options
        self.response_model = response_model
        self.think = think
        self._use_beta = use_beta
        self.add_json_schema_to_instruction = add_json_schema_to_instruction
        if not use_beta:
            self.add_json_schema_to_instruction = True

        # Store all model-related parameters in model_kwargs
        self.model_kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
        }
        self.extra_body = {
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "frequency_penalty": frequency_penalty,
        }

    async def _unified_client_call(
        self,
        messages: RawMsgs,
        extra_body: dict | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Unified method for all client interactions (caching handled by MAsyncOpenAI)."""
        converted_messages: Messages = (
            self._convert_messages(cast(LegacyMsgs, messages))
            if messages and isinstance(messages[0], dict)
            else cast(Messages, messages)
        )
        # override max_tokens if provided
        if max_tokens is not None:
            self.model_kwargs["max_tokens"] = max_tokens

        try:
            # Get completion from API (caching handled by MAsyncOpenAI)
            call_kwargs = {
                "messages": converted_messages,
                **self.model_kwargs,
            }
            if extra_body:
                call_kwargs["extra_body"] = extra_body

            completion = await self.client.chat.completions.create(**call_kwargs)

            if hasattr(completion, "model_dump"):
                completion = completion.model_dump()

        except (AuthenticationError, RateLimitError, BadRequestError) as exc:
            error_msg = f"OpenAI API error ({type(exc).__name__}): {exc}"
            logger.error(error_msg)
            raise

        return completion

    async def _call_and_parse(
        self,
        messages: list[dict],
        response_model: type[OutputModelType],
        json_schema: dict,
    ) -> tuple[dict, list[dict], OutputModelType]:
        """Unified call and parse with cache and error handling."""
        if self._use_beta:
            return await self._call_and_parse_with_beta(
                messages, response_model, json_schema
            )

        choice = None
        try:
            # Use unified client call
            completion = await self._unified_client_call(
                messages,
                extra_body={**self.extra_body},
            )

            # Parse the response
            choice = completion["choices"][0]["message"]
            if "content" not in choice:
                raise ValueError("Response choice must contain 'content' field.")

            content = choice["content"]
            if not content:
                raise ValueError("Response content is empty")

            parsed = response_model.model_validate(jloads_safe(content))

        except Exception as e:
            # Try fallback to beta mode if regular parsing fails
            if not isinstance(
                e, (AuthenticationError, RateLimitError, BadRequestError)
            ):
                content = choice.get("content", "N/A") if choice else "N/A"
                logger.info(
                    f"Regular parsing failed due to wrong format or content, now falling back to beta mode: {content=}, {e=}"
                )
                try:
                    return await self._call_and_parse_with_beta(
                        messages, response_model, json_schema
                    )
                except Exception as beta_e:
                    logger.warning(f"Beta mode fallback also failed: {beta_e}")
                    choice_info = choice if choice is not None else "N/A"
                    raise ValueError(
                        f"Failed to parse model response with both regular and beta modes. "
                        f"Regular error: {e}. Beta error: {beta_e}. "
                        f"Model response message: {choice_info}"
                    ) from e
            raise

        assistant_msg = self._extract_assistant_message(choice)
        full_messages = messages + [assistant_msg]

        return completion, full_messages, cast(OutputModelType, parsed)

    async def _call_and_parse_with_beta(
        self,
        messages: list[dict],
        response_model: type[OutputModelType],
        json_schema: dict,
    ) -> tuple[dict, list[dict], OutputModelType]:
        """Call and parse for beta mode with guided JSON."""
        choice = None
        try:
            # Use unified client call with guided JSON
            completion = await self._unified_client_call(
                messages,
                extra_body={"guided_json": json_schema, **self.extra_body},
            )

            # Parse the response
            choice = completion["choices"][0]["message"]
            parsed = self._parse_complete_output(completion, response_model)

        except Exception as e:
            choice_info = choice if choice is not None else "N/A"
            raise ValueError(
                f"Failed to parse model response: {e}\nModel response message: {choice_info}"
            ) from e

        assistant_msg = self._extract_assistant_message(choice)
        full_messages = messages + [assistant_msg]

        return completion, full_messages, cast(OutputModelType, parsed)

    def _extract_assistant_message(self, choice):  # -> dict[str, str] | dict[str, Any]:
        # TODO this current assume choice is a dict with "reasoning_content" and "content"
        has_reasoning = False
        reasoning_content = ""
        if "reasoning_content" in choice and isinstance(
            choice["reasoning_content"], str
        ):
            reasoning_content = choice["reasoning_content"].strip()
            has_reasoning = True

        content = choice["content"]
        _content = content.lstrip("\n")
        if has_reasoning:
            assistant_msg = {
                "role": "assistant",
                "content": f"<think>\n{reasoning_content}\n</think>\n\n{_content}",
            }
        else:
            assistant_msg = {"role": "assistant", "content": _content}

        return assistant_msg

    async def call_with_messages(
        self,
        prompt: str | None = None,
        messages: RawMsgs | None = None,
        max_tokens: int | None = None,
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

        assert (
            self.model_kwargs["model"] is not None
        ), "Model must be set before making a call."

        # Use unified client call
        raw_response = await self._unified_client_call(
            list(openai_msgs), max_tokens=max_tokens
        )

        if hasattr(raw_response, "model_dump"):
            raw_response = raw_response.model_dump()  # type: ignore

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

    def call_sync(
        self,
        prompt: str | None = None,
        messages: RawMsgs | None = None,
        max_tokens: int | None = None,
    ):
        """Synchronous wrapper around the async __call__ method."""
        import asyncio

        return asyncio.run(
            self.__call__(prompt=prompt, messages=messages, max_tokens=max_tokens)
        )

    async def parse(
        self,
        instruction,
        prompt,
    ) -> ParsedOutput[BaseModel]:
        """Parse response using guided JSON generation. Returns (parsed.model_dump(), messages)."""
        if not self._use_beta:
            assert (
                self.add_json_schema_to_instruction
            ), "add_json_schema_to_instruction must be True when use_beta is False. otherwise model will not be able to parse the response."

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

        completion, full_messages, parsed = await self._call_and_parse(
            messages,
            self.response_model,
            json_schema,
        )

        return ParsedOutput(
            messages=full_messages,
            parsed=cast(BaseModel, parsed),
            completion=completion,
            model_kwargs=self.model_kwargs,
        )

    def _parse_complete_output(
        self, completion: Any, response_model: type[BaseModel]
    ) -> BaseModel:
        """Parse completion output to response model."""
        if hasattr(completion, "model_dump"):
            completion = completion.model_dump()

        if "choices" not in completion or not completion["choices"]:
            raise ValueError("No choices in OpenAI response")

        content = completion["choices"][0]["message"]["content"]
        if not content:
            raise ValueError("Response content is empty")

        try:
            data = jloads(content)
            return response_model.model_validate(data)
        except Exception as exc:
            raise ValueError(
                f"Failed to validate against response model {response_model.__name__}: {exc}\nRaw content: {content}"
            ) from exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self, "_last_client"):
            last_client = self._last_client  # type: ignore
            await last_client._client.aclose()
        else:
            logger.warning("No last client to close")
