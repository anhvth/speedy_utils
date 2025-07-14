# from ._utils import *
from typing import (
    Any,
    List,
    Literal,
    Optional,
    Type,
    cast,
)

from loguru import logger
from openai import AuthenticationError, BadRequestError, RateLimitError
from pydantic import BaseModel
from speedy_utils import jloads

# from llm_utils.lm.async_lm.async_llm_task import OutputModelType
from llm_utils.lm.async_lm.async_lm_base import AsyncLMBase

from ._utils import (
    LegacyMsgs,
    Messages,
    OutputModelType,
    ParsedOutput,
    RawMsgs,
)


def jloads_safe(content: str) -> Any:
    # if contain ```json, remove it
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
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        top_k: int = 1,
        repetition_penalty: float = 1.0,
        frequency_penalty: Optional[float] = None,
    ) -> None:
        super().__init__(
            host=host,
            port=port,
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
        self.model_kwargs = dict(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            presence_penalty=presence_penalty,
        )
        self.extra_body = dict(
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
        )

    async def _unified_client_call(
        self,
        messages: list[dict],
        extra_body: Optional[dict] = None,
        cache_suffix: str = "",
    ) -> dict:
        """Unified method for all client interactions with caching and error handling."""
        converted_messages = self._convert_messages(messages)
        cache_key = None
        completion = None

        # Handle caching
        if self._cache:
            cache_data = {
                "messages": converted_messages,
                "model_kwargs": self.model_kwargs,
                "extra_body": extra_body or {},
                "cache_suffix": cache_suffix,
            }
            cache_key = self._cache_key(cache_data, {}, str)
            completion = self._load_cache(cache_key)

        # Check for cached error responses
        if (
            completion
            and isinstance(completion, dict)
            and "error" in completion
            and completion["error"]
        ):
            error_type = completion.get("error_type", "Unknown")
            error_message = completion.get("error_message", "Cached error")
            logger.warning(f"Found cached error ({error_type}): {error_message}")
            raise ValueError(f"Cached {error_type}: {error_message}")

        try:
            # Get completion from API if not cached
            if not completion:
                call_kwargs = {
                    "messages": converted_messages,
                    **self.model_kwargs,
                }
                if extra_body:
                    call_kwargs["extra_body"] = extra_body

                completion = await self.client.chat.completions.create(**call_kwargs)

                if hasattr(completion, "model_dump"):
                    completion = completion.model_dump()
                if cache_key:
                    self._dump_cache(cache_key, completion)

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

        return completion

    async def _call_and_parse(
        self,
        messages: list[dict],
        response_model: Type[OutputModelType],
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
                cache_suffix=f"_parse_{response_model.__name__}",
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
        response_model: Type[OutputModelType],
        json_schema: dict,
    ) -> tuple[dict, list[dict], OutputModelType]:
        """Call and parse for beta mode with guided JSON."""
        choice = None
        try:
            # Use unified client call with guided JSON
            completion = await self._unified_client_call(
                messages,
                extra_body={"guided_json": json_schema, **self.extra_body},
                cache_suffix=f"_beta_parse_{response_model.__name__}",
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

        assert self.model_kwargs["model"] is not None, (
            "Model must be set before making a call."
        )

        # Use unified client call
        raw_response = await self._unified_client_call(
            list(openai_msgs), cache_suffix="_call"
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

    async def parse(
        self,
        instruction,
        prompt,
    ) -> ParsedOutput[BaseModel]:
        """Parse response using guided JSON generation. Returns (parsed.model_dump(), messages)."""
        if not self._use_beta:
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
        self, completion: Any, response_model: Type[BaseModel]
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
