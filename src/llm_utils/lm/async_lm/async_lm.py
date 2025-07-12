# from ._utils import *
import base64
import hashlib
import json
import os
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    overload,
)

from httpx import URL
from loguru import logger
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

from speedy_utils import jloads

from ._utils import (
    LegacyMsgs,
    Messages,
    ParsedOutput,
    RawMsgs,
    TModel,
    TParsed,
    _blue,
    _green,
    _red,
    _yellow,
    get_tokenizer,
    inspect_word_probs_async,
)


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

    async def _set_model(self) -> None:
        if not self.model:
            models = await self.list_models(port=self.port, host=self.host)
            self.model = models[0] if models else None
            logger.info(
                f"No model specified. Using the first available model. {self.model}"
            )

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
        await self._set_model()

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

        self._last_log = [prompt, messages, raw_response]
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
        instruction,
        prompt,
        think: Literal[True, False, None] = None,
        add_json_schema_to_instruction: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        cache: Optional[bool] = None,
        use_beta: bool = False,
        **kwargs,
    ) -> ParsedOutput[TParsed]:
        """Parse response using guided JSON generation."""

        if not use_beta:
            assert add_json_schema_to_instruction, (
                "add_json_schema_to_instruction must be True when use_beta is False. otherwise model will not be able to parse the response."
            )

        json_schema = response_model.model_json_schema()

        # Build system message content in a single, clear block
        assert instruction is not None, "Instruction must be provided."
        assert prompt is not None, "Prompt must be provided."
        system_content = instruction

        # Add schema if needed
        system_content = self._build_system_prompt(
            response_model,
            add_json_schema_to_instruction,
            json_schema,
            system_content,
            think=think,
        )

        # Rebuild messages with updated system message if needed
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ]  # type: ignore

        model_kwargs = {}
        if temperature is not None:
            model_kwargs["temperature"] = temperature
        if max_tokens is not None:
            model_kwargs["max_tokens"] = max_tokens
        model_kwargs.update(kwargs)

        use_cache = self.do_cache if cache is None else cache
        cache_key = None
        completion = None
        choice = None
        parsed = None

        if use_cache:
            cache_data = {
                "messages": messages,
                "model_kwargs": model_kwargs,
                "guided_json": json_schema,
                "response_format": response_model.__name__,
                "use_beta": use_beta,
            }
            cache_key = self._cache_key(cache_data, {}, response_model)
            completion = self._load_cache(cache_key)  # dict

        if not completion:
            completion, choice, parsed = await self._call_and_parse_completion(
                messages,
                response_model,
                json_schema,
                use_beta=use_beta,
                model_kwargs=model_kwargs,
            )

            if cache_key:
                self._dump_cache(cache_key, completion)
        else:
            # Extract choice and parsed from cached completion
            choice = completion["choices"][0]["message"]
            try:
                parsed = self._parse_complete_output(completion, response_model)
            except Exception as e:
                raise ValueError(
                    f"Failed to parse cached completion: {e}\nRaw: {choice.get('content')}"
                ) from e

        assert isinstance(completion, dict), (
            "Completion must be a dictionary with OpenAI response format."
        )
        self._last_log = [prompt, messages, completion]

        reasoning_content = choice.get("reasoning_content", "").strip()
        _content = choice.get("content", "").lstrip("\n")
        content = f"<think>\n{reasoning_content}\n</think>\n\n{_content}"

        full_messages = messages + [{"role": "assistant", "content": content}]

        return ParsedOutput(
            messages=full_messages,
            completion=completion,
            parsed=parsed,  # type: ignore
        )

    def _build_system_prompt(
        self,
        response_model,
        add_json_schema_to_instruction,
        json_schema,
        system_content,
        think,
    ):
        if add_json_schema_to_instruction and response_model:
            schema_block = f"\n\n<output_json_schema>\n{json.dumps(json_schema, indent=2)}\n</output_json_schema>"
            # if schema_block not in system_content:
            if "<output_json_schema>" in system_content:
                # remove exsting schema block
                import re  # replace

                system_content = re.sub(
                    r"<output_json_schema>.*?</output_json_schema>",
                    "",
                    system_content,
                    flags=re.DOTALL,
                )
                system_content = system_content.strip()
            system_content += schema_block

        if think is True:
            if "/think" in system_content:
                pass
            elif "/no_think" in system_content:
                system_content = system_content.replace("/no_think", "/think")
            else:
                system_content += "\n\n/think"
        elif think is False:
            if "/no_think" in system_content:
                pass
            elif "/think" in system_content:
                system_content = system_content.replace("/think", "/no_think")
            else:
                system_content += "\n\n/no_think"
        return system_content

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
            # Enhanced error for debugging: show input tokens and their count

            # Try to extract tokens from the completion for debugging
            input_tokens = None
            try:
                input_tokens = completion.get("usage", {}).get("prompt_tokens")
            except Exception:
                input_tokens = None

            # Try to get the prompt/messages for tokenization
            prompt = None
            try:
                prompt = completion.get("messages") or completion.get("prompt")
            except Exception:
                prompt = None

            tokens_preview = ""
            if prompt is not None:
                try:
                    tokenizer = get_tokenizer(self.model)
                    if isinstance(prompt, list):
                        prompt_text = "\n".join(
                            m.get("content", "") for m in prompt if isinstance(m, dict)
                        )
                    else:
                        prompt_text = str(prompt)
                    tokens = tokenizer.encode(prompt_text)
                    n_tokens = len(tokens)
                    first_100 = tokens[:100]
                    last_100 = tokens[-100:] if n_tokens > 100 else []
                    tokens_preview = (
                        f"\nInput tokens: {n_tokens}"
                        f"\nFirst 100 tokens: {first_100}"
                        f"\nLast 100 tokens: {last_100}"
                    )
                except Exception as exc:
                    tokens_preview = f"\n[Tokenization failed: {exc}]"

            raise ValueError(
                f"Empty content in response."
                f"\nInput tokens (if available): {input_tokens}"
                f"{tokens_preview}"
            )

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

        last_conv = self._last_log
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

        prompt, messages, response = self._last_log
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

    async def _call_and_parse_completion(
        self,
        messages: list[dict],
        response_model: Type[TParsed],
        json_schema: dict,
        use_beta: bool,
        model_kwargs: dict,
    ) -> tuple[dict, dict, TParsed]:
        """Call vLLM or OpenAI-compatible endpoint and parse JSON response consistently."""
        await self._set_model()  # Ensure model is set before making the call
        # Convert messages to proper type
        converted_messages = self._convert_messages(messages)  # type: ignore

        if use_beta:
            # Use guided JSON for structure enforcement
            try:
                completion = await self.client.chat.completions.create(
                    model=str(self.model),  # type: ignore
                    messages=converted_messages,
                    extra_body={"guided_json": json_schema},  # type: ignore
                    **model_kwargs,
                )  # type: ignore
            except Exception:
                # Fallback if extra_body is not supported
                completion = await self.client.chat.completions.create(
                    model=str(self.model),  # type: ignore
                    messages=converted_messages,
                    response_format={"type": "json_object"},
                    **model_kwargs,
                )
        else:
            # Use OpenAI-style structured output
            completion = await self.client.chat.completions.create(
                model=str(self.model),  # type: ignore
                messages=converted_messages,
                response_format={"type": "json_object"},
                **model_kwargs,
            )

        if hasattr(completion, "model_dump"):
            completion = completion.model_dump()

        choice = completion["choices"][0]["message"]

        try:
            parsed = (
                self._parse_complete_output(completion, response_model)
                if use_beta
                else response_model.model_validate(jloads(choice.get("content")))
            )
        except Exception as e:
            raise ValueError(
                f"Failed to parse model response: {e}\nRaw: {choice.get('content')}"
            ) from e

        return completion, choice, parsed  # type: ignore
