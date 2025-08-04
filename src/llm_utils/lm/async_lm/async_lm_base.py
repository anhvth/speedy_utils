# from ._utils import *
import base64
import hashlib
import json
import os
from typing import (
    Any,
    List,
    Optional,
    Type,
    Union,
    cast,
    overload,
)

from httpx import URL
from loguru import logger
from openai import AsyncOpenAI
from openai.pagination import AsyncPage as AsyncSyncPage
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.model import Model
from pydantic import BaseModel

from ._utils import (
    LegacyMsgs,
    Messages,
    RawMsgs,
    TModel,
)


class AsyncLMBase:
    """Unified **async** language‑model wrapper with optional JSON parsing."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: Optional[int | str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: bool = True,
        ports: Optional[List[int]] = None,
    ) -> None:
        self._port = port
        self._host = host
        self.base_url = base_url or (f"http://{host}:{port}/v1" if port else None)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "abc")
        self._cache = cache
        self.ports = ports
        self._init_port = port  # <-- store the port provided at init

    @property
    def client(self) -> AsyncOpenAI:
        # if have multiple ports
        if self.ports:
            import random

            port = random.choice(self.ports)
            api_base = f"http://{self._host}:{port}/v1"
            logger.debug(f"Using port: {port}")
        else:
            api_base = self.base_url or f"http://{self._host}:{self._port}/v1"
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=api_base,
        )
        self._last_client = client
        return client

    # ------------------------------------------------------------------ #
    # Public API – typed overloads
    # ------------------------------------------------------------------ #
    @overload
    async def __call__(  # type: ignore
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

    # async def inspect_word_probs(
    #     self,
    #     messages: Optional[List[Dict[str, Any]]] = None,
    #     tokenizer: Optional[Any] = None,
    #     do_print=True,
    #     add_think: bool = True,
    # ) -> tuple[List[Dict[str, Any]], Any, str]:
    #     """
    #     Inspect word probabilities in a language model response.

    #     Args:
    #         tokenizer: Tokenizer instance to encode words.
    #         messages: List of messages to analyze.

    #     Returns:
    #         A tuple containing:
    #         - List of word probabilities with their log probabilities.
    #         - Token log probability dictionaries.
    #         - Rendered string with colored word probabilities.
    #     """
    #     if messages is None:
    #         messages = await self.last_messages(add_think=add_think)
    #         if messages is None:
    #             raise ValueError("No messages provided and no last messages available.")

    #     if tokenizer is None:
    #         tokenizer = get_tokenizer(self.model)

    #     ret = await inspect_word_probs_async(self, tokenizer, messages)
    #     if do_print:
    #         print(ret[-1])
    #     return ret

    # async def last_messages(
    #     self, add_think: bool = True
    # ) -> Optional[List[Dict[str, str]]]:
    #     """Get the last conversation messages including assistant response."""
    #     if not hasattr(self, "last_log"):
    #         return None

    #     last_conv = self._last_log
    #     messages = last_conv[1] if len(last_conv) > 1 else None
    #     last_msg = last_conv[2]
    #     if not isinstance(last_msg, dict):
    #         last_conv[2] = last_conv[2].model_dump()  # type: ignore
    #     msg = last_conv[2]
    #     # Ensure msg is a dict
    #     if hasattr(msg, "model_dump"):
    #         msg = msg.model_dump()
    #     message = msg["choices"][0]["message"]
    #     reasoning = message.get("reasoning_content")
    #     answer = message.get("content")
    #     if reasoning and add_think:
    #         final_answer = f"<think>{reasoning}</think>\n{answer}"
    #     else:
    #         final_answer = f"<think>\n\n</think>\n{answer}"
    #     assistant = {"role": "assistant", "content": final_answer}
    #     messages = messages + [assistant]  # type: ignore
    #     return messages if messages else None

    # async def inspect_history(self) -> None:
    #     """Inspect the conversation history with proper formatting."""
    #     if not hasattr(self, "last_log"):
    #         raise ValueError("No history available. Please call the model first.")

    #     prompt, messages, response = self._last_log
    #     if hasattr(response, "model_dump"):
    #         response = response.model_dump()
    #     if not messages:
    #         messages = [{"role": "user", "content": prompt}]

    #     print("\n\n")
    #     print(_blue("[Conversation History]") + "\n")

    #     for msg in messages:
    #         role = msg["role"]
    #         content = msg["content"]
    #         print(_red(f"{role.capitalize()}:"))
    #         if isinstance(content, str):
    #             print(content.strip())
    #         elif isinstance(content, list):
    #             for item in content:
    #                 if item.get("type") == "text":
    #                     print(item["text"].strip())
    #                 elif item.get("type") == "image_url":
    #                     image_url = item["image_url"]["url"]
    #                     if "base64" in image_url:
    #                         len_base64 = len(image_url.split("base64,")[1])
    #                         print(_blue(f"<IMAGE BASE64 ENCODED({len_base64})>"))
    #                     else:
    #                         print(_blue(f"<image_url: {image_url}>"))
    #         print("\n")

    #     print(_red("Response:"))
    #     if isinstance(response, dict) and response.get("choices"):
    #         message = response["choices"][0].get("message", {})
    #         reasoning = message.get("reasoning_content")
    #         parsed = message.get("parsed")
    #         content = message.get("content")
    #         if reasoning:
    #             print(_yellow("<think>"))
    #             print(reasoning.strip())
    #             print(_yellow("</think>\n"))
    #         if parsed:
    #             print(
    #                 json.dumps(
    #                     (
    #                         parsed.model_dump()
    #                         if hasattr(parsed, "model_dump")
    #                         else parsed
    #                     ),
    #                     indent=2,
    #                 )
    #                 + "\n"
    #             )
    #         elif content:
    #             print(content.strip())
    #         else:
    #             print(_green("[No content]"))
    #         if len(response["choices"]) > 1:
    #             print(
    #                 _blue(f"\n(Plus {len(response['choices']) - 1} other completions)")
    #             )
    #     else:
    #         print(_yellow("Warning: Not a standard OpenAI response object"))
    #         if isinstance(response, str):
    #             print(_green(response.strip()))
    #         elif isinstance(response, dict):
    #             print(_green(json.dumps(response, indent=2)))
    #         else:
    #             print(_green(str(response)))

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    async def list_models(port=None, host="localhost") -> List[str]:
        try:
            client = AsyncLMBase(port=port, host=host).client  # type: ignore[arg-type]
            base_url: URL = client.base_url
            logger.debug(f"Base URL: {base_url}")
            models: AsyncSyncPage[Model] = await client.models.list()  # type: ignore[assignment]
            return [model.id for model in models.data]
        except Exception as exc:
            logger.error(f"Failed to list models: {exc}")
            return []

    def build_system_prompt(
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

    async def inspect_history(self):
        """Inspect the history of the LLM calls."""
        pass
