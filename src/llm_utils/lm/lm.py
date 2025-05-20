from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
    overload,
    cast,
)

from httpx import URL
from huggingface_hub import repo_info
from loguru import logger
from numpy import isin
from openai import OpenAI, AuthenticationError, RateLimitError
from openai.pagination import SyncPage
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.parsed_chat_completion import ParsedChatCompletion
from openai.types.model import Model
from pydantic import BaseModel
import warnings

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
        assert self.model is not None, "Model must be set before calling."
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
        if isinstance(response, dict) and 'choices' in response and response['choices']:
            message = response['choices'][0].get('message', {})

            # Check for reasoning content (if available)
            reasoning = message.get('reasoning_content')

            # Check for parsed content (structured mode)
            parsed = message.get('parsed')

            # Get regular content
            content = message.get('content')

            # Display reasoning if available
            if reasoning:
                print(_yellow('<think>'))
                print(reasoning.strip())
                print(_yellow('</think>'))
                print()

            # Display parsed content for structured responses
            if parsed:
                # print(_green('<Parsed Structure>'))
                if hasattr(parsed, 'model_dump'):
                    print(json.dumps(parsed.model_dump(), indent=2))
                else:
                    print(json.dumps(parsed, indent=2))
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
            if len(response['choices']) > 1:
                print(_blue(f"\n(Plus {len(response['choices']) - 1} other completions)"))
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
            self._cache_key(messages, kw, response_format)
            if use_cache
            else None
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
        if hasattr(raw_response, 'model_dump'):
            raw_response = raw_response.model_dump()
            
        if response_format is str:
            # Extract the content from OpenAI response dict
            if isinstance(raw_response, dict) and 'choices' in raw_response:
                message = raw_response['choices'][0]['message']
                return message.get('content', '') or ''
            return cast(str, raw_response)
        
        # For the type-checker: we *know* it's a BaseModel subclass here.
        model_cls = cast(Type[BaseModel], response_format)

        # Handle structured response
        if isinstance(raw_response, dict) and 'choices' in raw_response:
            message = raw_response['choices'][0]['message']
            
            # Check if already parsed by OpenAI client
            if 'parsed' in message:
                return model_cls.model_validate(message['parsed'])
            
            # Need to parse the content
            content = message.get('content')
            if content is None:
                raise ValueError("Model returned empty content")
            
            try:
                data = json.loads(content)
                return model_cls.model_validate(data)
            except Exception as exc:
                raise ValueError(f"Failed to parse model output as JSON:\n{content}") from exc
        
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
            raise ValueError(f"Model did not return valid JSON:\n---\n{raw_response}") from exc

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
    def list_models(port=None, host="localhost") -> List[str]:
        """
        List available models.
        """
        try:
            client: OpenAI = LM(port=port, host=host).client
            base_url: URL = client.base_url
            logger.debug(f"Base URL: {base_url}")
            models: SyncPage[Model] = client.models.list()
            return [model.id for model in models.data]
        except Exception as exc:
            logger.error(f"Failed to list models: {exc}")
            return []
