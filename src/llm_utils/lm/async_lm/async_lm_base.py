# from ._utils import *
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

from llm_utils.lm.openai_memoize import MAsyncOpenAI

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
        base_url: str | None = None,
        api_key: str | None = None,
        cache: bool = True,
        ports: list[int] | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', 'abc')
        self._cache = cache
        self.ports = ports

    @property
    def client(self) -> 'MAsyncOpenAI': # type: ignore
        # if have multiple ports
        if self.ports and self.base_url:
            import random
            import re

            port = random.choice(self.ports)
            # Replace port in base_url if it exists
            base_url_pattern = r'(https?://[^:/]+):?\d*(/.*)?'
            match = re.match(base_url_pattern, self.base_url)
            if match:
                host_part = match.group(1)
                path_part = match.group(2) or '/v1'
                api_base = f'{host_part}:{port}{path_part}'
            else:
                api_base = self.base_url
            logger.debug(f'Using port: {port}')
        else:
            api_base = self.base_url

        if api_base is None:
            raise ValueError('base_url must be provided')

        client = MAsyncOpenAI(
            api_key=self.api_key,
            base_url=api_base,
            cache=self._cache,
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
        response_format: type[TModel],
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
            role = msg['role']
            content = msg['content']
            if role == 'user':
                converted.append(
                    ChatCompletionUserMessageParam(role='user', content=content)
                )
            elif role == 'assistant':
                converted.append(
                    ChatCompletionAssistantMessageParam(
                        role='assistant', content=content
                    )
                )
            elif role == 'system':
                converted.append(
                    ChatCompletionSystemMessageParam(role='system', content=content)
                )
            elif role == 'tool':
                converted.append(
                    ChatCompletionToolMessageParam(
                        role='tool',
                        content=content,
                        tool_call_id=msg.get('tool_call_id') or '',
                    )
                )
            else:
                converted.append({'role': role, 'content': content})  # type: ignore[arg-type]
        return converted

    @staticmethod
    def _parse_output(
        raw_response: Any, response_format: type[str] | type[BaseModel]
    ) -> str | BaseModel:
        if hasattr(raw_response, 'model_dump'):
            raw_response = raw_response.model_dump()

        if response_format is str:
            if isinstance(raw_response, dict) and 'choices' in raw_response:
                message = raw_response['choices'][0]['message']
                return message.get('content', '') or ''
            return cast(str, raw_response)

        model_cls = cast(type[BaseModel], response_format)

        if isinstance(raw_response, dict) and 'choices' in raw_response:
            message = raw_response['choices'][0]['message']
            if 'parsed' in message:
                return model_cls.model_validate(message['parsed'])
            content = message.get('content')
            if content is None:
                raise ValueError('Model returned empty content')
            try:
                data = json.loads(content)
                return model_cls.model_validate(data)
            except Exception as exc:
                raise ValueError(
                    f'Failed to parse model output as JSON:\n{content}'
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
                f'Model did not return valid JSON:\n---\n{raw_response}'
            ) from exc

    # ------------------------------------------------------------------ #
    # Misc helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    async def list_models(base_url: str | None = None) -> list[str]:
        try:
            if base_url is None:
                raise ValueError('base_url must be provided')
            client = AsyncLMBase(base_url=base_url).client
            base_url_obj: URL = client.base_url
            logger.debug(f'Base URL: {base_url_obj}')
            models: AsyncSyncPage[Model] = await client.models.list()  # type: ignore[assignment]
            return [model.id for model in models.data]
        except Exception as exc:
            logger.error(f'Failed to list models: {exc}')
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
            schema_block = f'\n\n<output_json_schema>\n{json.dumps(json_schema, indent=2)}\n</output_json_schema>'
            # if schema_block not in system_content:
            if '<output_json_schema>' in system_content:
                # remove exsting schema block
                import re  # replace

                system_content = re.sub(
                    r'<output_json_schema>.*?</output_json_schema>',
                    '',
                    system_content,
                    flags=re.DOTALL,
                )
                system_content = system_content.strip()
            system_content += schema_block

        if think is True:
            if '/think' in system_content:
                pass
            elif '/no_think' in system_content:
                system_content = system_content.replace('/no_think', '/think')
            else:
                system_content += '\n\n/think'
        elif think is False:
            if '/no_think' in system_content:
                pass
            elif '/think' in system_content:
                system_content = system_content.replace('/think', '/no_think')
            else:
                system_content += '\n\n/no_think'
        return system_content

    async def inspect_history(self):
        """Inspect the history of the LLM calls."""
