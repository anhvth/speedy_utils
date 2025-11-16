# type: ignore

"""
Simplified LLM Task module for handling language model interactions with structured input/output.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, create_model


# Type aliases for better readability
Messages = list[ChatCompletionMessageParam]

import json
from typing import TypeVar


B = TypeVar('B', bound='BasePromptBuilder')


class BasePromptBuilder(BaseModel, ABC):
    """
    Abstract base class for prompt builders.
    Provides a consistent interface for:
    - input/output key declaration
    - prompt building
    - schema enforcement via auto-built modget_io_keysels
    """

    # ------------------------------------------------------------------ #
    # Abstract methods
    # ------------------------------------------------------------------ #
    @abstractmethod
    def get_instruction(self) -> str:
        """Return the system instruction string (role of the model)."""
        raise NotImplementedError

    @abstractmethod
    def get_io_keys(self) -> tuple[list[str], list[str | tuple[str, str]]]:
        """
        Return (input_keys, output_keys).
        Each key must match a field of the subclass.
        For output_keys, you can use:
        - str: Use the field name as-is
        - tuple[str, str]: (original_field_name, renamed_field_name)
        Input keys are always strings.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Auto-build models from keys
    # ------------------------------------------------------------------ #
    def _build_model_from_keys(
        self, keys: list[str] | list[str | tuple[str, str]], name: str
    ) -> type[BaseModel]:
        fields: dict[str, tuple[Any, Any]] = {}
        for key in keys:
            if isinstance(key, tuple):
                # Handle tuple: (original_field_name, renamed_field_name)
                original_key, renamed_key = key
                if original_key not in self.model_fields:
                    raise ValueError(f"Key '{original_key}' not found in model fields")
                field_info = self.model_fields[original_key]
                field_type = (
                    field_info.annotation
                    if field_info.annotation is not None
                    else (Any,)
                )
                default = field_info.default if field_info.default is not None else ...
                fields[renamed_key] = (field_type, default)
            else:
                # Handle string key
                if key not in self.model_fields:
                    raise ValueError(f"Key '{key}' not found in model fields")
                field_info = self.model_fields[key]
                field_type = (
                    field_info.annotation
                    if field_info.annotation is not None
                    else (Any,)
                )
                default = field_info.default if field_info.default is not None else ...
                fields[key] = (field_type, default)
        return create_model(name, **fields)  # type: ignore

    def get_input_model(self) -> type[BaseModel]:
        input_keys, _ = self.get_io_keys()
        return self._build_model_from_keys(input_keys, 'InputModel')

    def get_output_model(self) -> type[BaseModel]:
        _, output_keys = self.get_io_keys()
        return self._build_model_from_keys(output_keys, 'OutputModel')

    # ------------------------------------------------------------------ #
    # Dump methods (JSON)
    # ------------------------------------------------------------------ #
    def _dump_json_unique(
        self,
        schema_model: type[BaseModel],
        keys: list[str] | list[str | tuple[str, str]],
        **kwargs,
    ) -> str:
        allowed = list(schema_model.model_fields.keys())
        seen = set()
        unique_keys = [k for k in allowed if not (k in seen or seen.add(k))]
        data = self.model_dump()

        # Handle key mapping for renamed fields
        filtered = {}
        for key in keys:
            if isinstance(key, tuple):
                original_key, renamed_key = key
                if original_key in data and renamed_key in unique_keys:
                    filtered[renamed_key] = data[original_key]
            else:
                if key in data and key in unique_keys:
                    filtered[key] = data[key]

        return schema_model(**filtered).model_dump_json(**kwargs)

    def model_dump_json_input(self, **kwargs) -> str:
        input_keys, _ = self.get_io_keys()
        return self._dump_json_unique(self.get_input_model(), input_keys, **kwargs)

    def model_dump_json_output(self, **kwargs) -> str:
        _, output_keys = self.get_io_keys()
        return self._dump_json_unique(self.get_output_model(), output_keys, **kwargs)

    # ------------------------------------------------------------------ #
    # Markdown helpers
    # ------------------------------------------------------------------ #
    def _to_markdown(self, obj: Any, level: int = 1, title: str | None = None) -> str:
        """
        Recursively convert dict/list/primitive into clean, generic Markdown.
        """
        md: list[str] = []

        # Format title if provided
        if title is not None:
            formatted_title = title.replace('_', ' ').title()
            if level <= 2:
                md.append(f'{"#" * level} {formatted_title}')
            else:
                md.append(f'**{formatted_title}:**')

        if isinstance(obj, dict):
            if not obj:  # Empty dict
                md.append('None')
            else:
                for k, v in obj.items():
                    if isinstance(v, (str, int, float, bool)) and len(str(v)) < 100:
                        # Short values inline
                        key_name = k.replace('_', ' ').title()
                        if level <= 2:
                            md.append(f'**{key_name}:** {v}')
                        else:
                            md.append(f'- **{key_name}:** {v}')
                    else:
                        # Complex values get recursive handling
                        md.append(self._to_markdown(v, level=level + 1, title=k))
        elif isinstance(obj, list):
            if not obj:  # Empty list
                md.append('None')
            elif all(isinstance(i, dict) for i in obj):
                # List of objects
                for i, item in enumerate(obj, 1):
                    if level <= 2:
                        md.append(f'### {title or "Item"} {i}')
                    else:
                        md.append(f'**{title or "Item"} {i}:**')
                    # Process dict items inline for cleaner output
                    for k, v in item.items():
                        key_name = k.replace('_', ' ').title()
                        md.append(f'- **{key_name}:** {v}')
                    if i < len(obj):  # Add spacing between items
                        md.append('')
            else:
                # Simple list
                for item in obj:
                    md.append(f'- {item}')
        else:
            # Primitive value
            value_str = str(obj) if obj is not None else 'None'
            if title is None:
                md.append(value_str)
            else:
                md.append(value_str)

        return '\n'.join(md)

    def _dump_markdown_unique(
        self, keys: list[str] | list[str | tuple[str, str]]
    ) -> str:
        data = self.model_dump()
        filtered: dict[str, Any] = {}
        for key in keys:
            if isinstance(key, tuple):
                original_key, renamed_key = key
                if original_key in data:
                    filtered[renamed_key] = data[original_key]
            else:
                if key in data:
                    filtered[key] = data[key]

        # Generate markdown without top-level headers to avoid duplication
        parts = []
        for key, value in filtered.items():
            if value is None:
                continue
            formatted_key = key.replace('_', ' ').title()
            if isinstance(value, (str, int, float, bool)) and len(str(value)) < 200:
                parts.append(f'**{formatted_key}:** {value}')
            else:
                parts.append(self._to_markdown(value, level=2, title=key))

        return '\n'.join(parts)

    def model_dump_markdown_input(self) -> str:
        input_keys, _ = self.get_io_keys()
        return self._dump_markdown_unique(input_keys)

    def model_dump_markdown_output(self) -> str:
        _, output_keys = self.get_io_keys()
        return self._dump_markdown_unique(output_keys)

    # ------------------------------------------------------------------ #
    # Training & preview (JSON or Markdown)
    # ------------------------------------------------------------------ #
    def build_training_data(self, format: str = 'json', indent=None) -> dict[str, Any]:
        """
        Build training data in either JSON (dict for OpenAI-style messages)
        or Markdown (clean format without role prefixes).
        """
        if format == 'json':
            return {
                'messages': [
                    {'role': 'system', 'content': self.get_instruction()},
                    {
                        'role': 'user',
                        'content': self.model_dump_json_input(indent=indent),
                    },
                    {
                        'role': 'assistant',
                        'content': self.model_dump_json_output(indent=indent),
                    },
                ]
            }
        if format == 'markdown':
            system_content = self.get_instruction()

            return {
                'messages': [
                    {'role': 'system', 'content': system_content},
                    {'role': 'user', 'content': self.model_dump_markdown_input()},
                    {'role': 'assistant', 'content': self.model_dump_markdown_output()},
                ]
            }
        raise ValueError("format must be either 'json' or 'markdown'")

    def __str__(self) -> str:
        # Return clean format without explicit role prefixes
        training_data = self.build_training_data(format='markdown')
        messages = training_data['messages']  # type: ignore[index]

        parts = []
        for msg in messages:
            content = msg['content']
            if msg['role'] == 'system' or msg['role'] == 'user':
                parts.append(content)
            elif msg['role'] == 'assistant':
                # Get output keys to determine the main output field name
                _, output_keys = self.get_io_keys()
                main_output = output_keys[0] if output_keys else 'response'
                if isinstance(main_output, tuple):
                    main_output = main_output[1]  # Use renamed key
                title = main_output.replace('_', ' ').title()
                parts.append(f'## {title}\n{content}')

        return '\n\n'.join(parts)

    @classmethod
    def from_messages(cls: type[B], messages: list[dict]) -> B:
        """
        Reconstruct a prompt builder instance from OpenAI-style messages.
        """
        user_msg = next((m for m in messages if m.get('role') == 'user'), None)
        assistant_msg = next(
            (m for m in messages if m.get('role') == 'assistant'), None
        )

        if user_msg is None:
            raise ValueError('No user message found')
        if assistant_msg is None:
            raise ValueError('No assistant message found')

        try:
            user_data = json.loads(user_msg['content'])  # type: ignore[index]
        except Exception as e:
            raise ValueError(f'Invalid user JSON content: {e}') from e

        try:
            assistant_data = json.loads(assistant_msg['content'])  # type: ignore[index]
        except Exception as e:
            raise ValueError(f'Invalid assistant JSON content: {e}') from e

        combined_data = {**user_data, **assistant_data}
        return cast(B, cls(**combined_data))
