"""
DSPy-like signature system for structured LLM interactions.

This module provides a declarative way to define LLM input/output schemas
with field descriptions and type annotations.
"""

import inspect
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Type,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, Field


class _FieldProxy:
    """Proxy that stores field information while appearing type-compatible."""

    def __init__(self, field_type: str, desc: str = "", **kwargs):
        self.field_type = field_type  # 'input' or 'output'
        self.desc = desc
        self.kwargs = kwargs


def InputField(desc: str = "", **kwargs) -> Any:
    """Create an input field descriptor."""
    return cast(Any, _FieldProxy("input", desc=desc, **kwargs))


def OutputField(desc: str = "", **kwargs) -> Any:
    """Create an output field descriptor."""
    return cast(Any, _FieldProxy("output", desc=desc, **kwargs))


# Type aliases for cleaner syntax
def Input(desc: str = "", **kwargs) -> Any:
    """Create an input field descriptor that's compatible with type annotations."""
    return InputField(desc=desc, **kwargs)


def Output(desc: str = "", **kwargs) -> Any:
    """Create an output field descriptor that's compatible with type annotations."""
    return OutputField(desc=desc, **kwargs)


class SignatureMeta(type):
    """Metaclass for Signature that processes field annotations."""

    def __new__(cls, name, bases, namespace, **kwargs):
        # Get type hints for this class
        annotations = namespace.get("__annotations__", {})

        # Store field information
        input_fields = {}
        output_fields = {}

        for field_name, field_type in annotations.items():
            field_value = namespace.get(field_name)
            field_desc = None

            # Handle Annotated[Type, Field(...)] syntax using get_origin/get_args
            if get_origin(field_type) is Annotated:
                # Extract args from Annotated type
                args = get_args(field_type)
                if args:
                    # First arg is the actual type
                    field_type = args[0]
                    # Look for _FieldProxy in the metadata
                    for metadata in args[1:]:
                        if isinstance(metadata, _FieldProxy):
                            field_desc = metadata
                            break

            # Handle old syntax with direct assignment
            if field_desc is None and isinstance(field_value, _FieldProxy):
                field_desc = field_value

            # Store field information
            if field_desc and field_desc.field_type == "input":
                input_fields[field_name] = {
                    "type": field_type,
                    "desc": field_desc.desc,
                    **field_desc.kwargs,
                }
            elif field_desc and field_desc.field_type == "output":
                output_fields[field_name] = {
                    "type": field_type,
                    "desc": field_desc.desc,
                    **field_desc.kwargs,
                }

        # Store in class attributes
        namespace["_input_fields"] = input_fields
        namespace["_output_fields"] = output_fields

        return super().__new__(cls, name, bases, namespace)


class Signature(metaclass=SignatureMeta):
    """Base class for defining LLM signatures with input and output fields."""

    _input_fields: dict[str, dict[str, Any]] = {}
    _output_fields: dict[str, dict[str, Any]] = {}

    def __init__(self, **kwargs):
        """Initialize signature with field values."""
        for field_name, value in kwargs.items():
            setattr(self, field_name, value)

    @classmethod
    def get_instruction(cls) -> str:
        """Generate instruction text from docstring and field descriptions."""
        instruction = cls.__doc__ or "Complete the following task."
        instruction = instruction.strip()

        # Add input field descriptions
        if cls._input_fields:
            instruction += "\n\n**Input Fields:**\n"
            for field_name, field_info in cls._input_fields.items():
                desc = field_info.get("desc", "")
                field_type = field_info["type"]
                type_str = getattr(field_type, "__name__", str(field_type))
                instruction += f"- {field_name} ({type_str}): {desc}\n"

        # Add output field descriptions
        if cls._output_fields:
            instruction += "\n**Output Fields:**\n"
            for field_name, field_info in cls._output_fields.items():
                desc = field_info.get("desc", "")
                field_type = field_info["type"]
                type_str = getattr(field_type, "__name__", str(field_type))
                instruction += f"- {field_name} ({type_str}): {desc}\n"

        return instruction

    @classmethod
    def get_input_model(cls) -> type[BaseModel]:
        """Generate Pydantic input model from input fields."""
        if not cls._input_fields:
            raise ValueError(
                f"Signature {cls.__name__} must have at least one input field"
            )

        fields = {}
        annotations = {}

        for field_name, field_info in cls._input_fields.items():
            field_type = field_info["type"]
            desc = field_info.get("desc", "")

            # Create Pydantic field
            field_kwargs = {
                k: v for k, v in field_info.items() if k not in ["type", "desc"]
            }
            if desc:
                field_kwargs["description"] = desc

            fields[field_name] = Field(**field_kwargs) if field_kwargs else Field()
            annotations[field_name] = field_type

        # Create dynamic Pydantic model
        input_model = type(
            f"{cls.__name__}Input",
            (BaseModel,),
            {"__annotations__": annotations, **fields},
        )

        return input_model

    @classmethod
    def get_output_model(cls) -> type[BaseModel]:
        """Generate Pydantic output model from output fields."""
        if not cls._output_fields:
            raise ValueError(
                f"Signature {cls.__name__} must have at least one output field"
            )

        fields = {}
        annotations = {}

        for field_name, field_info in cls._output_fields.items():
            field_type = field_info["type"]
            desc = field_info.get("desc", "")

            # Create Pydantic field
            field_kwargs = {
                k: v for k, v in field_info.items() if k not in ["type", "desc"]
            }
            if desc:
                field_kwargs["description"] = desc

            fields[field_name] = Field(**field_kwargs) if field_kwargs else Field()
            annotations[field_name] = field_type

        # Create dynamic Pydantic model
        output_model = type(
            f"{cls.__name__}Output",
            (BaseModel,),
            {"__annotations__": annotations, **fields},
        )

        return output_model

    def format_input(self, **kwargs) -> str:
        """Format input fields as a string."""
        input_data = {}

        # Collect input field values
        for field_name in self._input_fields:
            if field_name in kwargs:
                input_data[field_name] = kwargs[field_name]
            elif hasattr(self, field_name):
                input_data[field_name] = getattr(self, field_name)

        # Format as key-value pairs
        formatted_lines = []
        for field_name, value in input_data.items():
            field_info = self._input_fields[field_name]
            desc = field_info.get("desc", "")
            if desc:
                formatted_lines.append(f"{field_name} ({desc}): {value}")
            else:
                formatted_lines.append(f"{field_name}: {value}")

        return "\n".join(formatted_lines)


# Export functions for easier importing
__all__ = ["Signature", "InputField", "OutputField", "Input", "Output"]


# Example usage for testing
if __name__ == "__main__":
    # Define a signature like DSPy - using Annotated approach
    class FactJudge(Signature):
        """Judge if the answer is factually correct based on the context."""

        context: Annotated[str, Input("Context for the prediction")]
        question: Annotated[str, Input("Question to be answered")]
        answer: Annotated[str, Input("Answer for the question")]
        factually_correct: Annotated[
            bool, Output("Is the answer factually correct based on the context?")
        ]

    # Alternative syntax still works but will show type warnings
    class FactJudgeOldSyntax(Signature):
        """Judge if the answer is factually correct based on the context."""

        context: str = InputField(desc="Context for the prediction")  # type: ignore
        question: str = InputField(desc="Question to be answered")  # type: ignore
        answer: str = InputField(desc="Answer for the question")  # type: ignore
        factually_correct: bool = OutputField(desc="Is the answer factually correct based on the context?")  # type: ignore

    # Test both signatures
    for judge_class in [FactJudge, FactJudgeOldSyntax]:
        print(f"\n=== Testing {judge_class.__name__} ===")
        print("Instruction:")
        print(judge_class.get_instruction())

        print("\nInput Model:")
        input_model = judge_class.get_input_model()
        print(input_model.model_json_schema())

        print("\nOutput Model:")
        output_model = judge_class.get_output_model()
        print(output_model.model_json_schema())

        # Test instance usage
        judge = judge_class()
        input_text = judge.format_input(
            context="The sky is blue during daytime.",
            question="What color is the sky?",
            answer="Blue",
        )
        print("\nFormatted Input:")
        print(input_text)
