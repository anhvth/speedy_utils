"""
Sample test objects for utils_print tests.
These are used to test the different conversion methods that fprint supports.
"""

from dataclasses import dataclass
from typing import Any


class DictConvertibleClass:
    """Sample class with toDict method."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def toDict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "type": "toDict_conversion"}


class ToJsonClass:
    """Sample class with to_dict method."""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "type": "to_dict_conversion"}


@dataclass
class PydanticLikeClass:
    """Sample class with model_dump method (like Pydantic models)."""

    name: str
    value: int

    def model_dump(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value, "type": "model_dump_conversion"}


# Complex nested structure for testing flattening
NESTED_DICT_SAMPLE = {
    "user": {
        "personal": {"name": "Test User", "age": 30},
        "contact": {
            "email": "test@example.com",
            "phone": {"home": "123-456-7890", "work": "098-765-4321"},
        },
    },
    "settings": {"theme": "dark", "notifications": {"email": True, "push": False}},
}
