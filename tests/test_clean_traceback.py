"""Tests for the public clean_traceback decorator."""

import pytest
from pydantic import BaseModel, ValidationError

from speedy_utils import clean_traceback


class Payload(BaseModel):
    value: int


def test_clean_traceback_lets_validation_error_through():
    """ValidationError is a recoverable item error: keep its type for callers."""

    @clean_traceback
    def parse(data: dict) -> Payload:
        return Payload(**data)

    with pytest.raises(ValidationError):
        parse({"value": "not-an-int"})
