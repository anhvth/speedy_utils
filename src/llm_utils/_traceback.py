"""Deprecated: ``speedy_utils.common.utils_error`` owns the traceback renderer.

The decorator stays here and resolves that module on the first decorated call,
because importing it eagerly would drag all of ``speedy_utils`` (and its
``rich`` imports) into ``import llm_utils``.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


def clean_traceback(func: F) -> F:
    """Delegate to the shared ``clean_traceback``, imported on the first call."""
    owner_wrapper: Any = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal owner_wrapper
        if owner_wrapper is None:
            from speedy_utils.common.utils_error import clean_traceback as owner

            owner_wrapper = owner(func)
        return owner_wrapper(*args, **kwargs)

    return wrapper  # type: ignore[return-value]
