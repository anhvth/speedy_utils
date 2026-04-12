"""Compatibility wrapper for the surviving multi-process APIs."""

from __future__ import annotations

from tqdm import tqdm

from ._multi_process import multi_process
from .common import (
    SPEEDY_RUNNING_PROCESSES,
    ErrorHandlerType,
    ErrorStats,
    cleanup_phantom_workers,
)


__all__ = [
    "SPEEDY_RUNNING_PROCESSES",
    "ErrorStats",
    "ErrorHandlerType",
    "multi_process",
    "cleanup_phantom_workers",
    "tqdm",
]
