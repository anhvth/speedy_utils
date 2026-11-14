"""Shared HuggingFace tokenizer loader for the datasets_utils CLIs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_tokenizer_cache: dict[str, Any] = {}


def get_tokenizer(tokenizer_name: str) -> Any:
    """Load and cache a tokenizer by HF model id or local path."""
    cached = _tokenizer_cache.get(tokenizer_name)
    if cached is not None:
        return cached

    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required to decode tokenized rows. "
            "Install with: pip install 'speedy-utils[transformers]'"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(tokenizer_name).expanduser()), trust_remote_code=True
    )
    _tokenizer_cache[tokenizer_name] = tokenizer
    return tokenizer
