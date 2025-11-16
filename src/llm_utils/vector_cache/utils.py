"""Utility functions for the embed_cache package."""

import os
from typing import Optional


def get_default_cache_path() -> str:
    """Get the default cache path based on environment."""
    cache_dir = os.getenv("EMBED_CACHE_DIR", ".")
    return os.path.join(cache_dir, "embed_cache.sqlite")


def validate_model_name(model_name: str) -> bool:
    """Validate if a model name is supported."""
    # Check if it's a URL
    if model_name.startswith("http"):
        return True

    # Check if it's a valid model path/name
    supported_prefixes = [
        "Qwen/",
        "sentence-transformers/",
        "BAAI/",
        "intfloat/",
        "microsoft/",
        "nvidia/",
    ]

    return any(
        model_name.startswith(prefix) for prefix in supported_prefixes
    ) or os.path.exists(model_name)


def estimate_cache_size(num_texts: int, embedding_dim: int = 1024) -> str:
    """Estimate cache size for given number of texts."""
    # Rough estimate: hash (40 bytes) + text (avg 100 bytes) + embedding (embedding_dim * 4 bytes)
    bytes_per_entry = 40 + 100 + (embedding_dim * 4)
    total_bytes = num_texts * bytes_per_entry

    if total_bytes < 1024:
        return f"{total_bytes} bytes"
    if total_bytes < 1024 * 1024:
        return f"{total_bytes / 1024:.1f} KB"
    if total_bytes < 1024 * 1024 * 1024:
        return f"{total_bytes / (1024 * 1024):.1f} MB"
    return f"{total_bytes / (1024 * 1024 * 1024):.1f} GB"
