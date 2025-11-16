"""
Efficient embedding caching system using vLLM for offline embeddings.

This package provides a fast, SQLite-backed caching layer for text embeddings,
supporting both OpenAI API and local models via vLLM.

Classes:
    VectorCache: Main class for embedding computation and caching

Example:
    # Using local model
    cache = VectorCache("Qwen/Qwen3-Embedding-0.6B")
    embeddings = cache.embeds(["Hello world", "How are you?"])

    # Using OpenAI API
    cache = VectorCache("https://api.openai.com/v1")
    embeddings = cache.embeds(["Hello world", "How are you?"])
"""

from .core import VectorCache
from .utils import estimate_cache_size, get_default_cache_path, validate_model_name


__version__ = "0.1.0"
__author__ = "AnhVTH <anhvth.226@gmail.com>"
__all__ = [
    "VectorCache",
    "get_default_cache_path",
    "validate_model_name",
    "estimate_cache_size",
]
