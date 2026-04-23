"""datasets_utils - dataset inspection and conversion helpers."""

from __future__ import annotations


__all__ = [
    "convert_to_arrow",
    "extract_tools",
    "load_data",
    "main",
    "normalize_messages",
    "parse_args",
    "print_item",
    "sample_items",
]


def __getattr__(name: str):
    if name == "convert_to_arrow":
        from datasets_utils.convert_to_arrow import convert_to_arrow

        return convert_to_arrow
    if name in __all__:
        import sys as _sys

        mod = _sys.modules["datasets_utils.viz_chat"] = __import__(
            "datasets_utils.viz_chat", fromlist=[name]
        )
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
