from __future__ import annotations

import importlib
from typing import Any


__all__ = [  # type: ignore[misc]
    "LLM",  # type: ignore[misc]
    "Qwen3LLM",  # type: ignore[misc]
    "MOpenAI",  # type: ignore[misc]
    "LLMSignature",  # type: ignore[misc]
    "Signature",  # type: ignore[misc]
    "InputField",  # type: ignore[misc]
    "OutputField",  # type: ignore[misc]
    "Input",  # type: ignore[misc]
    "Output",  # type: ignore[misc]
    "show_chat",  # type: ignore[misc]
    "get_one_turn_conv",
    "turn",
    "msgs_turns",
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "LLM": ("llm_utils.lm", "LLM"),
    "Qwen3LLM": ("llm_utils.lm", "Qwen3LLM"),
    "MOpenAI": ("llm_utils.lm.openai_memoize", "MOpenAI"),
    "LLMSignature": ("llm_utils.lm", "LLMSignature"),
    "Signature": ("llm_utils.lm", "Signature"),
    "InputField": ("llm_utils.lm", "InputField"),
    "OutputField": ("llm_utils.lm", "OutputField"),
    "Input": ("llm_utils.lm", "Input"),
    "Output": ("llm_utils.lm", "Output"),
    "show_chat": ("llm_utils.chat_format", "show_chat"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_LAZY_ATTRS.keys(), *__all__})


def get_one_turn_conv(s: str, u: str, a: str | None = None):
    """Create a one-turn conversation with system, user, and optional assistant messages."""
    conv = [
        {"role": "system", "content": s},
        {"role": "user", "content": u},
    ]
    if a is not None:
        conv.append({"role": "assistant", "content": a})
    return conv


def turn(role, content):
    if role.startswith("a"):
        role = "assistant"
    elif role.startswith("s"):
        role = "system"
    elif role.startswith("u"):
        role = "user"
    return {"role": role, "content": content}


def msgs_turns(*args) -> list[dict]:
    return [turn(r, c) for r, c in args]
