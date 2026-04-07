from __future__ import annotations

import importlib
from typing import Any


__all__ = [
    "LLM",
    "Qwen3LLM",
    "MOpenAI",
    "LLMSignature",
    "Signature",
    "InputField",
    "OutputField",
    "Input",
    "Output",
    "show_chat",
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
