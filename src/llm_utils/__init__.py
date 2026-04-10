from __future__ import annotations

import importlib

from .chat_format import show_chat
from .utils import get_one_turn_conv, msgs_turns, turn

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "LLM": ("llm_utils.lm", "LLM"),
    "Qwen3LLM": ("llm_utils.lm", "Qwen3LLM"),
    "MOpenAI": ("llm_utils.lm.openai_memoize", "MOpenAI"),
    "LLMSignature": ("llm_utils.lm.llm_signature", "LLMSignature"),
    "Signature": ("llm_utils.lm.signature", "Signature"),
    "InputField": ("llm_utils.lm.signature", "InputField"),
    "OutputField": ("llm_utils.lm.signature", "OutputField"),
    "Input": ("llm_utils.lm.signature", "Input"),
    "Output": ("llm_utils.lm.signature", "Output"),
}


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "get_one_turn_conv",
    "turn",
    "msgs_turns",
]
