from __future__ import annotations

import importlib
from typing import Any


__all__ = [
    # chat_format
    "transform_messages",
    "transform_messages_to_chatml",
    "show_chat",
    "get_conversation_one_turn",
    "show_string_diff",
    "display_conversations",
    "build_chatml_input",
    "format_msgs",
    "display_chat_messages_as_html",
    # core LLM APIs (lazy)
    "AsyncLM",
    "AsyncLM_Qwen3",
    "AsyncLM_GLM5",
    "AsyncLM_DeepSeekR1",
    "AsyncLLMTask",
    "LLM",
    "LLMRay",
    "MOpenAI",
    "get_model_name",
    "VectorCache",
    "BasePromptBuilder",
    "kill_all_vllm",
    "kill_vllm_on_port",
    "LLMSignature",
    "Signature",
    "InputField",
    "OutputField",
    "Input",
    "Output",
    "LLM_TASK",  # Alias for LLM class
]

_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # ray wrapper
    "LLMRay": ("llm_utils.llm_ray", "LLMRay"),
    # lm exports
    "LLM": ("llm_utils.lm", "LLM"),
    "AsyncLLMTask": ("llm_utils.lm", "AsyncLLMTask"),
    "AsyncLM": ("llm_utils.lm", "AsyncLM"),
    "AsyncLM_Qwen3": ("llm_utils.lm", "AsyncLM_Qwen3"),
    "AsyncLM_GLM5": ("llm_utils.lm", "AsyncLM_GLM5"),
    "AsyncLM_DeepSeekR1": ("llm_utils.lm", "AsyncLM_DeepSeekR1"),
    "Input": ("llm_utils.lm", "Input"),
    "InputField": ("llm_utils.lm", "InputField"),
    "LLMSignature": ("llm_utils.lm", "LLMSignature"),
    "Output": ("llm_utils.lm", "Output"),
    "OutputField": ("llm_utils.lm", "OutputField"),
    "Signature": ("llm_utils.lm", "Signature"),
    # helpers
    "BasePromptBuilder": ("llm_utils.lm.base_prompt_builder", "BasePromptBuilder"),
    "get_model_name": ("llm_utils.lm.lm_base", "get_model_name"),
    "MOpenAI": ("llm_utils.lm.openai_memoize", "MOpenAI"),
    "VectorCache": ("llm_utils.vector_cache", "VectorCache"),
    # chat_format exports
    "build_chatml_input": ("llm_utils.chat_format", "build_chatml_input"),
    "display_chat_messages_as_html": (
        "llm_utils.chat_format",
        "display_chat_messages_as_html",
    ),
    "display_conversations": ("llm_utils.chat_format", "display_conversations"),
    "format_msgs": ("llm_utils.chat_format", "format_msgs"),
    "get_conversation_one_turn": ("llm_utils.chat_format", "get_conversation_one_turn"),
    "show_chat": ("llm_utils.chat_format", "show_chat"),
    "show_string_diff": ("llm_utils.chat_format", "show_string_diff"),
    "transform_messages": ("llm_utils.chat_format", "transform_messages"),
    "transform_messages_to_chatml": (
        "llm_utils.chat_format",
        "transform_messages_to_chatml",
    ),
}


def __getattr__(name: str) -> Any:
    if name == "LLM_TASK":
        value = __getattr__("LLM")
        globals()["LLM_TASK"] = value
        return value

    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *_LAZY_ATTRS.keys(), "LLM_TASK"})


def kill_all_vllm() -> int:
    """Kill all tracked VLLM server processes. Returns number of processes killed."""
    from llm_utils.lm import LLM

    return LLM.kill_all_vllm()


def kill_vllm_on_port(port: int) -> bool:
    """Kill VLLM server on specific port. Returns True if server was killed."""
    from llm_utils.lm import LLM

    return LLM.kill_vllm_on_port(port)
