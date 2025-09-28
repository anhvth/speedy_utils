from llm_utils.lm.openai_memoize import MOpenAI
from llm_utils.lm import  LLMTask, AsyncLM, AsyncLLMTask, LLMJudgeBase, ChainOfThought, TranslationEvaluatorJudge, Signature, InputField, OutputField, Input, Output
from llm_utils.vector_cache import VectorCache
from llm_utils.lm.lm_base import get_model_name
from llm_utils.lm.base_prompt_builder import BasePromptBuilder

LLM = LLMTask

# Convenience functions for killing VLLM servers
def kill_all_vllm() -> int:
    """Kill all tracked VLLM server processes. Returns number of processes killed."""
    return LLMTask.kill_all_vllm()

def kill_vllm_on_port(port: int) -> bool:
    """Kill VLLM server on specific port. Returns True if server was killed."""
    return LLMTask.kill_vllm_on_port(port)

from llm_utils.chat_format import (
    build_chatml_input,
    display_chat_messages_as_html,
    display_conversations,
    format_msgs,
    get_conversation_one_turn,
    show_chat,
    show_string_diff,
    transform_messages,
    transform_messages_to_chatml,
)

__all__ = [
    "transform_messages",
    "transform_messages_to_chatml",
    "show_chat",
    "get_conversation_one_turn",
    "show_string_diff",
    "display_conversations",
    "build_chatml_input",
    "format_msgs",
    "display_chat_messages_as_html",
    "AsyncLM",
    "AsyncLLMTask",
    "LLMTask",
    "MOpenAI",
    "get_model_name",
    "VectorCache",
    "BasePromptBuilder",
    "LLM",
    "kill_all_vllm",
    "kill_vllm_on_port",
    "LLMJudgeBase",
    "ChainOfThought",
    "TranslationEvaluatorJudge",
    "Signature",
    "InputField",
    "OutputField",
    "Input",
    "Output",
]
