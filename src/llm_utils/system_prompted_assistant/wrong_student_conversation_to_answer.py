"""Generate a corrected assistant response from an SDD teacher prompt."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from llm_utils.lm import LLM


DEFAULT_SYSTEM_PROMPT = """You repair an incorrect assistant turn.

The user supplies a fully rendered teacher prompt. It contains the original
conversation and may contain an <env_feedback> block with:
- <previous_assistant_response>: the rejected response;
- <feedback>: why it was rejected and how the assistant should behave.

Generate the expected assistant continuation only. Follow the tools, language,
format, and constraints in the teacher prompt. Apply the feedback, but never
mention the evaluator, feedback, teacher prompt, or rejected response. Do not
add commentary before or after the answer.

Examples:

Teacher prompt: User asks "What is 2+2?". Feedback says not to call a tool.
Expected continuation: 4

Teacher prompt: User asks to book a flight but gives no departure city.
Feedback says to request the missing departure city.
Expected continuation: What city will you be departing from?

Teacher prompt: User asks for Paris weather. Feedback says to call weather_get.
Expected continuation:
<tool_call>
<function=weather_get>
<parameter=city>
Paris
</parameter>
</function>
</tool_call>
"""


def _decode_teacher_ids(tokenizer: Any, teacher_ids: Sequence[int]) -> str:
    if isinstance(teacher_ids, (str, bytes)) or not teacher_ids:
        raise ValueError("teacher_ids must be a non-empty integer sequence")
    if any(type(token_id) is not int or token_id < 0 for token_id in teacher_ids):
        raise ValueError("teacher_ids must contain only non-negative integers")
    try:
        return str(tokenizer.decode(list(teacher_ids), skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode(list(teacher_ids)))


class WrongStudentConversationToAnswer(LLM):
    """Use privileged SDD context to generate the corrected assistant turn."""

    def __init__(
        self,
        *args: Any,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        tokenizer: Any | None = None,
        tokenizer_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name

    def _get_tokenizer(self) -> Any:
        if self.tokenizer is not None:
            return self.tokenizer
        if not self.tokenizer_name:
            raise ValueError(
                "item has teacher_ids but no tokenizer or tokenizer_name was provided"
            )
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=True,
        )
        return self.tokenizer

    def build_user_message(self, item: Mapping[str, Any]) -> str:
        teacher_prompt = item.get("teacher_prompt")
        if isinstance(teacher_prompt, str) and teacher_prompt.strip():
            return teacher_prompt

        teacher_ids = item.get("teacher_ids")
        if not isinstance(teacher_ids, Sequence):
            raise ValueError(
                "item must contain a non-empty teacher_prompt or teacher_ids"
            )
        return _decode_teacher_ids(self._get_tokenizer(), teacher_ids)

    def answer(self, item: Mapping[str, Any], **runtime_kwargs: Any) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_user_message(item)},
        ]
        runtime_kwargs.setdefault("enable_thinking", False)
        response = self(messages, **runtime_kwargs)
        content = (
            response.get("content")
            if isinstance(response, Mapping)
            else getattr(response, "content", None)
        )
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM returned an empty assistant response")
        return content
