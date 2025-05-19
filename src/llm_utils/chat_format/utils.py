from __future__ import annotations
from typing import List, Dict, Callable


def build_chatml_input(template: str, params: List[str]) -> Callable:
    def formator(**kwargs) -> List[List[Dict[str, str]]]:
        system_msg = kwargs.get("system_msg", None)
        kwargs.pop("system_msg", None)
        for param in params:
            if param not in kwargs:
                raise ValueError(f"Missing parameter: {param}")
        content = template.format(**kwargs)
        msgs = []
        if system_msg:
            msgs += [{"role": "system", "content": system_msg}]
        msgs += [{"role": "user", "content": content}]
        return msgs

    return formator


def _color_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"


def format_msgs(messages):
    from .transform import transform_messages_to_chatml

    messages = transform_messages_to_chatml(messages)
    output = []
    for msg in messages:
        role = msg.get("role", "unknown").lower()
        content = msg.get("content", "").strip()
        output.append(f"{role.capitalize()}:\t{content}")
        output.append("---")
    return "\n".join(output)


__all__ = [
    "build_chatml_input",
    "_color_text",
    "format_msgs",
]
