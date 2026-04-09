from __future__ import annotations


ChatMessage = dict[str, str]


def get_one_turn_conv(
    system: str, user: str, assistant: str | None = None
) -> list[ChatMessage]:
    """Create a one-turn conversation with system, user, and assistant messages."""
    messages: list[ChatMessage] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})
    return messages


def _normalize_role(role: str) -> str:
    if role.startswith("a"):
        return "assistant"
    if role.startswith("s"):
        return "system"
    if role.startswith("u"):
        return "user"
    return role


def turn(role: str, content: str) -> ChatMessage:
    return {"role": _normalize_role(role), "content": content}


def msgs_turns(*args: tuple[str, str]) -> list[ChatMessage]:
    return [turn(role, content) for role, content in args]


__all__ = [
    "get_one_turn_conv",
    "turn",
    "msgs_turns",
]
