from __future__ import annotations

import base64
import mimetypes
from pathlib import Path


ChatMessage = dict[str, object]


def _image_to_url(image: str) -> str:
    """Return a data-URI for local paths, or pass through public URLs unchanged."""
    path = Path(image)
    if path.exists():
        mime, _ = mimetypes.guess_type(str(path))
        if mime is None:
            mime = "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode()
        return f"data:{mime};base64,{data}"
    # assume it's already a public URL
    return image


def _build_user_content(text: str, images: list[str] | None) -> object:
    """Build the user content block, multimodal when images are provided."""
    if not images:
        return text
    parts: list[dict[str, object]] = []
    if text:
        parts.append({"type": "text", "text": text})
    for img in images:
        parts.append({"type": "image_url", "image_url": {"url": _image_to_url(img)}})
    return parts


def get_one_turn_conv(
    system: str,
    user: str = "",
    assistant: str | None = None,
    images: list[str] | None = None,
) -> list[ChatMessage]:
    """Create a one-turn conversation with optional multimodal images.

    Args:
        system: System prompt text.
        user: User message text.
        assistant: Optional prefilled assistant response.
        images: Optional list of image paths (local files) or public URLs.
                Local files are base64-encoded into data-URIs automatically.
    """
    messages: list[ChatMessage] = [
        {"role": "system", "content": system},
        {"role": "user", "content": _build_user_content(user, images)},
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
