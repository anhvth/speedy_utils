from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any


# Lazy import IPython.display
if TYPE_CHECKING:
    from IPython.display import HTML, display


def _preprocess_as_json(content: str) -> str:
    """Preprocess content as JSON with proper formatting."""
    try:
        parsed = json.loads(content)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return content


def _preprocess_as_markdown(content: str) -> str:
    """Preprocess content as markdown with proper formatting."""
    lines = content.split('\n')
    processed_lines = []

    for line in lines:
        # Convert **bold** to span with bold styling
        while '**' in line:
            first_pos = line.find('**')
            if first_pos == -1:
                break
            second_pos = line.find('**', first_pos + 2)
            if second_pos == -1:
                break
            before = line[:first_pos]
            bold_text = line[first_pos + 2 : second_pos]
            after = line[second_pos + 2 :]
            line = f'{before}<span style="font-weight: bold;">{bold_text}</span>{after}'

        # Convert *italic* to span with italic styling
        while '*' in line and line.count('*') >= 2:
            first_pos = line.find('*')
            if first_pos == -1:
                break
            second_pos = line.find('*', first_pos + 1)
            if second_pos == -1:
                break
            before = line[:first_pos]
            italic_text = line[first_pos + 1 : second_pos]
            after = line[second_pos + 1 :]
            line = (
                f'{before}<span style="font-style: italic;">{italic_text}</span>{after}'
            )

        # Convert # headers to bold headers
        if line.strip().startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            header_text = line.lstrip('# ').strip()
            line = f'<span style="font-weight: bold; font-size: 1.{min(4, level)}em;">{header_text}</span>'

        processed_lines.append(line)

    return '\n'.join(processed_lines)


def _truncate_text(text: str, max_length: int, head_ratio: float = 0.3) -> str:
    """
    Truncate text if it exceeds max_length, showing head and tail with skip indicator.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation
        head_ratio: Ratio of max_length to show at the head (default 0.3)

    Returns:
        Original text if within limit, otherwise truncated with [SKIP n chars] indicator
    """
    if len(text) <= max_length:
        return text

    head_len = int(max_length * head_ratio)
    tail_len = max_length - head_len
    skip_len = len(text) - head_len - tail_len

    return f'{text[:head_len]}\n...[SKIP {skip_len} chars]...\n{text[-tail_len:]}'


def _format_reasoning_content(
    reasoning: str, max_reasoning_length: int | None = None
) -> str:
    """
    Format reasoning content with <think> tags.

    Args:
        reasoning: The reasoning content
        max_reasoning_length: Max length before truncation (None = no truncation)

    Returns:
        Formatted reasoning with <think> tags
    """
    if max_reasoning_length is not None:
        reasoning = _truncate_text(reasoning, max_reasoning_length)
    return f'<think>\n{reasoning}\n</think>'


def _escape_html(content: str) -> str:
    """Escape HTML special characters and convert whitespace for display."""
    return (
        content.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('\n', '<br>')
        .replace('\t', '&nbsp;&nbsp;&nbsp;&nbsp;')
        .replace('  ', '&nbsp;&nbsp;')
    )


def _is_notebook() -> bool:
    """Detect if running in a notebook environment."""
    try:
        from IPython.core.getipython import get_ipython

        ipython = get_ipython()
        return ipython is not None and 'IPKernelApp' in ipython.config
    except (ImportError, AttributeError):
        return False


def _message_value(msg: Any, key: str, default: Any = None) -> Any:
    """Read a message field from a mapping or OpenAI-style object."""
    if isinstance(msg, dict):
        return msg.get(key, default)
    return getattr(msg, key, default)


def _split_embedded_think(content: str) -> tuple[str | None, str]:
    """Extract reasoning and answer text from an embedded <think> block."""
    if "<think>" not in content:
        return None, content

    start = content.find("<think>")
    body = content[start + len("<think>") :]
    end = body.find("</think>")
    if end == -1:
        reasoning = body.strip()
        return reasoning or None, ""

    reasoning = body[:end].strip()
    answer = body[end + len("</think>") :].strip()
    return reasoning or None, answer


# Color configurations
ROLE_COLORS_HTML = {
    'system': 'red',
    'user': 'darkorange',
    'assistant': 'green',
}

ROLE_COLORS_TERMINAL = {
    'system': '\033[91m',  # Red
    'user': '\033[38;5;208m',  # Orange
    'assistant': '\033[92m',  # Green
}

ROLE_LABELS = {
    'system': 'System Instruction:',
    'user': 'User:',
    'assistant': 'Assistant:',
}

TERMINAL_RESET = '\033[0m'
TERMINAL_BOLD = '\033[1m'
TERMINAL_GRAY = '\033[90m'
TERMINAL_DIM = '\033[2m'  # Dim text for reasoning

# HTML colors
HTML_REASONING_COLOR = '#AAAAAA'  # Lighter gray for better readability


def _build_assistant_content_parts(
    msg: dict[str, Any], max_reasoning_length: int | None
) -> tuple[str | None, str]:
    """
    Build display content parts for assistant message.

    Returns:
        Tuple of (reasoning_formatted, answer_content)
        reasoning_formatted is None if no reasoning present
    """
    content = _message_value(msg, 'content', '')
    if content is None:
        content = ''
    reasoning = _message_value(msg, 'reasoning_content')

    if reasoning is None and isinstance(content, str):
        embedded_reasoning, embedded_answer = _split_embedded_think(content)
        if embedded_reasoning is not None:
            reasoning = embedded_reasoning
            content = embedded_answer

    if reasoning:
        formatted_reasoning = _format_reasoning_content(reasoning, max_reasoning_length)
        return formatted_reasoning, str(content)

    return None, str(content)


def _chat_html(messages: list[dict[str, Any]], max_reasoning_length: int | None) -> str:
    """Build chat messages HTML."""
    html_parts = [
        "<div style='font-family:monospace; line-height:1.6em; white-space:pre-wrap;'>"
    ]
    separator = "<div style='color:#888; margin:0.5em 0;'>───────────────────────────────────────────────────</div>"

    for i, msg in enumerate(messages):
        role = str(_message_value(msg, 'role', 'unknown')).lower()
        color = ROLE_COLORS_HTML.get(role, 'black')
        label = ROLE_LABELS.get(role, f'{role.capitalize()}:')

        if role == 'assistant':
            reasoning, answer = _build_assistant_content_parts(
                msg, max_reasoning_length
            )
            html_parts.append(
                f"<div><strong style='color:{color}'>{label}</strong><br>"
            )
            if reasoning:
                escaped_reasoning = _escape_html(reasoning)
                html_parts.append(
                    f"<span style='color:{HTML_REASONING_COLOR}'>{escaped_reasoning}</span><br><br>"
                )
            if answer:
                escaped_answer = _escape_html(answer)
                html_parts.append(
                    f"<span style='color:{color}'>{escaped_answer}</span>"
                )
            html_parts.append('</div>')
        else:
            content = _message_value(msg, 'content', '') or ''
            escaped_content = _escape_html(content)
            html_parts.append(
                f"<div style='color:{color}'><strong>{label}</strong><br>{escaped_content}</div>"
            )

        if i < len(messages) - 1:
            html_parts.append(separator)

    html_parts.append('</div>')
    return ''.join(html_parts)


def _show_chat_html(
    messages: list[dict[str, Any]], max_reasoning_length: int | None
) -> None:
    """Display chat messages as HTML in notebook."""
    from IPython.display import HTML, display

    display(HTML(_chat_html(messages, max_reasoning_length)))


def _show_multi_conversations_html(
    conversations: list[list[dict[str, Any]]],
    max_reasoning_length: int | None,
    titles: list[str] | None = None,
) -> None:
    """Display multiple conversations as CSS-only notebook tabs."""
    from html import escape

    from IPython.display import HTML, display

    if not conversations:
        display(HTML("<div>No conversations to display.</div>"))
        return

    tabs_id = f"llm-utils-chat-tabs-{id(conversations)}"
    radios: list[str] = []
    labels: list[str] = []
    panels: list[str] = []
    css_rules: list[str] = []

    for i, messages in enumerate(conversations):
        tab_id = f"{tabs_id}-tab-{i}"
        checked = " checked" if i == 0 else ""
        if titles is not None and i < len(titles):
            title = titles[i]
        else:
            first_user = next(
                (
                    str(_message_value(msg, "content", ""))
                    for msg in messages
                    if str(_message_value(msg, "role", "")).lower() == "user"
                ),
                f"Conversation {i + 1}",
            )
            title = first_user[:42]

        radios.append(
            f'<input class="chat-tab-radio" type="radio" '
            f'name="{tabs_id}" id="{tab_id}"{checked}>'
        )
        labels.append(
            f'<label class="chat-tab-button" for="{tab_id}">'
            f"{i + 1}. {escape(title)}</label>"
        )
        panels.append(
            f'<section class="chat-tab-panel panel-{i}">'
            f"{_chat_html(messages, max_reasoning_length)}</section>"
        )
        css_rules.append(
            f"#{tab_id}:checked ~ .chat-tab-panels .panel-{i} "
            "{ display: block; }"
        )
        css_rules.append(
            f"#{tab_id}:checked ~ .chat-tab-buttons label[for='{tab_id}'] "
            "{ background: #18181b; color: #fafafa; border-color: #18181b; }"
        )

    display(
        HTML(
            f"""
<div id="{tabs_id}" class="llm-utils-chat-tabs">
  {''.join(radios)}
  <div class="chat-tab-buttons">{''.join(labels)}</div>
  <div class="chat-tab-panels">{''.join(panels)}</div>
</div>
<style>
  #{tabs_id} {{
    font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
  #{tabs_id} .chat-tab-radio {{
    position: absolute;
    opacity: 0;
    pointer-events: none;
  }}
  #{tabs_id} .chat-tab-buttons {{
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin: 0 0 10px;
  }}
  #{tabs_id} .chat-tab-button {{
    border: 1px solid #d4d4d8;
    background: #f4f4f5;
    color: #27272a;
    border-radius: 6px;
    padding: 5px 8px;
    cursor: pointer;
    max-width: 280px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}
  #{tabs_id} .chat-tab-button:hover {{
    background: #e4e4e7;
  }}
  #{tabs_id} .chat-tab-panel {{
    display: none;
  }}
  {' '.join(css_rules)}
</style>
"""
        )
    )


def _show_multi_conversations_terminal(
    conversations: list[list[dict[str, Any]]],
    max_reasoning_length: int | None,
    titles: list[str] | None = None,
) -> None:
    """Display multiple conversations in terminal."""
    for i, messages in enumerate(conversations):
        title = titles[i] if titles is not None and i < len(titles) else None
        header = title or f"Conversation {i + 1}"
        print(f"{TERMINAL_BOLD}=== {header} ==={TERMINAL_RESET}")
        _show_chat_terminal(messages, max_reasoning_length)
        if i < len(conversations) - 1:
            print()


def _show_chat_terminal(
    messages: list[dict[str, Any]], max_reasoning_length: int | None
) -> None:
    """Display chat messages with ANSI colors in terminal."""
    separator = f'{TERMINAL_GRAY}─────────────────────────────────────────────────────────{TERMINAL_RESET}'

    for i, msg in enumerate(messages):
        role = str(_message_value(msg, 'role', 'unknown')).lower()
        color = ROLE_COLORS_TERMINAL.get(role, '')
        label = ROLE_LABELS.get(role, f'{role.capitalize()}:')

        print(f'{color}{TERMINAL_BOLD}{label}{TERMINAL_RESET}')

        if role == 'assistant':
            reasoning, answer = _build_assistant_content_parts(
                msg, max_reasoning_length
            )
            if reasoning:
                # Use lighter gray without dim for better readability
                print(f'\033[38;5;246m{reasoning.strip()}{TERMINAL_RESET}')
                if answer:
                    print()  # Blank line between reasoning and answer
            if answer:
                print(f'{color}{answer.strip()}{TERMINAL_RESET}')
        else:
            content = _message_value(msg, 'content', '') or ''
            print(f'{color}{content}{TERMINAL_RESET}')

        if i < len(messages) - 1:
            print(separator)


def _normalize_multi_conversations(
    conversations: list[list[dict[str, Any]]] | list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Normalize multi-conversation input."""
    if not conversations:
        return []
    first = conversations[0]
    if isinstance(first, dict):
        return [conversations]  # type: ignore[list-item]
    return conversations  # type: ignore[return-value]


def show_chat(
    messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
    max_reasoning_length: int | None = 2000,
    *,
    mode: str = "single",
    titles: list[str] | None = None,
) -> None:
    """
    Display chat messages with colored formatting.

    Automatically detects notebook vs terminal environment and formats accordingly.
    Handles reasoning_content in assistant messages, formatting it with <think> tags.

    Args:
        messages: List of message dicts with 'role', 'content', and optionally 'reasoning_content'.
            For multi-conversation modes, pass a list of conversations, where each
            conversation is a list of message dicts.
        max_reasoning_length: Max chars for reasoning before truncation (None = no limit)
        mode: Display mode. Use "single" for one conversation or "multi-convs"
            for tabbed multi-chat display in notebooks.
        titles: Optional tab titles for multi-conversation modes.

    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi!", "reasoning_content": "User greeted me..."},
        ... ]
        >>> show_chat(messages)
    """
    if mode == "multi-convs":
        conversations = _normalize_multi_conversations(messages)
        if _is_notebook():
            _show_multi_conversations_html(
                conversations, max_reasoning_length, titles=titles
            )
        else:
            _show_multi_conversations_terminal(
                conversations, max_reasoning_length, titles=titles
        )
        return

    if mode != "single":
        raise ValueError(
            f"Unsupported show_chat mode: {mode!r}. "
            "Use 'single' or 'multi-convs'."
        )

    if _is_notebook():
        _show_chat_html(messages, max_reasoning_length)  # type: ignore[arg-type]
    else:
        _show_chat_terminal(messages, max_reasoning_length)  # type: ignore[arg-type]


def get_conversation_one_turn(
    system_msg: str | None = None,
    user_msg: str | None = None,
    assistant_msg: str | None = None,
    assistant_prefix: str | None = None,
    return_format: str = 'chatml',
) -> Any:
    """Build a one-turn conversation."""
    messages: list[dict[str, str]] = []

    if system_msg is not None:
        messages.append({'role': 'system', 'content': system_msg})
    if user_msg is not None:
        messages.append({'role': 'user', 'content': user_msg})
    if assistant_msg is not None:
        messages.append({'role': 'assistant', 'content': assistant_msg})

    if assistant_prefix is not None:
        if return_format == 'chatml':
            raise ValueError('Change return_format to "text" to use assistant_prefix')
        if not messages or messages[-1]['role'] != 'user':
            raise ValueError(
                'Last message must be from user when using assistant_prefix'
            )

        from .transform import transform_messages

        msg = transform_messages(messages, 'chatml', 'text', add_generation_prompt=True)
        return str(msg) + assistant_prefix

    if return_format != 'chatml':
        raise ValueError(f'Unsupported return_format: {return_format}')

    return messages


def highlight_diff_chars(text1: str, text2: str) -> str:
    """Return a string with deletions in red and additions in green."""
    matcher = SequenceMatcher(None, text1, text2)
    html_parts: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            html_parts.append(text1[i1:i2])
        elif tag == 'replace':
            if i1 != i2:
                html_parts.append(
                    f'<span style="background-color:#ffd6d6; color:#b20000;">{text1[i1:i2]}</span>'
                )
            if j1 != j2:
                html_parts.append(
                    f'<span style="background-color:#d6ffd6; color:#006600;">{text2[j1:j2]}</span>'
                )
        elif tag == 'delete':
            html_parts.append(
                f'<span style="background-color:#ffd6d6; color:#b20000;">{text1[i1:i2]}</span>'
            )
        elif tag == 'insert':
            html_parts.append(
                f'<span style="background-color:#d6ffd6; color:#006600;">{text2[j1:j2]}</span>'
            )

    return ''.join(html_parts)


def show_string_diff(old: str, new: str) -> None:
    """Display a visual diff between two strings (old -> new)."""
    from IPython.display import HTML, display

    display(HTML(highlight_diff_chars(old, new)))


def display_conversations(data1: Any, data2: Any) -> None:
    """Display two conversations side by side. Deprecated."""
    import warnings

    warnings.warn(
        'display_conversations is deprecated and will be removed.',
        DeprecationWarning,
        stacklevel=2,
    )
    print('=== Conversation 1 ===')
    show_chat(data1)
    print('\n=== Conversation 2 ===')
    show_chat(data2)


def display_chat_messages_as_html(
    messages: list[dict[str, Any]] | list[list[dict[str, Any]]],
    max_reasoning_length: int | None = 2000,
    *,
    mode: str = "single",
    titles: list[str] | None = None,
) -> None:
    """Deprecated alias for show_chat."""
    import warnings

    warnings.warn(
        'display_chat_messages_as_html is deprecated, use show_chat instead.',
        DeprecationWarning,
        stacklevel=2,
    )
    return show_chat(
        messages,
        max_reasoning_length,
        mode=mode,
        titles=titles,
    )


__all__ = [
    'show_chat',
    'get_conversation_one_turn',
    'highlight_diff_chars',
    'show_string_diff',
    'display_conversations',
    'display_chat_messages_as_html',
]
