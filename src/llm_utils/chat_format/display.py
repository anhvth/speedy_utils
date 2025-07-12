from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any, Optional

from IPython.display import HTML, display


def show_chat(
    msgs: Any,
    return_html: bool = False,
    file: str = "/tmp/conversation.html",
    theme: str = "default",
) -> Optional[str]:
    """
    Display chat messages as HTML.
    """
    if isinstance(msgs, dict) and "messages" in msgs:
        msgs = msgs["messages"]
    assert isinstance(msgs, list) and all(
        isinstance(msg, dict) and "role" in msg and "content" in msg for msg in msgs
    ), "The input format is not recognized. Please specify the input format."

    if isinstance(msgs[-1], dict) and "choices" in msgs[-1]:
        message = msgs[-1]["choices"][0]["message"]
        reasoning_content = message.get("reasoning_content")
        content = message.get("content", "")
        if reasoning_content:
            content = reasoning_content + "\n" + content
        msgs[-1] = {
            "role": message["role"],
            "content": content,
        }

    themes: dict[str, dict[str, dict[str, str]]] = {
        "default": {
            "system": {"background": "#ffaaaa", "text": "#222222"},  # More red
            "user": {"background": "#f8c57e", "text": "#222222"},  # More orange
            "assistant": {"background": "#9dfebd", "text": "#222222"},  # More green
            "function": {"background": "#eafde7", "text": "#222222"},
            "tool": {"background": "#fde7fa", "text": "#222222"},
            "default": {"background": "#ffffff", "text": "#222222"},
        },
        "light": {
            "system": {"background": "#ff6666", "text": "#000000"},  # More red
            "user": {"background": "#ffd580", "text": "#000000"},  # More orange
            "assistant": {"background": "#80ffb3", "text": "#000000"},  # More green
            "function": {"background": "#AFFFFF", "text": "#000000"},
            "tool": {"background": "#FFAAFF", "text": "#000000"},
            "default": {"background": "#FFFFFF", "text": "#000000"},
        },
        "dark": {
            "system": {"background": "#b22222", "text": "#fffbe7"},  # More red
            "user": {"background": "#ff8800", "text": "#18181b"},  # More orange
            "assistant": {"background": "#22c55e", "text": "#e0ffe0"},  # More green
            "function": {"background": "#134e4a", "text": "#e0fff7"},
            "tool": {"background": "#701a75", "text": "#ffe0fa"},
            "default": {"background": "#18181b", "text": "#f4f4f5"},
        },
    }

    color_scheme = themes.get(theme, themes["default"])

    conversation_html = ""
    for i, message in enumerate(msgs):
        role = message["role"]
        content = message.get("content", "")
        if not content:
            content = ""
        tool_calls = message.get("tool_calls")
        if not content and tool_calls:
            for tool_call in tool_calls:
                tool_call = tool_call["function"]
                name = tool_call["name"]
                args = tool_call["arguments"]
                content += f"Tool: {name}\nArguments: {args}"
        content = content.replace("\n", "<br>")
        content = content.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
        content = content.replace("  ", "&nbsp;&nbsp;")
        content = (
            content.replace("<br>", "TEMP_BR")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("TEMP_BR", "<br>")
        )
        if role in color_scheme:
            background_color = color_scheme[role]["background"]
            text_color = color_scheme[role]["text"]
        else:
            background_color = color_scheme["default"]["background"]
            text_color = color_scheme["default"]["text"]
        if role == "system":
            conversation_html += (
                f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;">'
                f'<strong>System:</strong><br><pre id="system-{i}">{content}</pre></div>'
            )
        elif role == "user":
            conversation_html += (
                f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;">'
                f'<strong>User:</strong><br><pre id="user-{i}">{content}</pre></div>'
            )
        elif role == "assistant":
            conversation_html += (
                f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;">'
                f'<strong>Assistant:</strong><br><pre id="assistant-{i}">{content}</pre></div>'
            )
        elif role == "function":
            conversation_html += (
                f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;">'
                f'<strong>Function:</strong><br><pre id="function-{i}">{content}</pre></div>'
            )
        else:
            conversation_html += (
                f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;">'
                f'<strong>{role}:</strong><br><pre id="{role}-{i}">{content}</pre><br>'
                f"<button onclick=\"copyContent('{role}-{i}')\">Copy</button></div>"
            )
    html: str = f"""
    <html>
    <head>
        <style>
            pre {{
                white-space: pre-wrap;
            }}
        </style>
    </head>
    <body>
        {conversation_html}
        <script>
            function copyContent(elementId) {{
                var element = document.getElementById(elementId);
                var text = element.innerText;
                navigator.clipboard.writeText(text)
                    .then(function() {{
                        alert("Content copied to clipboard!");
                    }})
                    .catch(function(error) {{
                        console.error("Error copying content: ", error);
                    }});
            }}
        </script>
    </body>
    </html>
    """
    if file:
        with open(file, "w") as f:
            f.write(html)
    if return_html:
        return html
    else:
        display(HTML(html))


def get_conversation_one_turn(
    system_msg: Optional[str] = None,
    user_msg: Optional[str] = None,
    assistant_msg: Optional[str] = None,
    assistant_prefix: Optional[str] = None,
    return_format: str = "chatml",
) -> Any:
    """
    Build a one-turn conversation.
    """
    messages: list[dict[str, str]] = []
    if system_msg is not None:
        messages.append({"role": "system", "content": system_msg})
    if user_msg is not None:
        messages.append({"role": "user", "content": user_msg})
    if assistant_msg is not None:
        messages.append({"role": "assistant", "content": assistant_msg})
    if assistant_prefix is not None:
        assert return_format != "chatml", (
            'Change return_format to "text" if you want to use assistant_prefix'
        )
        assert messages[-1]["role"] == "user"
        from .transform import transform_messages

        msg = transform_messages(messages, "chatml", "text", add_generation_prompt=True)
        if not isinstance(msg, str):
            msg = str(msg)
        msg += assistant_prefix
        return msg
    else:
        assert return_format in ["chatml"]
        return messages


def highlight_diff_chars(text1: str, text2: str) -> str:
    """
    Return a string with deletions in red and additions in green.
    """
    matcher = SequenceMatcher(None, text1, text2)
    html: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            html.append(text1[i1:i2])
        elif tag == "replace":
            if i1 != i2:
                html.append(
                    f'<span style="background-color:#ffd6d6; color:#b20000;">{text1[i1:i2]}</span>'
                )
            if j1 != j2:
                html.append(
                    f'<span style="background-color:#d6ffd6; color:#006600;">{text2[j1:j2]}</span>'
                )
        elif tag == "delete":
            html.append(
                f'<span style="background-color:#ffd6d6; color:#b20000;">{text1[i1:i2]}</span>'
            )
        elif tag == "insert":
            html.append(
                f'<span style="background-color:#d6ffd6; color:#006600;">{text2[j1:j2]}</span>'
            )
    return "".join(html)


def show_string_diff(old: str, new: str) -> None:
    """
    Display a one-line visual diff between two strings (old -> new).
    """
    html1 = highlight_diff_chars(old, new)
    display(HTML(html1))


def display_conversations(data1: Any, data2: Any, theme: str = "light") -> None:
    """
    Display two conversations side by side.
    """
    import warnings

    warnings.warn(
        "display_conversations will be deprecated in the next version.",
        DeprecationWarning,
        stacklevel=2,
    )
    html1 = show_chat(data1, return_html=True, theme=theme)
    html2 = show_chat(data2, return_html=True, theme=theme)
    html = f"""
    <html>
    <head>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            td {{
                width: 50%;
                vertical-align: top;
                padding: 10px;
            }}
        </style>
    </head>
    <body>
        <table>
            <tr>
                <td>{html1}</td>
                <td>{html2}</td>
            </tr>
        </table>
    </body>
    </html>
    """
    display(HTML(html))


def display_chat_messages_as_html(*args, **kwargs):
    """
    Use as show_chat and warn about the deprecated function.
    """
    import warnings

    warnings.warn(
        "display_chat_messages_as_html is deprecated, use show_chat instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return show_chat(*args, **kwargs)


__all__ = [
    "show_chat",
    "get_conversation_one_turn",
    "highlight_diff_chars",
    "show_string_diff",
    "display_conversations",
    "display_chat_messages_as_html",
]
