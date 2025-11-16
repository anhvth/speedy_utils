from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any, Optional

from IPython.display import HTML, display


def _preprocess_as_json(content: str) -> str:
    """
    Preprocess content as JSON with proper formatting and syntax highlighting.
    """
    try:
        # Try to parse and reformat JSON
        parsed = json.loads(content)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, return as-is
        return content


def _preprocess_as_markdown(content: str) -> str:
    """
    Preprocess content as markdown with proper formatting.
    """
    # Basic markdown preprocessing - convert common patterns
    lines = content.split("\n")
    processed_lines = []

    for line in lines:
        # Convert **bold** to span with bold styling
        while "**" in line:
            first_pos = line.find("**")
            if first_pos != -1:
                second_pos = line.find("**", first_pos + 2)
                if second_pos != -1:
                    before = line[:first_pos]
                    bold_text = line[first_pos + 2 : second_pos]
                    after = line[second_pos + 2 :]
                    line = f'{before}<span style="font-weight: bold;">{bold_text}</span>{after}'
                else:
                    break
            else:
                break

        # Convert *italic* to span with italic styling
        while "*" in line and line.count("*") >= 2:
            first_pos = line.find("*")
            if first_pos != -1:
                second_pos = line.find("*", first_pos + 1)
                if second_pos != -1:
                    before = line[:first_pos]
                    italic_text = line[first_pos + 1 : second_pos]
                    after = line[second_pos + 1 :]
                    line = f'{before}<span style="font-style: italic;">{italic_text}</span>{after}'
                else:
                    break
            else:
                break

        # Convert # headers to bold headers
        if line.strip().startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            header_text = line.lstrip("# ").strip()
            line = f'<span style="font-weight: bold; font-size: 1.{min(4, level)}em;">{header_text}</span>'

        processed_lines.append(line)

    return "\n".join(processed_lines)


def show_chat(
    msgs: Any,
    return_html: bool = False,
    file: str = "/tmp/conversation.html",
    theme: str = "default",
    as_markdown: bool = False,
    as_json: bool = False,
) -> str | None:
    """
    Display chat messages as HTML.

    Args:
        msgs: Chat messages in various formats
        return_html: If True, return HTML string instead of displaying
        file: Path to save HTML file
        theme: Color theme ('default', 'light', 'dark')
        as_markdown: If True, preprocess content as markdown
        as_json: If True, preprocess content as JSON
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

        # Preprocess content based on format options
        if as_json:
            content = _preprocess_as_json(content)
        elif as_markdown:
            content = _preprocess_as_markdown(content)

        # Handle HTML escaping differently for markdown vs regular content
        if as_markdown:
            # For markdown, preserve HTML tags but escape other characters carefully
            content = content.replace("\n", "<br>")
            content = content.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
            content = content.replace("  ", "&nbsp;&nbsp;")
            # Don't escape < and > for markdown since we want to preserve our span tags
        else:
            # Regular escaping for non-markdown content
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

        # Choose container based on whether we have markdown formatting
        content_container = "div" if as_markdown else "pre"
        container_style = 'style="white-space: pre-wrap;"' if as_markdown else ""

        if role == "system":
            conversation_html += (
                f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;">'
                f'<strong>System:</strong><br><{content_container} id="system-{i}" {container_style}>{content}</{content_container}></div>'
            )
        elif role == "user":
            conversation_html += (
                f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;">'
                f'<strong>User:</strong><br><{content_container} id="user-{i}" {container_style}>{content}</{content_container}></div>'
            )
        elif role == "assistant":
            conversation_html += (
                f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;">'
                f'<strong>Assistant:</strong><br><{content_container} id="assistant-{i}" {container_style}>{content}</{content_container}></div>'
            )
        elif role == "function":
            conversation_html += (
                f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;">'
                f'<strong>Function:</strong><br><{content_container} id="function-{i}" {container_style}>{content}</{content_container}></div>'
            )
        else:
            conversation_html += (
                f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;">'
                f'<strong>{role}:</strong><br><{content_container} id="{role}-{i}" {container_style}>{content}</{content_container}><br>'
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
    display(HTML(html))
    return None


def get_conversation_one_turn(
    system_msg: str | None = None,
    user_msg: str | None = None,
    assistant_msg: str | None = None,
    assistant_prefix: str | None = None,
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
        assert (
            return_format != "chatml"
        ), 'Change return_format to "text" if you want to use assistant_prefix'
        assert messages[-1]["role"] == "user"
        from .transform import transform_messages

        msg = transform_messages(messages, "chatml", "text", add_generation_prompt=True)
        if not isinstance(msg, str):
            msg = str(msg)
        msg += assistant_prefix
        return msg
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


def show_chat_v2(messages: list[dict[str, str]]):
    """
    Print only content of messages in different colors:
    system -> red, user -> orange, assistant -> green.
    Automatically detects notebook environment and uses appropriate display.
    """
    # Detect if running in a notebook environment
    try:
        from IPython.core.getipython import get_ipython

        ipython = get_ipython()
        is_notebook = ipython is not None and "IPKernelApp" in ipython.config
    except (ImportError, AttributeError):
        is_notebook = False

    if is_notebook:
        # Use HTML display in notebook
        from IPython.display import HTML, display

        role_colors = {
            "system": "red",
            "user": "darkorange",
            "assistant": "green",
        }

        role_labels = {
            "system": "System Instruction:",
            "user": "User:",
            "assistant": "Assistant:",
        }

        html = "<div style='font-family:monospace; line-height:1.6em; white-space:pre-wrap;'>"
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown").lower()
            content = msg.get("content", "")
            # Escape HTML characters
            content = (
                content.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
                .replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")
                .replace("  ", "&nbsp;&nbsp;")
            )
            color = role_colors.get(role, "black")
            label = role_labels.get(role, f"{role.capitalize()}:")
            html += f"<div style='color:{color}'><strong>{label}</strong><br>{content}</div>"
            # Add separator except after last message
            if i < len(messages) - 1:
                html += "<div style='color:#888; margin:0.5em 0;'>───────────────────────────────────────────────────</div>"
        html += "</div>"

        display(HTML(html))
    else:
        # Use normal terminal printing with ANSI colors
        role_colors = {
            "system": "\033[91m",  # Red
            "user": "\033[38;5;208m",  # Orange
            "assistant": "\033[92m",  # Green
        }
        reset = "\033[0m"
        separator_color = "\033[90m"  # Gray
        bold = "\033[1m"

        role_labels = {
            "system": "System Instruction:",
            "user": "User:",
            "assistant": "Assistant:",
        }

        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown").lower()
            content = msg.get("content", "")
            color = role_colors.get(role, "")
            label = role_labels.get(role, f"{role.capitalize()}:")
            print(f"{color}{bold}{label}{reset}")
            print(f"{color}{content}{reset}")
            # Add separator except after last message
            if i < len(messages) - 1:
                print(
                    f"{separator_color}─────────────────────────────────────────────────────────{reset}"
                )


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
