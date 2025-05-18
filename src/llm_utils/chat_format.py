from copy import deepcopy
from typing import Dict, List, Literal, Union

from IPython.display import HTML, Markdown, display
from loguru import logger


def identify_format(item):
    if isinstance(item, list) and "role" in item[0]:
        return "chatml"
    if isinstance(item, dict):
        if "conversations" in item:
            return "sharegpt"
    raise ValueError(
        f"The format of the item is not recognized. \n{type(item)=}, \n{item=}"
    )


def _transform_sharegpt_to_chatml(
    item, default_system_message="You are a helpful assistant.", print_msg=False
):
    # if isinstance(item, list):
    # return [_transform_sharegpt_to_chatml(item) for item in item]
    assert isinstance(
        item, dict
    ), "The item is not in the correct format. Please check the format of the item."

    messages = []
    system_msg = item.get("system", "")
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    elif default_system_message:
        messages.append({"role": "system", "content": default_system_message})
    conversations = item.get("conversations", [])
    if hasattr(conversations, "tolist"):
        conversations = conversations.tolist()
    # import ipdb; ipdb.set_trace()
    assert conversations, "The item does not have any conversations."
    for conversation in item.get("conversations", []):
        role = conversation["from"]
        content = conversation["value"]
        messages.append({"role": role, "content": content})

    return messages


def transform_messages(
    item,
    frm="chatml",
    to="text",
    add_generation_prompt=True,
    tokenizer=None,
    assistant_prefix=None,
):
    assert to in [
        "chatml",
        "text",
        "sharegpt",
        "simulated_chat",
    ], "The output format is not recognized. Please specify the output format."
    item = deepcopy(item)

    if tokenizer is not None:
        assert frm == "chatml", "Tokenizer is only supported for chatml format."
        prompt = tokenizer.apply_chat_template(
            item, tokenize=False, add_generation_prompt=True
        )
        assert isinstance(prompt, str), "Prompt must be a string."
        if assistant_prefix:
            prompt += f"{assistant_prefix}"
        return prompt

    if frm != to:
        # convert item to chatml format
        chatml_messages = transform_messages_to_chatml(item, input_format=frm)
        if to == "sharegpt":
            if chatml_messages[0]["role"] == "system":
                system_message = chatml_messages[0]["content"]
                ret = {"conversations": [], "system": system_message.strip()}
                for message in chatml_messages[1:]:
                    ret["conversations"].append(
                        {"from": message["role"], "value": message["content"]}
                    )
            else:
                system_message = "You are a helpful assistant."
                ret = {"conversations": [], "system": system_message.strip()}
                for message in chatml_messages:
                    ret["conversations"].append(
                        {"from": message["role"], "value": message["content"]}
                    )
            return ret
        elif to == "chatml":
            return _transform_sharegpt_to_chatml(item)
        elif to == "text":
            text = ""
            for turn in chatml_messages:
                text += f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
            if add_generation_prompt:
                text += "<|im_start|>assistant\n"
            return text
        elif to == "simulated_chat":
            text = "<role> Given the simulated chat, you are the assistant. Lets continue the conversation. \n\n"
            for turn in chatml_messages:
                prefix = {
                    "user": "Human",
                    "assistant": "AI",
                    "system": "System",
                    "function": "Function",
                }.get(turn["role"])
                text += f"{prefix}: {turn['content'].strip()}\n\n"
            if add_generation_prompt:
                text += "AI: [continue the conversation here]"
            return text
        else:
            raise ValueError(f"{to} is not supported.")

    else:
        return item


def transform_messages_to_chatml(input_data, input_format="auto"):
    if input_format == "auto":
        input_data = raw_data = deepcopy(input_data)
        if isinstance(input_data, list):
            input_format = "chatlm"
            assert (
                input_data[0].get("role") is not None
            ), "The input format is not recognized. Please specify the input format."
        elif isinstance(input_data, dict):
            input_data = _transform_sharegpt_to_chatml(input_data)
            input_format = "sharegpt"
        elif isinstance(input_data, str):
            # assume it has format <|im_start|>role\n content<|im_end|> use regex to parse
            assert (
                "<|im_end|>" in input_data
            ), "The input format is not recognized. Please specify the input format."
            input_format = "chatlm"
            parts = input_data.split("<|im_end|>")
            # for each part, split by <|im_start|> to get role and content
            input_data = []
            for part in parts:
                if not part.strip():
                    continue
                role = part.split("<|im_start|>")[1].split("\n")[0]
                # content is after |>role\n
                content = part.split(f"<|im_start|>{role}\n")[1]
                content = content.split("<|im_end|>")[0]
                input_data.append({"role": role.strip(), "content": content.strip()})

    return input_data


def display_chat_messages_as_html(
    msgs, return_html=False, file="/tmp/conversation.html", theme="light"
):
    if isinstance(msgs, dict) and "messages" in msgs:
        msgs = msgs["messages"]

    # ensure is a list of dict each dict has role and content
    assert isinstance(msgs, list) and all(
        isinstance(msg, dict) and "role" in msg and "content" in msg for msg in msgs
    ), "The input format is not recognized. Please specify the input format."

    color_scheme = {
        "system": {
            "background": "#FFAAAA",
            "text": "#000000",
        },  # Light red background, black text
        "user": {
            "background": "#AAFFAA",
            "text": "#000000",
        },  # Light green background, black text
        "assistant": {
            "background": "#AAAAFF",
            "text": "#000000",
        },  # Light blue background, black text
        "function": {
            "background": "#AFFFFF",
            "text": "#000000",
        },  # Light yellow background, black text
        "default": {
            "background": "#FFFFFF",
            "text": "#000000",
        },  # White background, black text
        "tool": {"background": "#FFAAFF", "text": "#000000"},
    }

    conversation_html = ""
    for i, message in enumerate(msgs):
        role = message["role"]
        content = message.get("content", "")
        if not content:
            content = ""

        tool_calls = message.get("tool_calls")
        if not content and tool_calls:
            # each tool call comes with name, and args

            for tool_call in tool_calls:
                tool_call = tool_call["function"]
                name = tool_call["name"]
                args = tool_call["arguments"]
                content += "Tool: " + name + "\n" + "Arguments: " + str(args)

        # Replace newlines with <br> tags
        content = content.replace("\n", "<br>")

        # Replace tabs with &nbsp; entities
        content = content.replace("\t", "&nbsp;&nbsp;&nbsp;&nbsp;")

        # Replace multiple consecutive spaces with &nbsp; entities
        content = content.replace("  ", "&nbsp;&nbsp;")
        # keep html tag without escaping
        # content = content.replace('&lt;', '<')

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
            conversation_html += f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;"><strong>System:</strong><br><pre id="system-{i}">{content}</pre></div>'
        elif role == "user":
            conversation_html += f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;"><strong>User:</strong><br><pre id="user-{i}">{content}</pre></div>'
        elif role == "assistant":
            conversation_html += f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;"><strong>Assistant:</strong><br><pre id="assistant-{i}">{content}</pre></div>'
        elif role == "function":
            conversation_html += f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;"><strong>Function:</strong><br><pre id="function-{i}">{content}</pre></div>'
        else:
            # logger.warning(f"Unknown role: {role}")
            conversation_html += f'<div style="background-color: {background_color}; color: {text_color}; padding: 10px; margin-bottom: 10px;"><strong>{role}:</strong><br><pre id="{role}-{i}">{content}</pre><br><button onclick="copyContent(\'{role}-{i}\')">Copy</button></div>'

    html = f"""
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
    system_msg=None,
    user_msg=None,
    assistant_msg=None,
    assistant_prefix=None,
    return_format="chatml",
):
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})
    if assistant_msg:
        messages.append({"role": "assistant", "content": assistant_msg})
    if assistant_prefix is not None:
        assert (
            return_format != "chatml"
        ), "Change return_format to 'text' if you want to use assistant_prefix"
        assert messages[-1]["role"] == "user"
        msg = transform_messages(messages, "chatml", "text", add_generation_prompt=True)
        msg += assistant_prefix
        return msg
    else:
        assert return_format in ["chatml"]
        return messages


from difflib import ndiff

from IPython.display import HTML


def display_diff_two_string(text1, text2):
    # Split the texts into lines
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()

    # Perform the diff
    diff = list(ndiff(lines1, lines2))

    # Create the HTML table
    table_rows = []
    for line in diff:
        if line.startswith("- "):
            table_rows.append(
                f'<tr><td style="background-color: #FFCCCB;">{line[2:]}</td><td></td></tr>'
            )
        elif line.startswith("+ "):
            table_rows.append(
                f'<tr><td></td><td style="background-color: #CCFFCC;">{line[2:]}</td></tr>'
            )
        elif line.startswith("? "):
            continue
        else:
            table_rows.append(f"<tr><td>{line}</td><td>{line}</td></tr>")

    table_html = '<table style="width: 100%; border-collapse: collapse;">'
    table_html += '<tr><th style="width: 50%; text-align: left;">Text 1</th><th style="width: 50%; text-align: left;">Text 2</th></tr>'
    table_html += "".join(table_rows)
    table_html += "</table>"

    # Display the HTML table
    display(HTML(table_html))


def display_conversations(data1, data2, theme="light"):
    html1 = display_chat_messages_as_html(data1, return_html=True, theme=theme)
    html2 = display_chat_messages_as_html(data2, return_html=True, theme=theme)

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


from typing import Callable, Dict, List


def build_chatml_input(template: str, params: List[str]) -> Callable:
    def formator(**kwargs) -> List[List[Dict[str, str]]]:
        system_msg = kwargs.get("system_msg", None)
        # remove system
        kwargs.pop("system_msg", None)
        # Ensure all required parameters are present in kwargs
        for param in params:
            if param not in kwargs:
                raise ValueError(f"Missing parameter: {param}")

        # Use the **kwargs directly in the format method
        content = template.format(**kwargs)
        msgs = []
        if system_msg:
            msgs += [{"role": "system", "content": system_msg}]
        msgs += [{"role": "user", "content": content}]
        return msgs

    return formator


def _color_text(text, color_code):
    """Helper function to color text based on the provided ANSI color code."""
    return f"\033[{color_code}m{text}\033[0m"


def format_msgs(messages):
    """Formats the role and content of a list of messages into a string."""
    messages = transform_messages_to_chatml(messages)
    output = []

    for msg in messages:
        role = msg.get("role", "unknown").lower()
        content = msg.get("content", "").strip()
        output.append(f"{role.capitalize()}:\t{content}")
        output.append("---")

    return "\n".join(output)


__all__ = [
    "transform_messages",
    "transform_messages_to_chatml",
    "display_chat_messages_as_html",
    "get_conversation_one_turn",
    "display_diff_two_string",
    "display_conversations",
    "build_chatml_input",
    "format_msgs",
]
