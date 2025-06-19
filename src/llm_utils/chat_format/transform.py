from __future__ import annotations
from copy import deepcopy


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
        input_data = deepcopy(input_data)
        if isinstance(input_data, list):
            input_format = "chatlm"
            assert (
                input_data[0].get("role") is not None
            ), "The input format is not recognized. Please specify the input format."
        elif isinstance(input_data, dict):
            input_data = _transform_sharegpt_to_chatml(input_data)
            input_format = "sharegpt"
        elif isinstance(input_data, str):
            assert (
                "<|im_end|>" in input_data
            ), "The input format is not recognized. Please specify the input format."
            input_format = "chatlm"
            parts = input_data.split("<|im_end|>")
            input_data = []
            for part in parts:
                if not part.strip():
                    continue
                role = part.split("<|im_start|>")[1].split("\n")[0]
                content = part.split(f"<|im_start|>{role}\n")[1]
                content = content.split("<|im_end|>")[0]
                input_data.append({"role": role.strip(), "content": content.strip()})

    return input_data


__all__ = [
    "identify_format",
    "_transform_sharegpt_to_chatml",
    "transform_messages",
    "transform_messages_to_chatml",
]
