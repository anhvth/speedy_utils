from llm_utils.chat_format.transform import transform_messages


def main():
    # 1. Define ShareGPT data
    sharegpt_data = {
        "conversations": [
            {"from": "human", "value": "What is the capital of France?"},
            {"from": "gpt", "value": "The capital of France is Paris."}
        ]
    }
    print("Original ShareGPT:", sharegpt_data)

    # 2. Convert to ChatML
    chatml_data = transform_messages(sharegpt_data, frm="sharegpt", to="chatml")
    print("\nConverted to ChatML:", chatml_data)

    # 3. Convert to Text (Prompt)
    text_data = transform_messages(chatml_data, frm="chatml", to="text")
    print("\nConverted to Text Prompt:\n", text_data)

    # 4. Convert to Simulated Chat
    sim_chat = transform_messages(chatml_data, frm="chatml", to="simulated_chat")
    print("\nConverted to Simulated Chat:\n", sim_chat)

if __name__ == "__main__":
    main()
