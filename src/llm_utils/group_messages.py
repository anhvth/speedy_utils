import random
from typing import Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

from speedy_utils import multi_thread


def split_indices_by_length(
    lengths: list[int],
    batch_size_by_mean_length: int,
    random_seed: int,
    verbose: bool,
    shuffle: bool,
    mean_length: Optional[int] = None,
) -> list[list[int]]:
    if mean_length is None:
        mean_length = int(np.mean(lengths))
    max_batch_length = mean_length * batch_size_by_mean_length

    r = random.Random(random_seed)
    indices = list(range(len(lengths)))

    if shuffle:
        r.shuffle(indices)

    batches = []
    current_batch = []
    current_batch_length = 0

    for idx in indices:
        length = lengths[idx]
        if current_batch_length + length <= max_batch_length:
            current_batch.append(idx)
            current_batch_length += length
        else:
            batches.append(current_batch)
            current_batch = [idx]
            current_batch_length = length

    if current_batch:
        batches.append(current_batch)

    if verbose:
        batch_lengths = [sum(lengths[idx] for idx in batch) for batch in batches]
        desc = pd.Series(batch_lengths).describe()

        table = [
            ["New avg item len", desc["mean"]],
            ["Number groups", len(batches)],
            ["Max length", max_batch_length],
        ]

        print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))

    return batches


def group_messages_by_len(
    messages, model_name="Qwen/Qwen2.5-7B-Instruct", batch_size=4, mean_length=512
):
    """
    Groups a list of messages into batches based on token length and concatenates them.
    Args:
        messages (list[dict]): OpenAI message format, each dict should contain a "messages" key with a list of messages. ensure the system prompt are shared.
        model_name (str): The name of the model to use for tokenization. Default is "Qwen/Qwen2.5-7B-Instruct".
        batch_size (int): The number of messages to include in each batch. Default is 4.
        mean_length (int): The mean length of tokens for each batch. Default is 512.
    Returns:
        list: A list of concatenated message dictionaries, where each dictionary contains a "messages" key with the grouped messages.
    Raises:
        ValueError: If the messages parameter is None.
    """
    if messages is None:
        raise ValueError("messages parameter cannot be None")
    from transformers.models.auto.tokenization_auto import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_batches(messages):
        def get_token_length(message):
            ids = tokenizer.apply_chat_template(message["messages"][1:], tokenize=True)
            return len(ids)

        # lengths = [get_token_length(msg) for msg in messages]
        lengths = multi_thread(get_token_length, messages, workers=64)
        list_ids = split_indices_by_length(
            lengths,
            batch_size,
            random_seed=0,
            verbose=True,
            shuffle=True,
            mean_length=mean_length,
        )
        concatenated_messages = []

        def concatenate_messages(conversations):
            system_message = conversations[0][0]
            turns = []
            for conv in conversations:
                turns.extend(conv[1:])
            return {"messages": [system_message] + turns}

        for batch_ids in list_ids:
            if not batch_ids:
                continue
            conversations = [messages[i]["messages"] for i in batch_ids]
            concatenated_messages.append(concatenate_messages(conversations))
        return concatenated_messages

    chunked_messages = create_batches(messages)
    return chunked_messages

__all__ = [
    "split_indices_by_length",
    "group_messages_by_len",
]
