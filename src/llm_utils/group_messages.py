from __future__ import annotations

import random
from typing import Sequence, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

from speedy_utils import multi_thread


def split_indices_by_length(
    lengths: Sequence[int],
    batch_size_by_mean_length: int,
    random_seed: int,
    verbose: bool,
    shuffle: bool,
    mean_length: Optional[int] = None,
) -> list[list[int]]:
    """
    Split indices into batches so that the sum of lengths in each batch does not exceed max_batch_length.
    """
    if mean_length is None:
        mean_length = int(np.mean(lengths))
    max_batch_length: int = mean_length * batch_size_by_mean_length

    r: random.Random = random.Random(random_seed)
    indices: list[int] = list(range(len(lengths)))

    if shuffle:
        r.shuffle(indices)

    batches: list[list[int]] = []
    current_batch: list[int] = []
    current_batch_length: int = 0

    for idx in indices:
        length: int = lengths[idx]
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
        batch_lengths: list[int] = [
            sum(lengths[idx] for idx in batch) for batch in batches
        ]
        desc = pd.Series(batch_lengths).describe()

        table = [
            ["New avg item len", desc["mean"]],
            ["Number groups", len(batches)],
            ["Max length", max_batch_length],
        ]

        print(tabulate(table, headers=["Metric", "Value"], tablefmt="pretty"))

    return batches


def group_messages_by_len(
    messages: Sequence[dict],
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    batch_size: int = 4,
    mean_length: int = 512,
) -> list[dict]:
    """
    Groups messages into batches based on token length and concatenates them.
    """
    if messages is None:
        raise ValueError("messages parameter cannot be None")
    from transformers.models.auto.tokenization_auto import AutoTokenizer # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def create_batches(messages: Sequence[dict]) -> list[dict]:
        def get_token_length(message: dict) -> int:
            ids = tokenizer.apply_chat_template(message["messages"][1:], tokenize=True)
            return len(ids)

        lengths: list[int] = multi_thread(get_token_length, messages, workers=64)
        list_ids: list[list[int]] = split_indices_by_length(
            lengths,
            batch_size,
            random_seed=0,
            verbose=True,
            shuffle=True,
            mean_length=mean_length,
        )
        concatenated_messages: list[dict] = []

        def concatenate_messages(conversations: Sequence[Sequence[dict]]) -> dict:
            system_message = conversations[0][0]
            turns: list[dict] = []
            for conv in conversations:
                turns.extend(conv[1:])
            return {"messages": [system_message] + turns}

        for batch_ids in list_ids:
            if not batch_ids:
                continue
            conversations = [messages[i]["messages"] for i in batch_ids]
            concatenated_messages.append(concatenate_messages(conversations))
        return concatenated_messages

    chunked_messages: list[dict] = create_batches(messages)
    return chunked_messages


__all__ = [
    "split_indices_by_length",
    "group_messages_by_len",
]
