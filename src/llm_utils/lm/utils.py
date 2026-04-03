from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from openai import OpenAI


def get_base_client(
    client=None,
    cache: bool = True,
    api_key: str = "abc",
) -> "OpenAI | Any":
    """Return an OpenAI-compatible client for local use or a provided base URL."""
    from openai import OpenAI

    from llm_utils import MOpenAI

    if client is None:
        return MOpenAI(
            base_url="http://localhost:8000/v1",
            api_key=api_key,
            cache=cache,
        )
    if isinstance(client, int):
        return MOpenAI(
            base_url=f"http://localhost:{client}/v1",
            api_key=api_key,
            cache=cache,
        )
    if isinstance(client, str):
        return MOpenAI(base_url=client, api_key=api_key, cache=cache)
    if isinstance(client, OpenAI):
        return MOpenAI(base_url=client.base_url, api_key=api_key, cache=cache)
    raise ValueError(
        "Invalid client type. Must be OpenAI, port (int), base_url (str), or None."
    )
