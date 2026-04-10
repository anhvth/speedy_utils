from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from openai import OpenAI


def _create_single_client(url: str, api_key: str, cache: bool) -> Any:
    """Create a single MOpenAI client for a given URL."""
    from .openai_memoize import MOpenAI

    return MOpenAI(base_url=url, api_key=api_key, cache=cache)


def get_base_client(
    client=None,
    cache: bool = True,
    api_key: str = "abc",
) -> "OpenAI | Any | list[Any]":
    """Return an OpenAI-compatible client for local use or a provided base URL.

    When client is a list, returns a list of clients for load balancing.
    """
    from openai import OpenAI
    from .openai_memoize import MOpenAI

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
    if isinstance(client, list):
        clients = []
        for item in client:
            if isinstance(item, int):
                clients.append(
                    MOpenAI(
                        base_url=f"http://localhost:{item}/v1",
                        api_key=api_key,
                        cache=cache,
                    )
                )
            elif isinstance(item, str):
                clients.append(MOpenAI(base_url=item, api_key=api_key, cache=cache))
            elif isinstance(item, OpenAI):
                clients.append(
                    MOpenAI(base_url=item.base_url, api_key=api_key, cache=cache)
                )
            else:
                raise ValueError(
                    f"Invalid item type in client list: {type(item)}. "
                    "Must be OpenAI, port (int), or base_url (str)."
                )
        return clients
    raise ValueError(
        "Invalid client type. Must be OpenAI, port (int), base_url (str), list of these, or None."
    )
