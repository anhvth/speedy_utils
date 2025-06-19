from typing import Any, Optional


from llm_utils.lm.sync_lm import LM, RawMsgs


class LMJson(LM):
    "Regex-based reasoning wrapper for LM."

    def __init__(
        self,
        model: str | None = None,
        *,
        temperature: float = 0.0,
        max_tokens: int = 2_000,
        host: str = "localhost",
        port: Optional[int | str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: bool = True,
        **openai_kwargs: Any,
    ) -> None:
        """
        Initialize the LMJson instance.

        Args:
            model (str | None): The model name to use.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum number of tokens to generate.
            host (str): Host for the API.
            port (int | str, optional): Port for the API.
            base_url (str, optional): Base URL for the API.
            api_key (str, optional): API key for authentication.
            cache (bool): Whether to cache responses.
            **openai_kwargs: Additional OpenAI parameters.
        """
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            host=host,
            port=port,
            base_url=base_url,
            api_key=api_key,
            cache=cache,
            **openai_kwargs,
        )

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[RawMsgs] = None,
        cache: Optional[bool] = None,
        max_tokens: Optional[int] = None,
        return_openai_response: bool = False,
        **kwargs: Any,
    ):

        output = super().__call__(
            prompt=prompt,
            messages=messages,
            response_format=str,
            cache=cache,
            max_tokens=max_tokens,
            return_openai_response=return_openai_response,
            **kwargs,
        )
        return output
