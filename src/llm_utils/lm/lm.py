# # from ._utils import *
# from typing import (
#     Any,
#     List,
#     Literal,
#     Optional,
#     Type,
#     Union,
#     cast,
# )

# from loguru import logger
# from openai import AuthenticationError, BadRequestError, OpenAI, RateLimitError
# from pydantic import BaseModel
# from speedy_utils import jloads

# # from llm_utils.lm.async_lm.async_llm_task import OutputModelType
# from llm_utils.lm.lm_base import LMBase

# from .async_lm._utils import (
#     LegacyMsgs,
#     Messages,
#     OutputModelType,
#     ParsedOutput,
#     RawMsgs,
# )


# class LM(LMBase):
#     """Unified **sync** language‑model wrapper with optional JSON parsing."""

#     def __init__(
#         self,
#         *,
#         model: Optional[str] = None,
#         response_model: Optional[type[BaseModel]] = None,
#         temperature: float = 0.0,
#         max_tokens: int = 2_000,
#         base_url: Optional[str] = None,
#         api_key: Optional[str] = None,
#         cache: bool = True,
#         ports: Optional[List[int]] = None,
#         top_p: float = 1.0,
#         presence_penalty: float = 0.0,
#         top_k: int = 1,
#         repetition_penalty: float = 1.0,
#         frequency_penalty: Optional[float] = None,
#     ) -> None:

#         if model is None:
#             if base_url is None:
#                 raise ValueError("Either model or base_url must be provided")
#             models = OpenAI(base_url=base_url, api_key=api_key or 'abc').models.list().data
#             assert len(models) == 1, f"Found {len(models)} models, please specify one."
#             model = models[0].id
#             print(f"Using model: {model}")

#         super().__init__(
#             ports=ports,
#             base_url=base_url,
#             cache=cache,
#             api_key=api_key,
#         )

#         # Model behavior options
#         self.response_model = response_model

#         # Store all model-related parameters in model_kwargs
#         self.model_kwargs = dict(
#             model=model,
#             temperature=temperature,
#             max_tokens=max_tokens,
#             top_p=top_p,
#             presence_penalty=presence_penalty,
#         )
#         self.extra_body = dict(
#             top_k=top_k,
#             repetition_penalty=repetition_penalty,
#             frequency_penalty=frequency_penalty,
#         )

#     def _unified_client_call(
#         self,
#         messages: RawMsgs,
#         extra_body: Optional[dict] = None,
#         max_tokens: Optional[int] = None,
#     ) -> dict:
#         """Unified method for all client interactions (caching handled by MOpenAI)."""
#         converted_messages: Messages = (
#             self._convert_messages(cast(LegacyMsgs, messages))
#             if messages and isinstance(messages[0], dict)
#             else cast(Messages, messages)
#         )
#         if max_tokens is not None:
#             self.model_kwargs["max_tokens"] = max_tokens

#         try:
#             # Get completion from API (caching handled by MOpenAI)
#             call_kwargs = {
#                 "messages": converted_messages,
#                 **self.model_kwargs,
#             }
#             if extra_body:
#                 call_kwargs["extra_body"] = extra_body

#             completion = self.client.chat.completions.create(**call_kwargs)

#             if hasattr(completion, "model_dump"):
#                 completion = completion.model_dump()

#         except (AuthenticationError, RateLimitError, BadRequestError) as exc:
#             error_msg = f"OpenAI API error ({type(exc).__name__}): {exc}"
#             logger.error(error_msg)
#             raise

#         return completion

#     def __call__(
#         self,
#         prompt: Optional[str] = None,
#         messages: Optional[RawMsgs] = None,
#         max_tokens: Optional[int] = None,
#     ):  # -> tuple[Any | dict[Any, Any], list[ChatCompletionMessagePar...:# -> tuple[Any | dict[Any, Any], list[ChatCompletionMessagePar...:
#         """Unified sync call for language model, returns (assistant_message.model_dump(), messages)."""
#         if (prompt is None) == (messages is None):
#             raise ValueError("Provide *either* `prompt` or `messages` (but not both).")

#         if prompt is not None:
#             messages = [{"role": "user", "content": prompt}]

#         assert messages is not None

#         openai_msgs: Messages = (
#             self._convert_messages(cast(LegacyMsgs, messages))
#             if isinstance(messages[0], dict)
#             else cast(Messages, messages)
#         )

#         assert self.model_kwargs["model"] is not None, (
#             "Model must be set before making a call."
#         )

#         # Use unified client call
#         raw_response = self._unified_client_call(
#             list(openai_msgs), max_tokens=max_tokens
#         )

#         if hasattr(raw_response, "model_dump"):
#             raw_response = raw_response.model_dump()  # type: ignore

#         # Extract the assistant's message
#         assistant_msg = raw_response["choices"][0]["message"]
#         # Build the full messages list (input + assistant reply)
#         full_messages = list(messages) + [
#             {"role": assistant_msg["role"], "content": assistant_msg["content"]}
#         ]
#         # Return the OpenAI message as model_dump (if available) and the messages list
#         if hasattr(assistant_msg, "model_dump"):
#             msg_dump = assistant_msg.model_dump()
#         else:
#             msg_dump = dict(assistant_msg)
#         return msg_dump, full_messages

#     def parse(
#         self,
#         messages: Messages,
#         response_model: Optional[type[BaseModel]] = None,
#     ) -> ParsedOutput[BaseModel]:
#         """Parse response using OpenAI's native parse API."""
#         # Use provided response_model or fall back to instance default
#         model_to_use = response_model or self.response_model
#         assert model_to_use is not None, "response_model must be provided or set at init."

#         # Use OpenAI's native parse API directly
#         response = self.client.chat.completions.parse(
#             model=self.model_kwargs["model"],
#             messages=messages,
#             response_format=model_to_use,
#             **{k: v for k, v in self.model_kwargs.items() if k != "model"}
#         )
        
#         parsed = response.choices[0].message.parsed
#         completion = response.model_dump() if hasattr(response, "model_dump") else {}
#         full_messages = list(messages) + [
#             {"role": "assistant", "content": parsed}
#         ]

#         return ParsedOutput(
#             messages=full_messages,
#             parsed=cast(BaseModel, parsed),
#             completion=completion,
#             model_kwargs=self.model_kwargs,
#         )



#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         if hasattr(self, "_last_client"):
#             last_client = self._last_client  # type: ignore
#             if hasattr(last_client, "close"):
#                 last_client.close()
#         else:
#             logger.warning("No last client to close")
LM = None