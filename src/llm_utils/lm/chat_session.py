from copy import deepcopy
import time  # Add time for possible retries
from typing import (
    Any,
    List,
    Literal,
    Optional,
    TypedDict,
    Type,
    Union,
    TypeVar,
    overload,
)
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str | BaseModel


class ChatSession:
    def __init__(
        self,
        lm: Any,
        system_prompt: Optional[str] = None,
        history: List[Message] = [],
        callback=None,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        self.lm = deepcopy(lm)
        self.history = deepcopy(history)
        self.callback = callback
        self.response_format = response_format
        if system_prompt:
            system_message: Message = {
                "role": "system",
                "content": system_prompt,
            }
            self.history.insert(0, system_message)

    def __len__(self):
        return len(self.history)

    @overload
    def __call__(
        self, text, response_format: Type[T], display=False, max_prev_turns=3, **kwargs
    ) -> T: ...
    @overload
    def __call__(
        self,
        text,
        response_format: None = None,
        display=False,
        max_prev_turns=3,
        **kwargs,
    ) -> str: ...
    def __call__(
        self,
        text,
        response_format: Optional[Type[BaseModel]] = None,
        display=False,
        max_prev_turns=3,
        **kwargs,
    ) -> Union[str, BaseModel]:
        current_response_format = response_format or self.response_format
        self.history.append({"role": "user", "content": text})
        output = self.lm(
            messages=self.parse_history(),
            response_format=current_response_format,
            **kwargs,
        )
        if isinstance(output, BaseModel):
            self.history.append({"role": "assistant", "content": output})
        else:
            assert response_format is None
            self.history.append({"role": "assistant", "content": output})
        if display:
            self.inspect_history(max_prev_turns=max_prev_turns)
        if self.callback:
            self.callback(self, output)
        return output

    def send_message(self, text, **kwargs):
        return self.__call__(text, **kwargs)

    def parse_history(self, indent=None):
        parsed_history = []
        for m in self.history:
            if isinstance(m["content"], str):
                parsed_history.append(m)
            elif isinstance(m["content"], BaseModel):
                parsed_history.append(
                    {
                        "role": m["role"],
                        "content": m["content"].model_dump_json(indent=indent),
                    }
                )
            else:
                raise ValueError(f"Unexpected content type: {type(m['content'])}")
        return parsed_history

    def inspect_history(self, max_prev_turns=3):
        from llm_utils import display_chat_messages_as_html

        h = self.parse_history(indent=2)
        try:
            from IPython.display import clear_output

            clear_output()
            display_chat_messages_as_html(h[-max_prev_turns * 2 :])
        except:
            pass
