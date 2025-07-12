from abc import ABC
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

# from openai.pagination import AsyncSyncPage
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from pydantic import BaseModel

from llm_utils.chat_format.display import get_conversation_one_turn

from .async_lm import AsyncLM

# --------------------------------------------------------------------------- #
# type helpers
# --------------------------------------------------------------------------- #
TModel = TypeVar("TModel", bound=BaseModel)
Messages = List[ChatCompletionMessageParam]
LegacyMsgs = List[Dict[str, str]]
RawMsgs = Union[Messages, LegacyMsgs]


# --------------------------------------------------------------------------- #
# Async LLMTask class
# --------------------------------------------------------------------------- #

InputModelType = TypeVar("InputModelType", bound=BaseModel)
OutputModelType = TypeVar("OutputModelType", bound=BaseModel)


class AsyncLLMTask(ABC, Generic[InputModelType, OutputModelType]):
    """
    Async callable wrapper around an AsyncLM endpoint.

    Sub-classes must set:
      • lm              – the async language-model instance
      • InputModel      – a Pydantic input class
      • OutputModel     – a Pydantic output class

    Optional flags:
      • temperature     – float (default 0.6)
      • think           – bool  (if the backend supports "chain-of-thought")
      • add_json_schema – bool  (include schema in the instruction)

    The **docstring** of each sub-class is sent as the LM instruction.
    Example
    ```python
        class DemoTask(AsyncLLMTask):
            "TODO: SYSTEM_PROMPT_INSTURCTION HERE"

            lm = AsyncLM(port=8130, cache=False, model="gpt-3.5-turbo")

            class InputModel(BaseModel):
                text_to_translate:str

            class OutputModel(BaseModel):
                translation:str
                glossary_use:str

            temperature = 0.6
            think=False

        demo_task = DemoTask()
        result = await demo_task({'text_to_translate': 'Translate from english to vietnamese: Hello how are you'})
    ```
    """

    lm: "AsyncLM"
    InputModel: InputModelType
    OutputModel: OutputModelType

    temperature: float = 0.6
    think: bool = False
    add_json_schema: bool = False
    cache: bool = False

    async def __call__(
        self,
        data: BaseModel | dict,
        temperature: float = 0.1,
        cache: bool = False,
        think: Optional[bool] = None,  # if not None, overrides self.think
    ) -> tuple[OutputModelType, List[Dict[str, Any]]]:
        # Get the input and output model types from the generic parameters
        type_args = getattr(self.__class__, "__orig_bases__", None)
        if (
            type_args
            and hasattr(type_args[0], "__args__")
            and len(type_args[0].__args__) >= 2
        ):
            input_model = type_args[0].__args__[0]
            output_model = type_args[0].__args__[1]
        else:
            # Fallback to the old way if type introspection fails
            if (
                not hasattr(self, "InputModel")
                or not hasattr(self, "OutputModel")
                or not hasattr(self, "lm")
            ):
                raise NotImplementedError(
                    f"{self.__class__.__name__} must define lm, InputModel, and OutputModel as class attributes or use proper generic typing."
                )
            input_model = self.InputModel
            output_model = self.OutputModel

        # Ensure input_model is a class before calling
        if isinstance(data, BaseModel):
            item = data
        elif isinstance(input_model, type) and issubclass(input_model, BaseModel):
            item = input_model(**data)
        else:
            raise TypeError("InputModel must be a subclass of BaseModel")

        assert isinstance(output_model, type) and issubclass(output_model, BaseModel), (
            "OutputModel must be a subclass of BaseModel"
        )

        result = await self.lm.parse(
            prompt=item.model_dump_json(),
            instruction=self.__doc__ or "",
            response_model=output_model,
            temperature=temperature or self.temperature,
            think=think if think is not None else self.think,
            add_json_schema_to_instruction=self.add_json_schema,
            cache=self.cache or cache,
        )

        return (
            cast(OutputModelType, result["parsed"]),  # type: ignore
            cast(List[dict], result["messages"]),  # type: ignore
        )

    def generate_training_data(
        self, input_dict: Dict[str, Any], output: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return share gpt like format"""
        system_prompt = self.__doc__ or ""
        user_msg = self.InputModel(**input_dict).model_dump_json()  # type: ignore[attr-defined]
        assistant_msg = self.OutputModel(**output).model_dump_json()  # type: ignore[attr-defined]
        messages = get_conversation_one_turn(
            system_msg=system_prompt, user_msg=user_msg, assistant_msg=assistant_msg
        )
        return {"messages": messages}

    arun = __call__  # alias for compatibility with other LLMTask implementations
