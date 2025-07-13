import copy
import os
import pathlib
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

from openai.types.chat import (
    ChatCompletionMessageParam,
)
from pydantic import BaseModel
from speedy_utils.all import dump_json_or_pickle, identify

from llm_utils.chat_format.display import get_conversation_one_turn
from llm_utils.lm.async_lm._utils import InputModelType, OutputModelType, ParsedOutput
from llm_utils.lm.async_lm.async_lm import AsyncLM  # Import AsyncLM

TModel = TypeVar("TModel", bound=BaseModel)
Messages = List[ChatCompletionMessageParam]
LegacyMsgs = List[Dict[str, str]]
RawMsgs = Union[Messages, LegacyMsgs]


class AsyncLLMTask(ABC, Generic[InputModelType, OutputModelType]):
    @property
    def lm(self) -> "AsyncLM":
        # Get the output model using the same introspection logic as _base_call
        type_args = getattr(self.__class__, "__orig_bases__", None)
        output_model = None

        if (
            type_args
            and hasattr(type_args[0], "__args__")
            and len(type_args[0].__args__) >= 2
        ):
            output_model = type_args[0].__args__[1]
        else:
            # Fallback to class attribute if it exists
            if hasattr(self, "OutputModel"):
                output_model = self.OutputModel

        return AsyncLM(
            model=os.environ["LLM_MODEL"],
            response_model=output_model,  # type: ignore[assignment]
            # port=os.environ.get("LLM_PORT", 8130),
            # host=os.environ.get("LLM_HOST", "localhost"),
            base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8130/v1"),
        )

    InputModel: InputModelType
    OutputModel: OutputModelType

    temperature: float = 0.6
    think: bool = False
    add_json_schema: bool = True
    cache: bool = False
    collect_data: bool = False  # Set to True to enable data collection for training

    async def _base_call(
        self,
        data: BaseModel | dict,
    ) -> ParsedOutput[OutputModelType]:
        """Base call method that handles the core LM interaction."""
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

        return cast(
            ParsedOutput[OutputModelType],
            await self.lm.parse(
                instruction=self.__doc__ or "",
                prompt=item.model_dump_json(),
            ),
        )

    async def _generate_training_data_with_thinking_mode(
        self,
        input_data: InputModelType,
        expected_response: Optional[OutputModelType] = None,
        label: Optional[str] = None,
        cache_dir: pathlib.Path = pathlib.Path(
            "~/.cache/llm_utils/async_llm_task"
        ).expanduser(),
    ) -> OutputModelType:
        """
        Generate both think and no-think training data for the given input,
        save it to cache, and return the parsed output.
        """
        orig_think = self.think
        self.think = True

        # Call base method to get both parsed and messages
        output = await self._base_call(input_data)
        parsed = output["parsed"]
        messages_think = output["messages"]

        # Format the last assistant message
        system_message = messages_think[0]["content"]
        system_message_no_think = system_message.replace("/think", "/no_think")

        assistant_message = messages_think[-1]["content"]
        assistant_message_no_think = (
            "<think>\n\n</think>\n\n" + assistant_message.split("</think>")[1].strip()
        )
        messages_no_think = copy.deepcopy(messages_think)
        messages_no_think[0]["content"] = system_message_no_think
        messages_no_think[-1]["content"] = assistant_message_no_think
        self.think = orig_think

        # Save training data
        input_id = identify(input_data.model_dump())
        this_class_dir = cache_dir / f"{self.__class__.__name__}/"
        this_class_dir.mkdir(parents=True, exist_ok=True)

        # Combine both think and no-think data in one file
        combined_data = {
            "think_messages": messages_think,
            "no_think_messages": messages_no_think,
            "model_kwargs": output["model_kwargs"],
            "input_data": input_data.model_dump(),
            "label": label,
        }
        if expected_response:
            combined_data["expected_response"] = expected_response.model_dump()

        training_file = this_class_dir / f"{input_id}.json"
        dump_json_or_pickle(combined_data, str(training_file))

        return parsed

    async def __call__(
        self,
        input_data: InputModelType,
        expected_response: Optional[OutputModelType] = None,
        label: Optional[str] = None,  # available during data collection
        **kwargs,
    ) -> OutputModelType:
        """
        Call the LLM task. If collect_data is enabled, automatically save training data.
        """
        if (
            self.collect_data
            or os.getenv("IS_DATA_COLLECTION", "false").lower() == "true"
        ):
            return await self._generate_training_data_with_thinking_mode(
                input_data, expected_response, label
            )
        else:
            output = await self._base_call(input_data)
            return output["parsed"]

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
