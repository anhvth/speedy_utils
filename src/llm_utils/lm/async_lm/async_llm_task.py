"""
Async LLM Task module for handling language model interactions with structured input/output.
"""

import copy
import pathlib
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union, cast

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from pytest import Cache
from speedy_utils import jdumps
from speedy_utils.all import dump_json_or_pickle, identify

from llm_utils.chat_format.display import get_conversation_one_turn
from llm_utils.lm.async_lm._utils import InputModelType, OutputModelType, ParsedOutput
from llm_utils.lm.async_lm.async_lm import AsyncLM

# Type aliases for better readability
TModel = TypeVar("TModel", bound=BaseModel)
Messages = List[ChatCompletionMessageParam]
LegacyMsgs = List[Dict[str, str]]
RawMsgs = Union[Messages, LegacyMsgs]

# Default configuration constants


@dataclass
class LMConfiguration:
    """Configuration class for language model parameters."""

    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    host: Optional[str] = None
    port: Optional[Union[int, str]] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    cache: Optional[bool] = True
    think: Optional[Literal[True, False]] = None
    add_json_schema_to_instruction: Optional[bool] = None
    use_beta: Optional[bool] = False
    ports: Optional[List[int]] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "host": self.host,
            "port": self.port,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "cache": self.cache,
            "think": self.think,
            "add_json_schema_to_instruction": self.add_json_schema_to_instruction,
            "use_beta": self.use_beta,
            "ports": self.ports,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
        }


class AsyncLLMTask(ABC, Generic[InputModelType, OutputModelType]):
    """
    Abstract base class for asynchronous language model tasks with structured I/O.

    This class provides a framework for creating LLM tasks with strongly typed
    input and output models, automatic training data collection, and support
    for both thinking and non-thinking modes.

    Type Parameters:
        InputModelType: Pydantic model type for task input
        OutputModelType: Pydantic model type for task output
    """

    InputModel: InputModelType
    OutputModel: OutputModelType

    # default class attributes for configuration
    DEFAULT_MODEL: Optional[str] = None
    DEFAULT_CACHE_DIR: Optional[pathlib.Path] = None
    DEFAULT_TEMPERATURE: Optional[float] = None
    DEFAULT_MAX_TOKENS: Optional[int] = None
    DEFAULT_HOST: Optional[str] = None
    DEFAULT_PORT: Optional[Union[int, str]] = None
    DEFAULT_TOP_P: Optional[float] = None
    DEFAULT_PRESENCE_PENALTY: Optional[float] = None
    DEFAULT_TOP_K: Optional[int] = None
    DEFAULT_REPETITION_PENALTY: Optional[float] = None
    DEFAULT_CACHE: Optional[bool] = True
    DEFAULT_THINK: Optional[Literal[True, False]] = None
    DEFAULT_PORTS: Optional[List[int]] = None
    DEFAULT_USE_BETA: Optional[bool] = False
    DEFAULT_ADD_JSON_SCHEMA_TO_INSTRUCTION: Optional[bool] = True
    DEFAULT_COLLECT_DATA: Optional[bool] = None
    DEFAULT_BASE_URL: Optional[str] = None
    DEFAULT_API_KEY: Optional[str] = None

    IS_DATA_COLLECTION: bool = False

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        host: Optional[str] = None,
        port: Optional[Union[int, str]] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        cache: Optional[bool] = None,
        think: Optional[Literal[True, False]] = None,
        add_json_schema_to_instruction: Optional[bool] = None,
        use_beta: Optional[bool] = None,
        ports: Optional[List[int]] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> None:
        """
        Initialize the AsyncLLMTask with language model configuration.

        All arguments are optional; defaults are taken from class attributes if not provided.
        """
        self._config = LMConfiguration(
            model=model if model is not None else self.DEFAULT_MODEL,
            temperature=temperature
            if temperature is not None
            else self.DEFAULT_TEMPERATURE,
            max_tokens=max_tokens
            if max_tokens is not None
            else self.DEFAULT_MAX_TOKENS,
            host=host if host is not None else self.DEFAULT_HOST,
            port=port if port is not None else self.DEFAULT_PORT,
            base_url=base_url if base_url is not None else self.DEFAULT_BASE_URL,
            api_key=api_key if api_key is not None else self.DEFAULT_API_KEY,
            cache=cache if cache is not None else self.DEFAULT_CACHE,
            think=think if think is not None else self.DEFAULT_THINK,
            add_json_schema_to_instruction=add_json_schema_to_instruction
            if add_json_schema_to_instruction is not None
            else self.DEFAULT_ADD_JSON_SCHEMA_TO_INSTRUCTION,
            use_beta=use_beta if use_beta is not None else self.DEFAULT_USE_BETA,
            ports=ports if ports is not None else self.DEFAULT_PORTS,
            top_p=top_p if top_p is not None else self.DEFAULT_TOP_P,
            presence_penalty=presence_penalty
            if presence_penalty is not None
            else self.DEFAULT_PRESENCE_PENALTY,
            top_k=top_k if top_k is not None else self.DEFAULT_TOP_K,
            repetition_penalty=repetition_penalty
            if repetition_penalty is not None
            else self.DEFAULT_REPETITION_PENALTY,
        )
        self._lm: Optional[AsyncLM] = None

    @property
    def lm(self) -> AsyncLM:
        """
        Lazy-loaded AsyncLM instance with proper configuration.

        Returns:
            Configured AsyncLM instance for this task
        """
        if self._lm is None:
            self._lm = AsyncLM(
                **self._config.to_dict(),
                response_model=self._get_output_model_type(),
            )
        return self._lm

    def _get_output_model_type(self) -> type[OutputModelType]:
        """
        Extract the output model type from generic type arguments.

        Returns:
            The OutputModelType class

        Raises:
            TypeError: If output model type cannot be determined
        """
        # Try to get type from generic base classes
        orig_bases = getattr(self.__class__, "__orig_bases__", None)
        if (
            orig_bases
            and hasattr(orig_bases[0], "__args__")
            and len(orig_bases[0].__args__) >= 2
        ):
            return orig_bases[0].__args__[1]

        # Fallback to class attribute
        if hasattr(self, "OutputModel"):
            return self.OutputModel  # type: ignore

        raise TypeError(
            f"{self.__class__.__name__} must define OutputModel as a class attribute "
            "or use proper generic typing with AsyncLLMTask[InputModel, OutputModel]"
        )

    def _get_input_model_type(self) -> type[InputModelType]:
        """
        Extract the input model type from generic type arguments.

        Returns:
            The InputModelType class

        Raises:
            TypeError: If input model type cannot be determined
        """
        # Try to get type from generic base classes
        orig_bases = getattr(self.__class__, "__orig_bases__", None)
        if (
            orig_bases
            and hasattr(orig_bases[0], "__args__")
            and len(orig_bases[0].__args__) >= 2
        ):
            return orig_bases[0].__args__[0]

        raise TypeError(
            f"{self.__class__.__name__} must define InputModel as a class attribute "
            "or use proper generic typing with AsyncLLMTask[InputModel, OutputModel]"
        )

    def _validate_and_convert_input(self, data: Union[BaseModel, dict]) -> BaseModel:
        """
        Validate and convert input data to the expected input model type.

        Args:
            data: Input data as BaseModel instance or dictionary

        Returns:
            Validated BaseModel instance

        Raises:
            TypeError: If input data cannot be converted to InputModel
        """
        if isinstance(data, BaseModel):
            return data

        input_model_type = self._get_input_model_type()
        if isinstance(input_model_type, type) and issubclass(
            input_model_type, BaseModel
        ):
            try:
                return input_model_type(**data)
            except Exception as e:
                raise TypeError(
                    f"Failed to convert input data to {input_model_type.__name__}: {e}"
                ) from e

        raise TypeError("InputModel must be a subclass of BaseModel")

    def _validate_output_model(self) -> type[BaseModel]:
        """
        Validate that the output model is properly configured.

        Returns:
            The validated output model type

        Raises:
            TypeError: If output model is not a valid BaseModel subclass
        """
        output_model_type = self._get_output_model_type()
        if not (
            isinstance(output_model_type, type)
            and issubclass(output_model_type, BaseModel)
        ):
            raise TypeError("OutputModel must be a subclass of BaseModel")
        return output_model_type

    async def _base_call(
        self, data: Union[BaseModel, dict]
    ) -> ParsedOutput[OutputModelType]:
        """
        Core method that handles language model interaction with type safety.

        Args:
            data: Input data as BaseModel instance or dictionary

        Returns:
            Parsed output from the language model

        Raises:
            TypeError: If input/output models are not properly configured
        """
        # Validate input and output models
        validated_input = self._validate_and_convert_input(data)
        self._validate_output_model()

        # Execute the language model call
        return cast(
            ParsedOutput[OutputModelType],
            await self.lm.parse(
                instruction=self.__doc__ or "",
                prompt=validated_input.model_dump_json(),
            ),
        )

    def _create_no_think_messages(self, think_messages: Messages) -> Messages:
        """
        Convert thinking mode messages to non-thinking mode.

        Args:
            think_messages: Original messages with thinking mode enabled

        Returns:
            Messages converted to non-thinking mode
        """
        if not think_messages:
            return think_messages

        # Create deep copy to avoid modifying original
        no_think_messages = copy.deepcopy(think_messages)

        # Update system message
        if no_think_messages and "content" in no_think_messages[0]:
            system_content = no_think_messages[0]["content"]
            if isinstance(system_content, str):
                no_think_messages[0]["content"] = system_content.replace(
                    "/think", "/no_think"
                )

        # Update assistant message (last message)
        if len(no_think_messages) > 1 and "content" in no_think_messages[-1]:
            assistant_content = no_think_messages[-1]["content"]
            if isinstance(assistant_content, str) and "</think>" in assistant_content:
                # Extract content after thinking block
                post_think_content = assistant_content.split("</think>", 1)[1].strip()
                no_think_messages[-1]["content"] = (
                    f"<think>\n\n</think>\n\n{post_think_content}"
                )

        return no_think_messages

    def _save_training_data(
        self,
        input_data: InputModelType,
        think_messages: Messages,
        no_think_messages: Messages,
        model_kwargs: Dict[str, Any],
        cache_dir: pathlib.Path,
        expected_response: Optional[OutputModelType] = None,
        label: Optional[str] = None,
    ) -> None:
        """
        Save training data to cache directory.

        Args:
            input_data: Input data for the task
            think_messages: Messages with thinking mode
            no_think_messages: Messages without thinking mode
            model_kwargs: Model configuration used
            cache_dir: Directory to save training data
            expected_response: Expected response for validation
            label: Optional label for the training data
        """
        # Create unique identifier for this input
        input_id = identify(input_data.model_dump())
        class_cache_dir = cache_dir / self.__class__.__name__
        class_cache_dir.mkdir(parents=True, exist_ok=True)

        # Prepare combined training data
        training_data = {
            "think_messages": think_messages,
            "no_think_messages": no_think_messages,
            "model_kwargs": model_kwargs,
            "input_data": input_data.model_dump(),
            "label": label,
        }

        if expected_response is not None:
            training_data["expected_response"] = expected_response.model_dump()

        # Save to file
        training_file = class_cache_dir / f"{input_id}.json"
        dump_json_or_pickle(training_data, str(training_file))

    async def _generate_training_data_with_thinking_mode(
        self,
        input_data: InputModelType,
        expected_response: Optional[OutputModelType] = None,
        label: Optional[str] = None,
        cache_dir: pathlib.Path = DEFAULT_CACHE_DIR,
    ) -> OutputModelType:
        """
        Generate training data for both thinking and non-thinking modes.

        This method executes the task in thinking mode, then creates equivalent
        non-thinking mode data for training purposes. Both versions are saved
        to the cache directory for later use in model training.

        Args:
            input_data: Input data for the task
            expected_response: Expected response for validation
            label: Optional label for the training data
            cache_dir: Directory to save training data

        Returns:
            Parsed output from the language model
        """
        # Execute the base call to get thinking mode data
        output = await self._base_call(input_data)
        parsed_result = output["parsed"]
        think_messages = output["messages"]

        # Create non-thinking mode equivalent
        no_think_messages = self._create_no_think_messages(think_messages)

        # Save training data
        self._save_training_data(
            input_data=input_data,
            think_messages=think_messages,
            no_think_messages=no_think_messages,
            model_kwargs=output["model_kwargs"],
            cache_dir=cache_dir,
            expected_response=expected_response,
            label=label,
        )

        return parsed_result

    def _should_collect_data(self) -> bool:
        """
        Determine if training data should be collected for this call.

        Returns:
            True if data collection is enabled
        """
        return self.IS_DATA_COLLECTION

    async def __call__(
        self,
        input_data: InputModelType,
        expected_response: Optional[OutputModelType] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> OutputModelType:
        """
        Execute the LLM task with the provided input data.

        This is the main entry point for task execution. If data collection
        is enabled (either via instance configuration or environment variable),
        training data will be automatically generated and saved.

        Args:
            input_data: Input data conforming to InputModelType
            expected_response: Expected response for validation during data collection
            label: Optional label for training data categorization
            **kwargs: Additional keyword arguments (for future extensibility)

        Returns:
            Parsed output conforming to OutputModelType
        """
        if self._should_collect_data():
            return await self._generate_training_data_with_thinking_mode(
                input_data=input_data,
                expected_response=expected_response,
                label=label,
            )
        else:
            output = await self._base_call(input_data)
            return output["parsed"]

    def generate_training_data(
        self, input_json: str, output_json: str
    ) -> Dict[str, Any]:
        """
        Generate training data in ShareGPT format for the given input/output pair.

        This method is useful for creating training datasets from existing
        input/output pairs without executing the language model.

        Args:
            input_dict: Input data as dictionary
            output: Output data as dictionary

        Returns:
            Training data in ShareGPT message format

        Raises:
            AttributeError: If InputModel or OutputModel are not properly defined
        """
        # if not hasattr(self, "InputModel") or not hasattr(self, "OutputModel"):
        #     raise AttributeError(
        #         f"{self.__class__.__name__} must define InputModel and OutputModel "
        #         "as class attributes to use generate_training_data"
        #     )

        system_prompt = self.__doc__ or ""
        assert isinstance(input_json, str), "Input must be a JSON string"
        assert isinstance(output_json, str), "Output must be a JSON string"
        messages = get_conversation_one_turn(
            system_msg=system_prompt,
            user_msg=input_json,
            assistant_msg=output_json,
        )

        return {"messages": messages}

    # Compatibility alias for other LLMTask implementations
    arun = __call__
