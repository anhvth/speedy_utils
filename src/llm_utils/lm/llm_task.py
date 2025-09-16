# type: ignore

"""
Simplified LLM Task module for handling language model interactions with structured input/output.
"""

from typing import Any, Dict, List, Optional, Type, Union, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from .base_prompt_builder import BasePromptBuilder

# Type aliases for better readability
Messages = List[ChatCompletionMessageParam]

from typing import Type

class LLMTask:
    """
    Language model task with structured input/output and optional system instruction.

    Supports str or Pydantic models for both input and output. Automatically handles
    message formatting and response parsing.

    Two main APIs:
    - text(): Returns raw text responses
    - parse(): Returns parsed Pydantic model responses
    - __call__(): Backward compatibility method that delegates based on output_model

    Example:
        ```python
        from pydantic import BaseModel
        from llm_utils.lm.llm_task import LLMTask

        class EmailOutput(BaseModel):
            content: str
            estimated_read_time: int

        # Set up task with Pydantic output model
        task = LLMTask(
            instruction="Generate professional email content.",
            output_model=EmailOutput,
            client=OpenAI(),
            temperature=0.7
        )

        # Use parse() for structured output
        results = task.parse("Write a meeting follow-up email")
        result, messages = results[0]
        print(result.content, result.estimated_read_time)

        # Use text() for plain text output
        results = task.text("Write a meeting follow-up email")
        text_result, messages = results[0]
        print(text_result)

        # Multiple responses
        results = task.parse("Write a meeting follow-up email", n=3)
        for result, messages in results:
            print(f"Content: {result.content}")

        # Override parameters at runtime
        results = task.text(
            "Write a meeting follow-up email",
            temperature=0.9,
            n=2,
            max_tokens=500
        )
        for text_result, messages in results:
            print(text_result)

        # Backward compatibility (uses output_model to choose method)
        results = task("Write a meeting follow-up email")  # Calls parse()
        result, messages = results[0]
        ```
    """

    def __init__(
        self,
        instruction: Optional[str] = None,
        input_model: Union[Type[BaseModel], type[str]] = str,
        output_model: Type[BaseModel]|Type[str] = None,
        client: Union[OpenAI, int, str, None] = None,
        cache=True,
        **model_kwargs
    ):
        """
        Initialize the LLMTask.

        Args:
            instruction: Optional system instruction for the task
            input_model: Input type (str or BaseModel subclass)
            output_model: Output BaseModel type
            client: OpenAI client, port number, or base_url string
            cache: Whether to use cached responses (default True)
            **model_kwargs: Additional model parameters including:
                - temperature: Controls randomness (0.0 to 2.0)
                - n: Number of responses to generate (when n > 1, returns list)
                - max_tokens: Maximum tokens in response
                - model: Model name (auto-detected if not provided)
        """
        self.instruction = instruction
        self.input_model = input_model
        self.output_model = output_model
        self.model_kwargs = model_kwargs
        
        # Set up LM with the client and output model
        if isinstance(client, int):
            base_url = f"http://localhost:{client}/v1"
            api_key = 'abc'
        elif isinstance(client, str):
            base_url = client
            api_key = 'abc'
        elif isinstance(client, OpenAI):
            # Extract base_url and api_key from existing client if possible
            base_url = getattr(client, 'base_url', None)
            api_key = getattr(client, 'api_key', 'abc')
        else:
            base_url = None
            api_key = None

        if cache:
            print("Caching is enabled will use llm_utils.MOpenAI")
            from llm_utils import MOpenAI
            self.client = MOpenAI(base_url=base_url, api_key=api_key)
        else:
            self.client = OpenAI(base_url=base_url, api_key=api_key)

        if not self.model_kwargs.get('model', ''):
            self.model_kwargs['model'] = self.client.models.list().data[0].id
        print(self.model_kwargs)

    def _prepare_input(self, input_data: Union[str, BaseModel, List[Dict]]) -> Messages:
        """Convert input to messages format."""
        if isinstance(input_data, list):
            assert isinstance(input_data[0], dict) and "role" in input_data[0], "If input_data is a list, it must be a list of messages with 'role' and 'content' keys."
            return cast(Messages, input_data)
        else:
            # Convert input to string format
            if isinstance(input_data, str):
                user_content = input_data
            elif hasattr(input_data, 'model_dump_json'):
                user_content = input_data.model_dump_json()
            elif isinstance(input_data, dict):
                user_content = str(input_data)
            else:
                user_content = str(input_data)
            
            # Build messages
            messages = [
                {"role": "system", "content": self.instruction},
            ] if self.instruction is not None else []
            
            messages.append({"role": "user", "content": user_content})
            return cast(Messages, messages)

    def text_completion(
        self,
        input_data: Union[str, BaseModel, list[Dict]],
        **runtime_kwargs
    ) -> List[tuple[str, Messages]]:
        """
        Execute the LLM task and return text responses.

        Args:
            input_data: Input as string or BaseModel
            **runtime_kwargs: Runtime model parameters that override defaults
                - temperature: Controls randomness (0.0 to 2.0)
                - n: Number of responses to generate
                - max_tokens: Maximum tokens in response
                - model: Model name override
                - Any other model parameters supported by OpenAI API

        Returns:
            List of tuples [(text_response, messages), ...]
            When n=1: List contains one tuple
            When n>1: List contains multiple tuples
        """
        # Prepare messages
        messages = self._prepare_input(input_data)

        # Merge runtime kwargs with default model kwargs (runtime takes precedence)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name = effective_kwargs.get('model', self.model_kwargs['model'])
        
        # Extract model name from kwargs for API call
        api_kwargs = {k: v for k, v in effective_kwargs.items() if k != "model"}
        
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            **api_kwargs
        )
        # print(completion)
        
        results: List[tuple[str, Messages]] = []
        for choice in completion.choices:
            choice_messages = cast(
                Messages,
                messages + [{"role": "assistant", "content": choice.message.content}],
            )
            results.append((choice.message.content, choice_messages))
        return results

    def pydantic_parse(
        self,
        input_data: Union[str, BaseModel, list[Dict]],
        response_model: Optional[Type[BaseModel]] = None,
        **runtime_kwargs
    ) -> List[tuple[BaseModel, Messages]]:
        """
        Execute the LLM task and return parsed Pydantic model responses.

        Args:
            input_data: Input as string or BaseModel
            response_model: Pydantic model for response parsing (overrides default)
            **runtime_kwargs: Runtime model parameters that override defaults
                - temperature: Controls randomness (0.0 to 2.0)
                - n: Number of responses to generate
                - max_tokens: Maximum tokens in response
                - model: Model name override
                - Any other model parameters supported by OpenAI API

        Returns:
            List of tuples [(parsed_model, messages), ...]
            When n=1: List contains one tuple
            When n>1: List contains multiple tuples
        """
        # Prepare messages
        messages = self._prepare_input(input_data)

        # Merge runtime kwargs with default model kwargs (runtime takes precedence)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name = effective_kwargs.get('model', self.model_kwargs['model'])
        
        # Extract model name from kwargs for API call
        api_kwargs = {k: v for k, v in effective_kwargs.items() if k != "model"}
        
        pydantic_model_to_use_opt = response_model or self.output_model
        if pydantic_model_to_use_opt is None:
            raise ValueError("No response model specified. Either set output_model in constructor or pass response_model parameter.")
        pydantic_model_to_use: Type[BaseModel] = cast(Type[BaseModel], pydantic_model_to_use_opt)
        
        completion = self.client.chat.completions.parse(
            model=model_name,
            messages=messages,
            response_format=pydantic_model_to_use,
            **api_kwargs
        )
        
        results: List[tuple[BaseModel, Messages]] = []
        for choice in completion.choices:  # type: ignore[attr-defined]
            choice_messages = cast(
                Messages,
                messages + [{"role": "assistant", "content": choice.message.content}],
            )
            results.append((choice.message.parsed, choice_messages))  # type: ignore[attr-defined]
        return results

    def __call__(
        self,
        input_data: Union[str, BaseModel, list[Dict]],
        response_model: Optional[Type[BaseModel]] = None,
        **runtime_kwargs
    ) -> List[tuple[Any, Messages]]:
        """
        Execute the LLM task. Delegates to text() or parse() based on output_model.
        
        This method maintains backward compatibility by automatically choosing
        between text and parse methods based on the output_model configuration.

        Args:
            input_data: Input as string or BaseModel
            response_model: Optional override for output model
            **runtime_kwargs: Runtime model parameters

        Returns:
            List of tuples [(response, messages), ...]
        """
        pydantic_model_to_use = response_model or self.output_model
        
        if pydantic_model_to_use is str or pydantic_model_to_use is None:
            return self.text_completion(input_data, **runtime_kwargs)
        else:
            return self.pydantic_parse(input_data, response_model=response_model, **runtime_kwargs)

    
    
    @classmethod
    def from_prompt_builder(builder: BasePromptBuilder, client: Union[OpenAI, int, str, None] = None, cache=True, **model_kwargs) -> 'LLMTask':
        """
        Create an LLMTask instance from a BasePromptBuilder instance.
        
        This method extracts the instruction, input model, and output model
        from the provided builder and initializes an LLMTask accordingly.
        """
        instruction = builder.get_instruction()
        input_model = builder.get_input_model()
        output_model = builder.get_output_model()
        
        # Extract data from the builder to initialize LLMTask
        return LLMTask(
            instruction=instruction,
            input_model=input_model,
            output_model=output_model,
            client=client,
        )
