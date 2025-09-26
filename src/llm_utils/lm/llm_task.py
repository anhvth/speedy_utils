# type: ignore

"""
Simplified LLM Task module for handling language model interactions with structured input/output.
"""

import os
import subprocess
from typing import Any, Dict, List, Optional, Type, Union, cast

import requests
from loguru import logger
from openai import OpenAI, AuthenticationError, BadRequestError, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from .utils import (
    _extract_port_from_vllm_cmd,
    _start_vllm_server,
    _kill_vllm_on_port,
    _is_server_running,
    get_base_client,
    _is_lora_path,
    _get_port_from_client,
    _load_lora_adapter,
    _unload_lora_adapter,
    kill_all_vllm_processes,
    stop_vllm_process,
)
from .base_prompt_builder import BasePromptBuilder

# Type aliases for better readability
Messages = List[ChatCompletionMessageParam]


class LLMTask:
    """LLM task with structured input/output handling."""

    def __init__(
        self,
        instruction: Optional[str] = None,
        input_model: Union[Type[BaseModel], type[str]] = str,
        output_model: Type[BaseModel] | Type[str] = None,
        client: Union[OpenAI, int, str, None] = None,
        cache=True,
        is_reasoning_model: bool = False,
        force_lora_unload: bool = False,
        lora_path: Optional[str] = None,
        vllm_cmd: Optional[str] = None,
        vllm_timeout: int = 1200,
        vllm_reuse: bool = True,
        **model_kwargs,
    ):
        """Initialize LLMTask."""
        self.instruction = instruction
        self.input_model = input_model
        self.output_model = output_model
        self.model_kwargs = model_kwargs
        self.is_reasoning_model = is_reasoning_model
        self.force_lora_unload = force_lora_unload
        self.lora_path = lora_path
        self.vllm_cmd = vllm_cmd
        self.vllm_timeout = vllm_timeout
        self.vllm_reuse = vllm_reuse
        self.vllm_process: Optional[subprocess.Popen] = None
        self.last_ai_response = None  # Store raw response from client

        # Handle VLLM server startup if vllm_cmd is provided
        if self.vllm_cmd:
            port = _extract_port_from_vllm_cmd(self.vllm_cmd)
            reuse_existing = False

            if self.vllm_reuse:
                try:
                    reuse_client = get_base_client(port, cache=False)
                    models_response = reuse_client.models.list()
                    if getattr(models_response, "data", None):
                        reuse_existing = True
                        logger.info(
                            f"VLLM server already running on port {port}, "
                            "reusing existing server (vllm_reuse=True)"
                        )
                    else:
                        logger.info(
                            f"No models returned from VLLM server on port {port}; "
                            "starting a new server"
                        )
                except Exception as exc:
                    logger.info(
                        f"Unable to reach VLLM server on port {port} (list_models failed): {exc}. "
                        "Starting a new server."
                    )

            if not self.vllm_reuse:
                if _is_server_running(port):
                    logger.info(
                        f"VLLM server already running on port {port}, killing it first (vllm_reuse=False)"
                    )
                    _kill_vllm_on_port(port)
                logger.info(f"Starting new VLLM server on port {port}")
                self.vllm_process = _start_vllm_server(self.vllm_cmd, self.vllm_timeout)
            elif not reuse_existing:
                logger.info(f"Starting VLLM server on port {port}")
                self.vllm_process = _start_vllm_server(self.vllm_cmd, self.vllm_timeout)

            # Set client to use the VLLM server port if not explicitly provided
            if client is None:
                client = port

        self.client = get_base_client(client, cache=cache, vllm_cmd=self.vllm_cmd, vllm_process=self.vllm_process)
        # check connection of client
        try:
            self.client.models.list()
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI client: {str(e)}, base_url={self.client.base_url}")
            raise e

        if not self.model_kwargs.get("model", ""):
            self.model_kwargs["model"] = self.client.models.list().data[0].id
        
        # Handle LoRA loading if lora_path is provided
        if self.lora_path:
            self._load_lora_adapter()

    def cleanup_vllm_server(self) -> None:
        """Stop the VLLM server process if it was started by this instance."""
        if self.vllm_process is not None:
            stop_vllm_process(self.vllm_process)
            self.vllm_process = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_vllm_server()

    def _load_lora_adapter(self) -> None:
        """
        Load LoRA adapter from the specified lora_path.
        
        This method:
        1. Validates that lora_path is a valid LoRA directory
        2. Checks if LoRA is already loaded (unless force_lora_unload is True)
        3. Loads the LoRA adapter and updates the model name
        """
        if not self.lora_path:
            return
            
        if not _is_lora_path(self.lora_path):
            raise ValueError(
                f"Invalid LoRA path '{self.lora_path}': "
                "Directory must contain 'adapter_config.json'"
            )
            
        logger.info(f"Loading LoRA adapter from: {self.lora_path}")
        
        # Get the expected LoRA name (basename of the path)
        lora_name = os.path.basename(self.lora_path.rstrip('/\\'))
        if not lora_name:  # Handle edge case of empty basename
            lora_name = os.path.basename(os.path.dirname(self.lora_path))
        
        # Get list of available models to check if LoRA is already loaded
        try:
            available_models = [m.id for m in self.client.models.list().data]
        except Exception as e:
            logger.warning(f"Failed to list models, proceeding with LoRA load: {str(e)[:100]}")
            available_models = []
        
        # Check if LoRA is already loaded
        if lora_name in available_models and not self.force_lora_unload:
            logger.info(f"LoRA adapter '{lora_name}' is already loaded, using existing model")
            self.model_kwargs["model"] = lora_name
            return
        
        # Force unload if requested
        if self.force_lora_unload and lora_name in available_models:
            logger.info(f"Force unloading LoRA adapter '{lora_name}' before reloading")
            port = _get_port_from_client(self.client)
            if port is not None:
                try:
                    LLMTask.unload_lora(port, lora_name)
                    logger.info(f"Successfully unloaded LoRA adapter: {lora_name}")
                except Exception as e:
                    logger.warning(f"Failed to unload LoRA adapter: {str(e)[:100]}")
        
        # Get port from client for API calls
        port = _get_port_from_client(self.client)
        if port is None:
            raise ValueError(
                f"Cannot load LoRA adapter '{self.lora_path}': "
                "Unable to determine port from client base_url. "
                "LoRA loading requires a client initialized with port number."
            )
        
        try:
            # Load the LoRA adapter
            loaded_lora_name = _load_lora_adapter(self.lora_path, port)
            logger.info(f"Successfully loaded LoRA adapter: {loaded_lora_name}")
            
            # Update model name to the loaded LoRA name
            self.model_kwargs["model"] = loaded_lora_name
            
        except requests.RequestException as e:
            # Check if the error is due to LoRA already being loaded
            error_msg = str(e)
            if "400" in error_msg or "Bad Request" in error_msg:
                logger.info(f"LoRA adapter may already be loaded, attempting to use '{lora_name}'")
                # Refresh the model list to check if it's now available
                try:
                    updated_models = [m.id for m in self.client.models.list().data]
                    if lora_name in updated_models:
                        logger.info(f"Found LoRA adapter '{lora_name}' in updated model list")
                        self.model_kwargs["model"] = lora_name
                        return
                except Exception:
                    pass  # Fall through to original error
            
            raise ValueError(
                f"Failed to load LoRA adapter from '{self.lora_path}': {error_msg[:100]}"
            )

    def unload_lora_adapter(self, lora_path: str) -> None:
        """
        Unload a LoRA adapter.
        
        Args:
            lora_path: Path to the LoRA adapter directory to unload
            
        Raises:
            ValueError: If unable to determine port from client
        """
        port = _get_port_from_client(self.client)
        if port is None:
            raise ValueError(
                "Cannot unload LoRA adapter: "
                "Unable to determine port from client base_url. "
                "LoRA operations require a client initialized with port number."
            )
        
        _unload_lora_adapter(lora_path, port)
        lora_name = os.path.basename(lora_path.rstrip('/\\'))
        logger.info(f"Unloaded LoRA adapter: {lora_name}")

    @staticmethod
    def unload_lora(port: int, lora_name: str) -> None:
        """Static method to unload a LoRA adapter by name.
        
        Args:
            port: Port number for the API endpoint
            lora_name: Name of the LoRA adapter to unload
            
        Raises:
            requests.RequestException: If the API call fails
        """
        try:
            response = requests.post(
                f'http://localhost:{port}/v1/unload_lora_adapter',
                headers={'accept': 'application/json', 'Content-Type': 'application/json'},
                json={"lora_name": lora_name, "lora_int_id": 0}
            )
            response.raise_for_status()
            logger.info(f"Successfully unloaded LoRA adapter: {lora_name}")
        except requests.RequestException as e:
            logger.error(f"Error unloading LoRA adapter '{lora_name}': {str(e)[:100]}")
            raise

    def _prepare_input(self, input_data: Union[str, BaseModel, List[Dict]]) -> Messages:
        """Convert input to messages format."""
        if isinstance(input_data, list):
            assert isinstance(input_data[0], dict) and "role" in input_data[0], (
                "If input_data is a list, it must be a list of messages with 'role' and 'content' keys."
            )
            return cast(Messages, input_data)
        else:
            # Convert input to string format
            if isinstance(input_data, str):
                user_content = input_data
            elif hasattr(input_data, "model_dump_json"):
                user_content = input_data.model_dump_json()
            elif isinstance(input_data, dict):
                user_content = str(input_data)
            else:
                user_content = str(input_data)

            # Build messages
            messages = (
                [
                    {"role": "system", "content": self.instruction},
                ]
                if self.instruction is not None
                else []
            )

            messages.append({"role": "user", "content": user_content})
            return cast(Messages, messages)

    def text_completion(
        self, input_data: Union[str, BaseModel, list[Dict]], **runtime_kwargs
    ) -> List[Dict[str, Any]]:
        """Execute LLM task and return text responses."""
        # Prepare messages
        messages = self._prepare_input(input_data)

        # Merge runtime kwargs with default model kwargs (runtime takes precedence)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name = effective_kwargs.get("model", self.model_kwargs["model"])

        # Extract model name from kwargs for API call
        api_kwargs = {k: v for k, v in effective_kwargs.items() if k != "model"}

        try:
            completion = self.client.chat.completions.create(
                model=model_name, messages=messages, **api_kwargs
            )
            # Store raw response from client
            self.last_ai_response = completion
        except (AuthenticationError, RateLimitError, BadRequestError) as exc:
            error_msg = f"OpenAI API error ({type(exc).__name__}): {exc}"
            logger.error(error_msg)
            raise
        except Exception as e:
            is_length_error = "Length" in str(e) or "maximum context length" in str(e)
            if is_length_error:
                raise ValueError(
                    f"Input too long for model {model_name}. Error: {str(e)[:100]}..."
                )
            # Re-raise all other exceptions
            raise
        # print(completion)

        results: List[Dict[str, Any]] = []
        for choice in completion.choices:
            choice_messages = cast(
                Messages,
                messages + [{"role": "assistant", "content": choice.message.content}],
            )
            result_dict = {"parsed": choice.message.content, "messages": choice_messages}
            
            # Add reasoning content if this is a reasoning model
            if self.is_reasoning_model and hasattr(choice.message, 'reasoning_content'):
                result_dict["reasoning_content"] = choice.message.reasoning_content
            
            results.append(result_dict)
        return results

    def pydantic_parse(
        self,
        input_data: Union[str, BaseModel, list[Dict]],
        response_model: Optional[Type[BaseModel]] | Type[str] = None,
        **runtime_kwargs,
    ) -> List[Dict[str, Any]]:
        """Execute LLM task and return parsed Pydantic model responses."""
        # Prepare messages
        messages = self._prepare_input(input_data)

        # Merge runtime kwargs with default model kwargs (runtime takes precedence)
        effective_kwargs = {**self.model_kwargs, **runtime_kwargs}
        model_name = effective_kwargs.get("model", self.model_kwargs["model"])

        # Extract model name from kwargs for API call
        api_kwargs = {k: v for k, v in effective_kwargs.items() if k != "model"}

        pydantic_model_to_use_opt = response_model or self.output_model
        if pydantic_model_to_use_opt is None:
            raise ValueError(
                "No response model specified. Either set output_model in constructor or pass response_model parameter."
            )
        pydantic_model_to_use: Type[BaseModel] = cast(
            Type[BaseModel], pydantic_model_to_use_opt
        )
        try:
            completion = self.client.chat.completions.parse(
                model=model_name,
                messages=messages,
                response_format=pydantic_model_to_use,
                **api_kwargs,
            )
            # Store raw response from client
            self.last_ai_response = completion
        except (AuthenticationError, RateLimitError, BadRequestError) as exc:
            error_msg = f"OpenAI API error ({type(exc).__name__}): {exc}"
            logger.error(error_msg)
            raise
        except Exception as e:
            is_length_error = "Length" in str(e) or "maximum context length" in str(e)
            if is_length_error:
                raise ValueError(
                    f"Input too long for model {model_name}. Error: {str(e)[:100]}..."
                )
            # Re-raise all other exceptions
            raise

        results: List[Dict[str, Any]] = []
        for choice in completion.choices:  # type: ignore[attr-defined]
            choice_messages = cast(
                Messages,
                messages + [{"role": "assistant", "content": choice.message.content}],
            )
            
            # Ensure consistent Pydantic model output for both fresh and cached responses
            parsed_content = choice.message.parsed  # type: ignore[attr-defined]
            if isinstance(parsed_content, dict):
                # Cached response: validate dict back to Pydantic model
                parsed_content = pydantic_model_to_use.model_validate(parsed_content)
            elif not isinstance(parsed_content, pydantic_model_to_use):
                # Fallback: ensure it's the correct type
                parsed_content = pydantic_model_to_use.model_validate(parsed_content)
            
            result_dict = {"parsed": parsed_content, "messages": choice_messages}
            
            # Add reasoning content if this is a reasoning model
            if self.is_reasoning_model and hasattr(choice.message, 'reasoning_content'):
                result_dict["reasoning_content"] = choice.message.reasoning_content
            
            results.append(result_dict)
        return results

    def __call__(
        self,
        input_data: Union[str, BaseModel, list[Dict]],
        response_model: Optional[Type[BaseModel] | Type[str]] = None,
        two_step_parse_pydantic=False,
        **runtime_kwargs,
    ) -> List[Dict[str, Any]]:
        """Execute LLM task. Delegates to text() or parse() based on output_model."""
        pydantic_model_to_use = response_model or self.output_model

        if pydantic_model_to_use is str or pydantic_model_to_use is None:
            return self.text_completion(input_data, **runtime_kwargs)
        elif two_step_parse_pydantic:
            # step 1: get text completions
            results = self.text_completion(input_data, **runtime_kwargs)
            parsed_results = []
            for result in results:
                response_text = result["parsed"]
                messages = result["messages"]
                # check if the pydantic_model_to_use is validated
                if "</think>" in response_text:
                    response_text = response_text.split("</think>")[1]
                try:
                    parsed = pydantic_model_to_use.model_validate_json(response_text)
                except Exception:
                    # Failed to parse JSON, falling back to LLM parsing
                    # use model to parse the response_text
                    _parsed_messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that extracts JSON from text.",
                        },
                        {
                            "role": "user",
                            "content": f"Extract JSON from the following text:\n{response_text}",
                        },
                    ]
                    parsed_result = self.pydantic_parse(
                        _parsed_messages,
                        response_model=pydantic_model_to_use,
                        **runtime_kwargs,
                    )[0]
                    parsed = parsed_result["parsed"]
                # ---
                parsed_results.append({"parsed": parsed, "messages": messages})
            return parsed_results

        else:
            return self.pydantic_parse(
                input_data, response_model=response_model, **runtime_kwargs
            )

    # Backward compatibility aliases
    def text(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Alias for text_completion() for backward compatibility."""
        return self.text_completion(*args, **kwargs)

    def parse(self, *args, **kwargs) -> List[Dict[str, Any]]:
        """Alias for pydantic_parse() for backward compatibility."""
        return self.pydantic_parse(*args, **kwargs)

    @classmethod
    def from_prompt_builder(
        builder: BasePromptBuilder,
        client: Union[OpenAI, int, str, None] = None,
        cache=True,
        is_reasoning_model: bool = False,
        lora_path: Optional[str] = None,
        vllm_cmd: Optional[str] = None,
        vllm_timeout: int = 120,
        vllm_reuse: bool = True,
        **model_kwargs,
    ) -> "LLMTask":
        """
        Create an LLMTask instance from a BasePromptBuilder instance.

        This method extracts the instruction, input model, and output model
        from the provided builder and initializes an LLMTask accordingly.
        
        Args:
            builder: BasePromptBuilder instance
            client: OpenAI client, port number, or base_url string  
            cache: Whether to use cached responses (default True)
            is_reasoning_model: Whether model is reasoning model (default False)
            lora_path: Optional path to LoRA adapter directory
            vllm_cmd: Optional VLLM command to start server automatically
            vllm_timeout: Timeout in seconds to wait for VLLM server (default 120)
            vllm_reuse: If True (default), reuse existing server on target port
            **model_kwargs: Additional model parameters
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
            cache=cache,
            is_reasoning_model=is_reasoning_model,
            lora_path=lora_path,
            vllm_cmd=vllm_cmd,
            vllm_timeout=vllm_timeout,
            vllm_reuse=vllm_reuse,
            **model_kwargs,
        )

    @staticmethod
    def list_models(client: Union[OpenAI, int, str, None] = None) -> List[str]:
        """
        List available models from the OpenAI client.

        Args:
            client: OpenAI client, port number, or base_url string

        Returns:
            List of available model names.
        """
        client = get_base_client(client, cache=False)
        models = client.models.list().data
        return [m.id for m in models]

    @staticmethod
    def kill_all_vllm() -> int:
        """Kill all tracked VLLM server processes."""
        return kill_all_vllm_processes()

    @staticmethod
    def kill_vllm_on_port(port: int) -> bool:
        """
        Kill VLLM server running on a specific port.
        
        Args:
            port: Port number to kill server on
            
        Returns:
            True if a server was killed, False if no server was running
        """
        return _kill_vllm_on_port(port)


# Example usage:
if __name__ == "__main__":
    # Example 1: Using VLLM with reuse (default behavior)
    vllm_command = (
        "vllm serve saves/vng/dpo/01 -tp 4 --port 8001 "
        "--gpu-memory-utilization 0.9 --served-model-name sft --quantization experts_int8"
    )
    
    print("Example 1: Using VLLM with server reuse (default)")
    # Create LLM instance - will reuse existing server if running on port 8001
    with LLMTask(vllm_cmd=vllm_command) as llm:  # vllm_reuse=True by default
        result = llm.text("Hello, how are you?")
        print("Response:", result[0]["parsed"])
    
    print("\nExample 2: Force restart server (vllm_reuse=False)")
    # This will kill any existing server on port 8001 and start fresh
    with LLMTask(vllm_cmd=vllm_command, vllm_reuse=False) as llm:
        result = llm.text("Tell me a joke")
        print("Joke:", result[0]["parsed"])
        
    print("\nExample 3: Multiple instances with reuse")
    # First instance starts the server
    llm1 = LLMTask(vllm_cmd=vllm_command)  # Starts server or reuses existing
    
    # Second instance reuses the same server
    llm2 = LLMTask(vllm_cmd=vllm_command)  # Reuses server on port 8001
    
    try:
        result1 = llm1.text("What's the weather like?")
        result2 = llm2.text("How's the traffic?")
        print("Weather response:", result1[0]["parsed"])
        print("Traffic response:", result2[0]["parsed"])
    finally:
        # Only cleanup if we started the process
        llm1.cleanup_vllm_server()
        llm2.cleanup_vllm_server()  # Won't do anything if process not owned
    
    print("\nExample 4: Different ports")
    # These will start separate servers
    llm_8001 = LLMTask(vllm_cmd="vllm serve model1 --port 8001", vllm_reuse=True)
    llm_8002 = LLMTask(vllm_cmd="vllm serve model2 --port 8002", vllm_reuse=True) 
    
    print("\nExample 5: Kill all VLLM servers")
    # Kill all tracked VLLM processes
    killed_count = LLMTask.kill_all_vllm()
    print(f"Killed {killed_count} VLLM servers")
    
    print("\nYou can check VLLM server output at: /tmp/vllm.txt")
    print("Server reuse behavior:")
    print("- vllm_reuse=True (default): Reuse existing server on target port")
    print("- vllm_reuse=False: Kill existing server first, then start fresh")

