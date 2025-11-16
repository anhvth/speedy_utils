"""Mixin classes for LLM functionality extensions."""

# type: ignore

from __future__ import annotations

import os
import subprocess
from time import sleep
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import requests
from loguru import logger


if TYPE_CHECKING:
    from openai import OpenAI
    from pydantic import BaseModel


class TemperatureRangeMixin:
    """Mixin for sampling with different temperature ranges."""

    def temperature_range_sampling(
        self,
        input_data: 'str | BaseModel | list[dict]',
        temperature_ranges: tuple[float, float],
        n: int = 32,
        response_model: 'type[BaseModel] | type[str] | None' = None,
        **runtime_kwargs,
    ) -> list[dict[str, Any]]:
        """
        Sample LLM responses with a range of temperatures.

        This method generates multiple responses by systematically varying
        the temperature parameter, which controls randomness in the output.

        Args:
            input_data: Input data (string, BaseModel, or message list)
            temperature_ranges: Tuple of (min_temp, max_temp) to sample
            n: Number of temperature samples to generate (must be >= 2)
            response_model: Optional response model override
            **runtime_kwargs: Additional runtime parameters

        Returns:
            List of response dictionaries from all temperature samples
        """
        from pydantic import BaseModel

        from speedy_utils.multi_worker.thread import multi_thread

        min_temp, max_temp = temperature_ranges
        if n < 2:
            raise ValueError(f'n must be >= 2, got {n}')

        step = (max_temp - min_temp) / (n - 1)
        list_kwargs = []

        for i in range(n):
            kwargs = dict(
                temperature=min_temp + i * step,
                i=i,
                **runtime_kwargs,
            )
            list_kwargs.append(kwargs)

        def f(kwargs):
            i = kwargs.pop('i')
            sleep(i * 0.05)
            return self.__inner_call__(
                input_data,
                response_model=response_model,
                **kwargs,
            )[0]

        choices = multi_thread(f, list_kwargs, progress=False)
        return [c for c in choices if c is not None]


class TwoStepPydanticMixin:
    """Mixin for two-step Pydantic parsing functionality."""

    def two_step_pydantic_parse(
        self,
        input_data,
        response_model,
        **runtime_kwargs,
    ) -> list[dict[str, Any]]:
        """
        Parse responses in two steps: text completion then Pydantic parsing.

        This is useful for models that may include reasoning or extra text
        before the JSON output.

        Args:
            input_data: Input data (string, BaseModel, or message list)
            response_model: Pydantic model to parse into
            **runtime_kwargs: Additional runtime parameters

        Returns:
            List of parsed response dictionaries
        """
        from pydantic import BaseModel

        # Step 1: Get text completions
        results = self.text_completion(input_data, **runtime_kwargs)
        parsed_results = []

        for result in results:
            response_text = result['parsed']
            messages = result['messages']

            # Handle reasoning models that use <think> tags
            if '</think>' in response_text:
                response_text = response_text.split('</think>')[1]

            try:
                # Try direct parsing - support both Pydantic v1 and v2
                if hasattr(response_model, 'model_validate_json'):
                    # Pydantic v2
                    parsed = response_model.model_validate_json(response_text)
                else:
                    # Pydantic v1
                    import json

                    parsed = response_model.parse_obj(json.loads(response_text))
            except Exception:
                # Fallback: use LLM to extract JSON
                logger.warning('Failed to parse JSON directly, using LLM to extract')
                _parsed_messages = [
                    {
                        'role': 'system',
                        'content': (
                            'You are a helpful assistant that extracts JSON from text.'
                        ),
                    },
                    {
                        'role': 'user',
                        'content': (
                            f'Extract JSON from the following text:\n{response_text}'
                        ),
                    },
                ]
                parsed_result = self.pydantic_parse(
                    _parsed_messages,
                    response_model=response_model,
                    **runtime_kwargs,
                )[0]
                parsed = parsed_result['parsed']

            parsed_results.append({'parsed': parsed, 'messages': messages})

        return parsed_results


class VLLMMixin:
    """Mixin for VLLM server management and LoRA operations."""

    def _setup_vllm_server(self) -> None:
        """
        Setup VLLM server if vllm_cmd is provided.

        This method handles:
        - Server reuse logic
        - Starting new servers
        - Port management

        Should be called from __init__.
        """
        from .utils import (
            _extract_port_from_vllm_cmd,
            _is_server_running,
            _kill_vllm_on_port,
            _start_vllm_server,
            get_base_client,
        )

        if not hasattr(self, 'vllm_cmd') or not self.vllm_cmd:
            return

        port = _extract_port_from_vllm_cmd(self.vllm_cmd)
        reuse_existing = False

        if self.vllm_reuse:
            try:
                reuse_client = get_base_client(port, cache=False)
                models_response = reuse_client.models.list()
                if getattr(models_response, 'data', None):
                    reuse_existing = True
                    logger.info(
                        f'VLLM server already running on port {port}, reusing existing server (vllm_reuse=True)'
                    )
                else:
                    logger.info(
                        f'No models returned from VLLM server on port {port}; starting a new server'
                    )
            except Exception as exc:
                logger.info(
                    f'Unable to reach VLLM server on port {port} (list_models failed): {exc}. Starting a new server.'
                )

        if not self.vllm_reuse:
            if _is_server_running(port):
                logger.info(
                    f'VLLM server already running on port {port}, killing it first (vllm_reuse=False)'
                )
                _kill_vllm_on_port(port)
            logger.info(f'Starting new VLLM server on port {port}')
            self.vllm_process = _start_vllm_server(self.vllm_cmd, self.vllm_timeout)
        elif not reuse_existing:
            logger.info(f'Starting VLLM server on port {port}')
            self.vllm_process = _start_vllm_server(self.vllm_cmd, self.vllm_timeout)

    def _load_lora_adapter(self) -> None:
        """
        Load LoRA adapter from the specified lora_path.

        This method:
        1. Validates that lora_path is a valid LoRA directory
        2. Checks if LoRA is already loaded (unless force_lora_unload)
        3. Loads the LoRA adapter and updates the model name
        """
        from .utils import (
            _get_port_from_client,
            _is_lora_path,
            _load_lora_adapter,
        )

        if not self.lora_path:
            return

        if not _is_lora_path(self.lora_path):
            raise ValueError(
                f"Invalid LoRA path '{self.lora_path}': Directory must contain 'adapter_config.json'"
            )

        logger.info(f'Loading LoRA adapter from: {self.lora_path}')

        # Get the expected LoRA name (basename of the path)
        lora_name = os.path.basename(self.lora_path.rstrip('/\\'))
        if not lora_name:  # Handle edge case of empty basename
            lora_name = os.path.basename(os.path.dirname(self.lora_path))

        # Get list of available models to check if LoRA is already loaded
        try:
            available_models = [m.id for m in self.client.models.list().data]
        except Exception as e:
            logger.warning(
                f'Failed to list models, proceeding with LoRA load: {str(e)[:100]}'
            )
            available_models = []

        # Check if LoRA is already loaded
        if lora_name in available_models and not self.force_lora_unload:
            logger.info(
                f"LoRA adapter '{lora_name}' is already loaded, using existing model"
            )
            self.model_kwargs['model'] = lora_name
            return

        # Force unload if requested
        if self.force_lora_unload and lora_name in available_models:
            logger.info(f"Force unloading LoRA adapter '{lora_name}' before reloading")
            port = _get_port_from_client(self.client)
            if port is not None:
                try:
                    VLLMMixin.unload_lora(port, lora_name)
                    logger.info(f'Successfully unloaded LoRA adapter: {lora_name}')
                except Exception as e:
                    logger.warning(f'Failed to unload LoRA adapter: {str(e)[:100]}')

        # Get port from client for API calls
        port = _get_port_from_client(self.client)
        if port is None:
            raise ValueError(
                f"Cannot load LoRA adapter '{self.lora_path}': "
                f'Unable to determine port from client base_url. '
                f'LoRA loading requires a client initialized with port.'
            )

        try:
            # Load the LoRA adapter
            loaded_lora_name = _load_lora_adapter(self.lora_path, port)
            logger.info(f'Successfully loaded LoRA adapter: {loaded_lora_name}')

            # Update model name to the loaded LoRA name
            self.model_kwargs['model'] = loaded_lora_name

        except requests.RequestException as e:
            # Check if error is due to LoRA already being loaded
            error_msg = str(e)
            if '400' in error_msg or 'Bad Request' in error_msg:
                logger.info(
                    f"LoRA adapter may already be loaded, attempting to use '{lora_name}'"
                )
                # Refresh the model list to check if it's now available
                try:
                    updated_models = [m.id for m in self.client.models.list().data]
                    if lora_name in updated_models:
                        logger.info(
                            f"Found LoRA adapter '{lora_name}' in updated model list"
                        )
                        self.model_kwargs['model'] = lora_name
                        return
                except Exception:
                    pass  # Fall through to original error

            raise ValueError(
                f"Failed to load LoRA adapter from '{self.lora_path}': {error_msg[:100]}"
            ) from e

    def unload_lora_adapter(self, lora_path: str) -> None:
        """
        Unload a LoRA adapter.

        Args:
            lora_path: Path to the LoRA adapter directory to unload

        Raises:
            ValueError: If unable to determine port from client
        """
        from .utils import _get_port_from_client, _unload_lora_adapter

        port = _get_port_from_client(self.client)
        if port is None:
            raise ValueError(
                'Cannot unload LoRA adapter: '
                'Unable to determine port from client base_url. '
                'LoRA operations require a client initialized with port.'
            )

        _unload_lora_adapter(lora_path, port)
        lora_name = os.path.basename(lora_path.rstrip('/\\'))
        logger.info(f'Unloaded LoRA adapter: {lora_name}')

    @staticmethod
    def unload_lora(port: int, lora_name: str) -> None:
        """
        Static method to unload a LoRA adapter by name.

        Args:
            port: Port number for the API endpoint
            lora_name: Name of the LoRA adapter to unload

        Raises:
            requests.RequestException: If the API call fails
        """
        try:
            response = requests.post(
                f'http://localhost:{port}/v1/unload_lora_adapter',
                headers={
                    'accept': 'application/json',
                    'Content-Type': 'application/json',
                },
                json={'lora_name': lora_name, 'lora_int_id': 0},
            )
            response.raise_for_status()
            logger.info(f'Successfully unloaded LoRA adapter: {lora_name}')
        except requests.RequestException as e:
            logger.error(f"Error unloading LoRA adapter '{lora_name}': {str(e)[:100]}")
            raise

    def cleanup_vllm_server(self) -> None:
        """Stop the VLLM server process if started by this instance."""
        from .utils import stop_vllm_process

        if hasattr(self, 'vllm_process') and self.vllm_process is not None:
            stop_vllm_process(self.vllm_process)
            self.vllm_process = None

    @staticmethod
    def kill_all_vllm() -> int:
        """
        Kill all tracked VLLM server processes.

        Returns:
            Number of processes killed
        """
        from .utils import kill_all_vllm_processes

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
        from .utils import _kill_vllm_on_port

        return _kill_vllm_on_port(port)


class ModelUtilsMixin:
    """Mixin for model utility methods."""

    @staticmethod
    def list_models(client=None) -> list[str]:
        """
        List available models from the OpenAI client.

        Args:
            client: OpenAI client, port number, or base_url string

        Returns:
            List of available model names
        """
        from openai import OpenAI

        from .utils import get_base_client

        client_instance = get_base_client(client, cache=False)
        models = client_instance.models.list().data
        return [m.id for m in models]
