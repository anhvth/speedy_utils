import os
import random
import time
from typing import Any, List, Literal, Optional, Type, Union, Dict, overload, Tuple
from pydantic import BaseModel
from speedy_utils import dump_json_or_pickle, identify_uuid, load_json_or_pickle
from loguru import logger
from copy import deepcopy
import numpy as np
import tempfile
import fcntl

T = Type[BaseModel]

class OAI_LM:
    def __init__(
        self,
        model: Optional[str] = None,
        model_type: Literal["chat", "text"] = "chat",
        temperature: float = 0.0,
        max_tokens: int = 2000,
        cache: bool = True,
        callbacks: Optional[Any] = None,
        num_retries: int = 3,
        provider=None,
        finetuning_model: Optional[str] = None,
        launch_kwargs: Optional[dict[str, Any]] = None,
        host: str = "localhost",
        port: Optional[int] = None,
        ports: Optional[List[int]] = None,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        import dspy
        self.ports = ports
        self.host = host
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "abc")
        resolved_base_url_from_kwarg = kwargs.get("base_url")
        if resolved_base_url_from_kwarg is not None and not isinstance(resolved_base_url_from_kwarg, str):
            logger.warning(f"base_url in kwargs was not a string ({type(resolved_base_url_from_kwarg)}), ignoring.")
            resolved_base_url_from_kwarg = None
        resolved_base_url: Optional[str] = resolved_base_url_from_kwarg
        if resolved_base_url is None:
            selected_port = port
            if selected_port is None and ports is not None and len(ports) > 0:
                selected_port = ports[0]
            if selected_port is not None:
                resolved_base_url = f"http://{host}:{selected_port}/v1"
        self.base_url = resolved_base_url
        if model is None:
            if self.base_url:
                try:
                    model_list = self.list_models()
                    if model_list:
                        model_name_from_list = model_list[0]
                        model = f"openai/{model_name_from_list}"
                        logger.info(f"Using default model: {model}")
                    else:
                        logger.warning(f"No models found at {self.base_url}. Please specify a model.")
                except Exception as e:
                    example_cmd = (
                        "LM.start_server('unsloth/gemma-3-1b-it')\n"
                        "# Or manually run: svllm serve --model unsloth/gemma-3-1b-it --gpus 0 -hp localhost:9150"
                    )
                    logger.error(
                        f"Failed to list models from {self.base_url}: {e}\n"
                        f"Make sure your model server is running and accessible.\n"
                        f"Example to start a server:\n{example_cmd}"
                    )
            else:
                logger.warning("base_url not configured, cannot fetch default model. Please specify a model.")
        assert model is not None, "Model name must be provided or discoverable via list_models"
        if not model.startswith("openai/"):
            model = f"openai/{model}"
        dspy_lm_kwargs = kwargs.copy()
        dspy_lm_kwargs["api_key"] = self.api_key
        if self.base_url and "base_url" not in dspy_lm_kwargs:
            dspy_lm_kwargs["base_url"] = self.base_url
        elif self.base_url and "base_url" in dspy_lm_kwargs and dspy_lm_kwargs["base_url"] != self.base_url:
            self.base_url = dspy_lm_kwargs["base_url"]
        self._dspy_lm = dspy.LM(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            num_retries=num_retries,
            provider=provider,
            finetuning_model=finetuning_model,
            launch_kwargs=launch_kwargs,
            **dspy_lm_kwargs,
        )
        self.kwargs = self._dspy_lm.kwargs
        self.model = self._dspy_lm.model
        self.base_url = self.kwargs.get("base_url")
        self.api_key = self.kwargs.get("api_key")
        self.do_cache = cache

    def dump_cache(self, id: str, result: Union[str, BaseModel, List[Union[str, BaseModel]]]):
        try:
            cache_file = f"~/.cache/oai_lm/{self.model}/{id}.pkl"
            cache_file = os.path.expanduser(cache_file)
            dump_json_or_pickle(result, cache_file)
        except Exception as e:
            logger.warning(f"Cache dump failed: {e}")

    def load_cache(self, id: str) -> Optional[Union[str, BaseModel, List[Union[str, BaseModel]]]]:
        try:
            cache_file = f"~/.cache/oai_lm/{self.model}/{id}.pkl"
            cache_file = os.path.expanduser(cache_file)
            if not os.path.exists(cache_file):
                return None
            return load_json_or_pickle(cache_file)
        except Exception as e:
            logger.warning(f"Cache load failed for {id}: {e}")
            return None

    def list_models(self) -> List[str]:
        import openai
        if not self.base_url:
            raise ValueError("Cannot list models: base_url is not configured.")
        if not self.api_key:
            logger.warning("API key not available for listing models. Using default 'abc'.")
        api_key_str = str(self.api_key) if self.api_key is not None else "abc"
        base_url_str = str(self.base_url) if self.base_url is not None else None
        if isinstance(self.base_url, float):
            raise TypeError(f"base_url must be a string or None, got float: {self.base_url}")
        client = openai.OpenAI(base_url=base_url_str, api_key=api_key_str)
        page = client.models.list()
        return [d.id for d in page.data]

    def get_least_used_port(self) -> int:
        if self.ports is None:
            raise ValueError("Ports must be configured to pick the least used port.")
        if not self.ports:
            raise ValueError("Ports list is empty, cannot pick a port.")
        return self._pick_least_used_port(self.ports)

    def _pick_least_used_port(self, ports: List[int]) -> int:
        global_lock_file = "/tmp/ports.lock"
        with open(global_lock_file, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                port_use: Dict[int, int] = {}
                for port in ports:
                    file_counter = f"/tmp/port_use_counter_{port}.npy"
                    if os.path.exists(file_counter):
                        try:
                            counter = np.load(file_counter)
                        except Exception as e:
                            logger.warning(f"Corrupted usage file {file_counter}: {e}")
                            counter = np.array([0])
                    else:
                        counter = np.array([0])
                    port_use[port] = counter[0]
                if not port_use:
                    if ports:
                        raise ValueError("Port usage data is empty, cannot pick a port.")
                    else:
                        raise ValueError("No ports provided to pick from.")
                lsp = min(port_use, key=lambda k: port_use[k])
                self._update_port_use(lsp, 1)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        return lsp

    def _update_port_use(self, port: int, increment: int):
        file_counter = f"/tmp/port_use_counter_{port}.npy"
        file_counter_lock = f"/tmp/port_use_counter_{port}.lock"
        with open(file_counter_lock, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if os.path.exists(file_counter):
                    try:
                        counter = np.load(file_counter)
                    except Exception as e:
                        logger.warning(f"Corrupted usage file {file_counter}: {e}")
                        counter = np.array([0])
                else:
                    counter = np.array([0])
                counter[0] += increment
                self._atomic_save(counter, file_counter)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _atomic_save(self, array: np.ndarray, filename: str):
        tmp_dir: str = os.path.dirname(filename) or "."
        with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as tmp:
            np.save(tmp, array)
            temp_name: str = tmp.name
        os.replace(temp_name, filename)

    def _prepare_call_inputs(
        self,
        prompt: Optional[str],
        messages: Optional[List[Any]],
        max_tokens: Optional[int],
        port: Optional[int],
        use_loadbalance: Optional[bool],
        cache: Optional[bool],
        **kwargs
    ) -> Tuple[dict, bool, Optional[int], Union[str, List[Any]]]:
        """Prepare inputs for the LLM call."""
        # Prepare kwargs
        effective_kwargs = {**self.kwargs, **kwargs}
        if max_tokens is not None:
            effective_kwargs["max_tokens"] = max_tokens

        # Set effective cache
        effective_cache = cache if cache is not None else self.do_cache

        # Setup port
        current_port = port
        if self.ports and not current_port:
            current_port = (
                self.get_least_used_port()
                if use_loadbalance
                else random.choice(self.ports)
            )
        if current_port:
            effective_kwargs["base_url"] = f"http://{self.host}:{current_port}/v1"

        # Prepare main input
        dspy_main_input: Union[str, List[Any]]
        if messages is not None:
            dspy_main_input = messages
        elif prompt is not None:
            dspy_main_input = prompt
        else:
            raise ValueError("Either 'prompt' or 'messages' must be provided.")

        return effective_kwargs, effective_cache, current_port, dspy_main_input

    def _call_llm(
        self,
        dspy_main_input: Union[str, List[Any]],
        current_port: Optional[int],
        use_loadbalance: Optional[bool],
        **kwargs
    ) -> Any:
        """Call the LLM and get raw output."""
        llm_outputs_list = self._dspy_lm(
            dspy_main_input,
            **kwargs,
        )
        if not llm_outputs_list:
            raise ValueError("LLM call returned an empty list.")

        llm_output = (
            llm_outputs_list[0]
            if isinstance(llm_outputs_list, list)
            else llm_outputs_list
        )

        # Update port usage stats if needed
        if current_port and use_loadbalance is True:
            self._update_port_use(current_port, -1)

        return llm_output

    def _generate_cache_key_base(
        self,
        prompt: Optional[str],
        messages: Optional[List[Any]],
        effective_kwargs: dict,
    ) -> List[Any]:
        """Base method to generate cache key components."""
        return [
            prompt,
            messages,
            effective_kwargs.get("temperature"),
            effective_kwargs.get("max_tokens"),
            self.model,
        ]

    def _store_in_cache_base(
        self, effective_cache: bool, id_for_cache: Optional[str], result: Any
    ):
        """Base method to store result in cache if caching is enabled."""
        if effective_cache and id_for_cache:
            self.dump_cache(id_for_cache, result)

