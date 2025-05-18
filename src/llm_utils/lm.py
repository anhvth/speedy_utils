import fcntl
import os
import random
import tempfile
from copy import deepcopy
import time
from typing import Any, List, Literal, Optional, TypedDict, Dict, Type, Union, cast


import numpy as np
from loguru import logger
from pydantic import BaseModel
from speedy_utils import dump_json_or_pickle, identify_uuid, load_json_or_pickle


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str | BaseModel


class ChatSession:

    def __init__(
        self,
        lm: "OAI_LM",
        system_prompt: Optional[str] = None,
        history: List[Message] = [],  # Default to empty list, deepcopy happens below
        callback=None,
        response_format: Optional[Type[BaseModel]] = None,
    ):
        self.lm = deepcopy(lm)
        self.history = deepcopy(history)  # Deepcopy the provided history
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

    def __call__(
        self,
        text,
        response_format: Optional[Type[BaseModel]] = None,
        display=False,
        max_prev_turns=3,
        **kwargs,
    ) -> str | BaseModel:
        current_response_format = response_format or self.response_format
        self.history.append({"role": "user", "content": text})
        output = self.lm(
            messages=self.parse_history(),
            response_format=current_response_format,
            **kwargs,
        )
        # output could be a string or a pydantic model
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
        """
        Wrapper around __call__ method for sending messages.
        This maintains compatibility with the test suite.
        """
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


def _clear_port_use(ports):
    """
    Clear the usage counters for all ports.
    """
    for port in ports:
        file_counter = f"/tmp/port_use_counter_{port}.npy"
        if os.path.exists(file_counter):
            os.remove(file_counter)


def _atomic_save(array: np.ndarray, filename: str):
    """
    Write `array` to `filename` with an atomic rename to avoid partial writes.
    """
    # The temp file must be on the same filesystem as `filename` to ensure
    # that os.replace() is truly atomic.
    tmp_dir = os.path.dirname(filename) or "."
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as tmp:
        np.save(tmp, array)
        temp_name = tmp.name

    # Atomically rename the temp file to the final name.
    # On POSIX systems, os.replace is an atomic operation.
    os.replace(temp_name, filename)


def _update_port_use(port: int, increment: int):
    """
    Update the usage counter for a given port, safely with an exclusive lock
    and atomic writes to avoid file corruption.
    """
    file_counter = f"/tmp/port_use_counter_{port}.npy"
    file_counter_lock = f"/tmp/port_use_counter_{port}.lock"

    with open(file_counter_lock, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            # If file exists, load it. Otherwise assume zero usage.
            if os.path.exists(file_counter):
                try:
                    counter = np.load(file_counter)
                except Exception as e:
                    # If we fail to load (e.g. file corrupted), start from zero
                    logger.warning(f"Corrupted usage file {file_counter}: {e}")
                    counter = np.array([0])
            else:
                counter = np.array([0])

            # Increment usage and atomically overwrite the old file
            counter[0] += increment
            _atomic_save(counter, file_counter)

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _pick_least_used_port(ports: List[int]) -> int:
    """
    Pick the least-used port among the provided list, safely under a global lock
    so that no two processes pick a port at the same time.
    """
    global_lock_file = "/tmp/ports.lock"

    with open(global_lock_file, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            port_use: Dict[int, int] = {}
            # Read usage for each port
            for port in ports:
                file_counter = f"/tmp/port_use_counter_{port}.npy"
                if os.path.exists(file_counter):
                    try:
                        counter = np.load(file_counter)
                    except Exception as e:
                        # If the file is corrupted, reset usage to 0
                        logger.warning(f"Corrupted usage file {file_counter}: {e}")
                        counter = np.array([0])
                else:
                    counter = np.array([0])
                port_use[port] = counter[0]

            logger.debug(f"Port use: {port_use}")

            if not port_use:
                if ports:
                    raise ValueError("Port usage data is empty, cannot pick a port.")
                else:
                    raise ValueError("No ports provided to pick from.")

            # Pick the least-used port
            lsp = min(port_use, key=lambda k: port_use[k])

            # Increment usage of that port
            _update_port_use(lsp, 1)

        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    return lsp


class OAI_LM:
    """
    A language model supporting chat or text completion requests for use with DSPy modules.
    """

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
        # Lazy import dspy
        import dspy

        self.ports = ports
        self.host = host
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "abc")

        # Determine base_url: kwargs["base_url"] > http://host:port > http://host:ports[0]
        resolved_base_url_from_kwarg = kwargs.get("base_url")
        if resolved_base_url_from_kwarg is not None and not isinstance(
            resolved_base_url_from_kwarg, str
        ):
            logger.warning(
                f"base_url in kwargs was not a string ({type(resolved_base_url_from_kwarg)}), ignoring."
            )
            resolved_base_url_from_kwarg = None

        resolved_base_url: Optional[str] = cast(
            Optional[str], resolved_base_url_from_kwarg
        )

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
                    model_list = (
                        self.list_models()
                    )  # Uses self.base_url and self.api_key
                    if model_list:
                        model_name_from_list = model_list[0]
                        model = f"openai/{model_name_from_list}"
                        logger.info(f"Using default model: {model}")
                    else:
                        logger.warning(
                            f"No models found at {self.base_url}. Please specify a model."
                        )
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
                logger.warning(
                    "base_url not configured, cannot fetch default model. Please specify a model."
                )

        assert (
            model is not None
        ), "Model name must be provided or discoverable via list_models"

        if not model.startswith("openai/"):
            model = f"openai/{model}"

        dspy_lm_kwargs = kwargs.copy()
        dspy_lm_kwargs["api_key"] = self.api_key  # Ensure dspy.LM gets this

        if self.base_url and "base_url" not in dspy_lm_kwargs:
            dspy_lm_kwargs["base_url"] = self.base_url
        elif (
            self.base_url
            and "base_url" in dspy_lm_kwargs
            and dspy_lm_kwargs["base_url"] != self.base_url
        ):
            # If kwarg['base_url'] exists and differs from derived self.base_url,
            # dspy.LM will use kwarg['base_url']. Update self.base_url to reflect this.
            self.base_url = dspy_lm_kwargs["base_url"]

        self._dspy_lm: dspy.LM = dspy.LM(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            num_retries=num_retries,
            provider=provider,
            finetuning_model=finetuning_model,
            launch_kwargs=launch_kwargs,
            # api_key is passed via dspy_lm_kwargs
            **dspy_lm_kwargs,
        )
        # Store the actual kwargs used by dspy.LM
        self.kwargs = self._dspy_lm.kwargs
        self.model = self._dspy_lm.model  # self.model is str

        # Ensure self.base_url and self.api_key are consistent with what dspy.LM is using
        self.base_url = self.kwargs.get("base_url")
        self.api_key = self.kwargs.get("api_key")

        self.do_cache = cache

    @property
    def last_response(self):
        return self._dspy_lm.history[-1]["response"].model_dump()["choices"][0][
            "message"
        ]

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Message]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        cache: Optional[bool] = None,
        retry_count: int = 0,
        port: Optional[int] = None,
        error: Optional[Exception] = None,
        use_loadbalance: Optional[bool] = None,
        must_load_cache: bool = False,
        max_tokens: Optional[int] = None,
        num_retries: int = 10,
        **kwargs,
    ) -> Union[str, BaseModel]:
        if retry_count > num_retries:
            logger.error(f"Retry limit exceeded, error: {error}")
            if error:
                raise error
            raise ValueError("Retry limit exceeded with no specific error.")

        effective_kwargs = {**self.kwargs, **kwargs}
        id_for_cache: Optional[str] = None

        effective_cache = cache if cache is not None else self.do_cache

        if max_tokens is not None:
            effective_kwargs["max_tokens"] = max_tokens

        if response_format:
            assert issubclass(
                response_format, BaseModel
            ), f"response_format must be a Pydantic model class, {type(response_format)} provided"

        cached_result: Optional[Union[str, BaseModel, List[Union[str, BaseModel]]]] = (
            None
        )
        if effective_cache:
            cache_key_list = [
                prompt,
                messages,
                (response_format.model_json_schema() if response_format else None),
                effective_kwargs.get("temperature"),
                effective_kwargs.get("max_tokens"),
                self.model,
            ]
            s = str(cache_key_list)
            id_for_cache = identify_uuid(s)
            cached_result = self.load_cache(id_for_cache)

        if cached_result is not None:
            if response_format:
                if isinstance(cached_result, str):
                    try:
                        import json_repair

                        parsed = json_repair.loads(cached_result)
                        if not isinstance(parsed, dict):
                            raise ValueError("Parsed cached_result is not a dict")
                        # Ensure keys are strings
                        parsed = {str(k): v for k, v in parsed.items()}
                        return response_format(**parsed)
                    except Exception as e_parse:
                        logger.warning(
                            f"Failed to parse cached string for {id_for_cache} into {response_format.__name__}: {e_parse}. Retrying LLM call."
                        )
                elif isinstance(cached_result, response_format):
                    return cached_result
                else:
                    logger.warning(
                        f"Cached result for {id_for_cache} has unexpected type {type(cached_result)}. Expected {response_format.__name__} or str. Retrying LLM call."
                    )
            else:  # No response_format, expect string
                if isinstance(cached_result, str):
                    return cached_result
                else:
                    logger.warning(
                        f"Cached result for {id_for_cache} has unexpected type {type(cached_result)}. Expected str. Retrying LLM call."
                    )

        if (
            must_load_cache and cached_result is None
        ):  # If we are here, cache load failed or was not suitable
            raise ValueError(
                "must_load_cache is True, but failed to load a valid response from cache."
            )

        import litellm

        current_port: int | None = port
        if self.ports and not current_port:
            if use_loadbalance:
                current_port = self.get_least_used_port()
            else:
                current_port = random.choice(self.ports)

        if current_port:
            effective_kwargs["base_url"] = f"http://{self.host}:{current_port}/v1"

        llm_output_or_outputs: Union[str, BaseModel, List[Union[str, BaseModel]]]
        try:
            dspy_main_input: Union[str, List[Message]]
            if messages is not None:
                dspy_main_input = messages
            elif prompt is not None:
                dspy_main_input = prompt
            else:
                # Depending on LM capabilities, this might be valid if other means of generation are used (e.g. tool use)
                # For now, assume one is needed for typical completion/chat.
                # Consider if _dspy_lm can handle None/empty input gracefully or if an error is better.
                # If dspy.LM expects a non-null primary argument, this will fail there.
                # For safety, let's raise if both are None, assuming typical usage.
                raise ValueError(
                    "Either 'prompt' or 'messages' must be provided for the LLM call."
                )

            llm_outputs_list = self._dspy_lm(
                dspy_main_input,  # Pass as positional argument
                response_format=response_format,  # Pass as keyword argument, dspy will handle it in its **kwargs
                **effective_kwargs,
            )

            if not llm_outputs_list:
                raise ValueError("LLM call returned an empty list.")

            # Convert dict outputs to string to match expected return type
            def convert_output(o):
                if isinstance(o, dict):
                    import json

                    return json.dumps(o)
                return o

            if effective_kwargs.get("n", 1) == 1:
                llm_output_or_outputs = convert_output(llm_outputs_list[0])
            else:
                llm_output_or_outputs = [convert_output(o) for o in llm_outputs_list]

        except (litellm.exceptions.APIError, litellm.exceptions.Timeout) as e_llm:
            t = 3
            base_url_info = effective_kwargs.get("base_url", "N/A")
            log_msg = f"[{base_url_info=}] {type(e_llm).__name__}: {str(e_llm)[:100]}, will sleep for {t}s and retry"
            logger.warning(log_msg)  # Always warn on retry for these
            time.sleep(t)
            return self.__call__(
                prompt=prompt,
                messages=messages,
                response_format=response_format,
                cache=cache,
                retry_count=retry_count + 1,
                port=current_port,
                error=e_llm,
                use_loadbalance=use_loadbalance,
                must_load_cache=must_load_cache,
                max_tokens=max_tokens,
                num_retries=num_retries,
                **kwargs,
            )
        except litellm.exceptions.ContextWindowExceededError as e_cwe:
            logger.error(f"Context window exceeded: {e_cwe}")
            raise
        except Exception as e_generic:
            logger.error(f"Generic error during LLM call: {e_generic}")
            import traceback

            traceback.print_exc()
            raise
        finally:
            if (
                current_port and use_loadbalance is True
            ):  # Ensure use_loadbalance is explicitly True
                _update_port_use(current_port, -1)

        if effective_cache and id_for_cache:
            self.dump_cache(id_for_cache, llm_output_or_outputs)

        # Ensure single return if n=1, which is implied by method signature str | BaseModel
        final_output: Union[str, BaseModel]
        if isinstance(llm_output_or_outputs, list):
            # This should ideally not happen if n=1 was handled correctly above.
            # If it's a list, it means n > 1. The method signature needs to change for that.
            # For now, stick to returning the first element if it's a list.
            logger.warning(
                "LLM returned multiple completions; __call__ expects single. Returning first."
            )
            final_output = llm_output_or_outputs[0]
        else:
            final_output = llm_output_or_outputs  # type: ignore # It's already Union[str, BaseModel]

        if response_format:
            if not isinstance(final_output, response_format):
                if isinstance(final_output, str):
                    logger.warning(
                        f"LLM call returned string, but expected {response_format.__name__}. Attempting parse."
                    )
                    try:
                        import json_repair

                        parsed_dict = json_repair.loads(final_output)
                        if not isinstance(parsed_dict, dict):
                            raise ValueError("Parsed output is not a dict")
                        parsed_dict = {str(k): v for k, v in parsed_dict.items()}
                        parsed_output = response_format(**parsed_dict)
                        if effective_cache and id_for_cache:
                            self.dump_cache(
                                id_for_cache, parsed_output
                            )  # Cache the successfully parsed model
                        return parsed_output
                    except Exception as e_final_parse:
                        logger.error(
                            f"Final attempt to parse LLM string output into {response_format.__name__} failed: {e_final_parse}"
                        )
                        # Retry without cache to force regeneration
                        return self.__call__(
                            prompt=prompt,
                            messages=messages,
                            response_format=response_format,
                            cache=False,
                            retry_count=retry_count + 1,
                            port=current_port,
                            error=e_final_parse,
                            use_loadbalance=use_loadbalance,
                            must_load_cache=False,
                            max_tokens=max_tokens,
                            num_retries=num_retries,
                            **kwargs,
                        )
                else:
                    logger.error(
                        f"LLM output type mismatch. Expected {response_format.__name__} or str, got {type(final_output)}. Raising error."
                    )
                    raise TypeError(
                        f"LLM output type mismatch: expected {response_format.__name__}, got {type(final_output)}"
                    )
            return final_output  # Already a response_format instance
        else:  # No response_format, expect string
            if not isinstance(final_output, str):
                # This could happen if LLM returns structured data and dspy parses it even without response_format
                logger.warning(
                    f"LLM output type mismatch. Expected str, got {type(final_output)}. Attempting to convert to string."
                )
                # Convert to string, or handle as error depending on desired strictness
                return str(final_output)  # Or raise TypeError
            return final_output

    def clear_port_use(self):
        if self.ports:
            _clear_port_use(self.ports)
        else:
            logger.warning("No ports configured to clear usage for.")

    def get_least_used_port(self) -> int:
        if self.ports is None:
            raise ValueError("Ports must be configured to pick the least used port.")
        if not self.ports:
            raise ValueError("Ports list is empty, cannot pick a port.")
        return _pick_least_used_port(self.ports)

    def get_session(
        self,
        system_prompt: Optional[str],
        history: Optional[List[Message]] = None,
        callback=None,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs,  # kwargs are not used by ChatSession constructor
    ) -> ChatSession:
        actual_history = deepcopy(history) if history is not None else []
        return ChatSession(
            self,
            system_prompt=system_prompt,
            history=actual_history,
            callback=callback,
            response_format=response_format,
            # **kwargs, # ChatSession constructor does not accept **kwargs
        )

    def dump_cache(
        self, id: str, result: Union[str, BaseModel, List[Union[str, BaseModel]]]
    ):
        try:
            cache_file = f"~/.cache/oai_lm/{self.model}/{id}.pkl"
            cache_file = os.path.expanduser(cache_file)

            dump_json_or_pickle(result, cache_file)
        except Exception as e:
            logger.warning(f"Cache dump failed: {e}")

    def load_cache(
        self, id: str
    ) -> Optional[Union[str, BaseModel, List[Union[str, BaseModel]]]]:
        try:
            cache_file = f"~/.cache/oai_lm/{self.model}/{id}.pkl"
            cache_file = os.path.expanduser(cache_file)
            if not os.path.exists(cache_file):
                return
            return load_json_or_pickle(cache_file)
        except Exception as e:
            logger.warning(f"Cache load failed for {id}: {e}")  # Added id to log
            return None

    def list_models(self) -> List[str]:
        import openai

        if not self.base_url:
            raise ValueError("Cannot list models: base_url is not configured.")
        if not self.api_key:  # api_key should be set by __init__
            logger.warning(
                "API key not available for listing models. Using default 'abc'."
            )

        api_key_str = str(self.api_key) if self.api_key is not None else "abc"
        base_url_str = str(self.base_url) if self.base_url is not None else None
        if isinstance(self.base_url, float):
            raise TypeError(f"base_url must be a string or None, got float: {self.base_url}")
        client = openai.OpenAI(base_url=base_url_str, api_key=api_key_str)
        page = client.models.list()
        return [d.id for d in page.data]

    @property
    def client(self):
        import openai
        if not self.base_url:
            raise ValueError("Cannot create client: base_url is not configured.")
        if not self.api_key:
            logger.warning("API key not available for client. Using default 'abc'.")

        base_url_str = str(self.base_url) if self.base_url is not None else None
        api_key_str = str(self.api_key) if self.api_key is not None else "abc"
        return openai.OpenAI(base_url=base_url_str, api_key=api_key_str)

    def __getattr__(self, name):
        """
        Delegate any attributes not found in OAI_LM to the underlying dspy.LM instance.
        This makes sure any dspy.LM methods not explicitly defined in OAI_LM are still accessible.
        """
        # Check __dict__ directly to avoid recursion via hasattr
        if "_dspy_lm" in self.__dict__ and hasattr(self._dspy_lm, name):
            return getattr(self._dspy_lm, name)
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    @classmethod
    def get_deepseek_chat(
        cls, api_key: Optional[str] = None, max_tokens: int = 2000, **kwargs
    ):
        api_key_to_pass = cast(
            Optional[str], api_key or os.environ.get("DEEPSEEK_API_KEY")
        )
        return cls(  # Use cls instead of OAI_LM
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
            api_key=api_key_to_pass,
            max_tokens=max_tokens,
            **kwargs,
        )

    @classmethod
    def get_deepseek_reasoner(
        cls, api_key: Optional[str] = None, max_tokens: int = 2000, **kwargs
    ):
        api_key_to_pass = cast(
            Optional[str], api_key or os.environ.get("DEEPSEEK_API_KEY")
        )
        return cls(  # Use cls instead of OAI_LM
            base_url="https://api.deepseek.com/v1",
            model="deepseek-reasoner",
            api_key=api_key_to_pass,
            max_tokens=max_tokens,
            **kwargs,
        )

    @classmethod
    def start_server(
        cls, model_name: str, gpus: str = "4567", port: int = 9150, eager: bool = True
    ):
        cmd = f"svllm serve --model {model_name} --gpus {gpus} -hp localhost:{port}"
        if eager:
            cmd += " --eager"
        session_name = f"vllm_{port}"
        is_session_exists = os.system(f"tmux has-session -t {session_name}")
        logger.info(f"Starting server with command: {cmd}")
        if is_session_exists == 0:
            logger.warning(
                f"Session {session_name} exists, please kill it before running the script"
            )
            # as user if they want to kill the session
            user_input = input(
                f"Session {session_name} exists, do you want to kill it? (y/n): "
            )
            if user_input.lower() == "y":
                os.system(f"tmux kill-session -t {session_name}")
                logger.info(f"Session {session_name} killed")
        os.system(cmd)
        # return subprocess.Popen(shlex.split(cmd))

    # set get_agent is get_session
    get_agent = get_session


LM = OAI_LM
