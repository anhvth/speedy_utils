#!/usr/bin/env python3
"""Minimalist Chainlit chat UI for quickly testing vLLM servers."""

from __future__ import annotations

import os
import sys
import subprocess
import time
import html
from pathlib import Path
from typing import Any, Iterable, List
from urllib.parse import urlparse

# --- Configuration & CLI Parsing ---

HELP_TEXT = """\
sp_chat: Chainlit chat UI for vLLM

Usage:
  sp_chat
  sp_chat client=8000
  sp_chat client=http://10.0.0.3:8000/v1 port=5010 model=Qwen/Qwen2.5-7B-Instruct

Supported key=value args:
  client   vLLM client endpoint or port (default: http://localhost:4343/v1)
  port     web port (default: 5009)
  host     bind host (default: 0.0.0.0)
  model    fixed model id (default: auto-detect from /v1/models)
  api_key  API key for OpenAI-compatible endpoint (default: abc)
  thinking enable model thinking/reasoning stream (default: true)
"""

DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SYSTEM_PROMPT = ""

class ChatConfig:
    __slots__ = ("client", "app_port", "app_host", "model", "api_key", "thinking")

    def __init__(
        self,
        client: str = "http://localhost:4343/v1",
        app_port: int = 5009,
        app_host: str = "0.0.0.0",
        model: str | None = None,
        api_key: str = "abc",
        thinking: bool = True,
    ):
        self.client = client
        self.app_port = app_port
        self.app_host = app_host
        self.model = model
        self.api_key = api_key
        self.thinking = thinking

    def __repr__(self) -> str:
        return (
            "ChatConfig("
            f"client={self.client!r}, "
            f"app_port={self.app_port!r}, "
            f"app_host={self.app_host!r}, "
            f"model={self.model!r}, "
            f"api_key={self.api_key!r}, "
            f"thinking={self.thinking!r}"
            ")"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ChatConfig):
            return False
        return (
            self.client == other.client
            and self.app_port == other.app_port
            and self.app_host == other.app_host
            and self.model == other.model
            and self.api_key == other.api_key
            and self.thinking == other.thinking
        )


def _build_history_title(messages: list[dict[str, str]], index: int) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "").strip()
        if not content:
            continue
        compact = " ".join(content.split())
        if len(compact) > 30:
            compact = f"{compact[:27]}..."
        return f"Chat {index}: {compact}"
    return f"Chat {index}"


def _archive_current_chat(session_state: Any) -> None:
    messages = list(session_state.get("messages", []))
    if not messages:
        return

    history_index = int(session_state.get("history_counter", 1))
    copied_messages = [dict(msg) for msg in messages]
    history_entry = {
        "id": history_index,
        "title": _build_history_title(copied_messages, history_index),
        "messages": copied_messages,
        "turn_count": len(copied_messages),
    }

    history = session_state.setdefault("chat_history", [])
    history.append(history_entry)
    session_state["dimmed_messages"] = [dict(msg) for msg in copied_messages]
    session_state["history_counter"] = history_index + 1


def _reset_chat_state(session_state: Any, config: ChatConfig) -> None:
    session_state["messages"] = []
    session_state["temperature"] = DEFAULT_TEMPERATURE
    session_state["max_tokens"] = DEFAULT_MAX_TOKENS
    session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    session_state["enable_thinking"] = bool(config.thinking)

    # Keep these mirrored keys for backward compatibility with previous UI state shape.
    session_state["temp_slider"] = DEFAULT_TEMPERATURE
    session_state["max_tokens_input"] = DEFAULT_MAX_TOKENS
    session_state["system_prompt_input"] = DEFAULT_SYSTEM_PROMPT
    session_state["enable_thinking_toggle"] = bool(config.thinking)


def _render_streaming_blocks(
    placeholder: Any,
    *,
    thinking_text: str,
    answer_text: str,
) -> None:
    parts: list[str] = []
    cursor = (
        "<span style=\"display:inline-block;width:4px;height:1em;"
        "background:#ECECEC;animation:blink 1s step-end infinite;\"></span>"
    )

    if thinking_text.strip():
        parts.append(
            "<details class=\"sp-thinking-stream\" "
            "style=\"color:#A3A3A3;font-size:0.9em;margin-bottom:1rem;"
            "padding:0.5rem;border-left:2px solid #444;\">"
            "<summary style=\"cursor:pointer;margin-bottom:0.5rem;\">"
            "Thought Process</summary>"
            f"<div style=\"white-space:pre-wrap;\">{html.escape(thinking_text)}"
            f"{cursor if not answer_text else ''}</div></details>"
        )

    if answer_text.strip():
        parts.append(
            f"<div style=\"white-space:pre-wrap;line-height:1.6;\">"
            f"{html.escape(answer_text)}{cursor}</div>"
        )
    elif not thinking_text.strip():
        parts.append(f"<div>{cursor}</div>")

    placeholder.markdown("\n".join(parts), unsafe_allow_html=True)

def normalize_client_base_url(client: str | int | None) -> str:
    if client is None:
        client = "http://localhost:8000/v1"
    raw = str(client).strip()
    if not raw:
        raw = "http://localhost:8000/v1"

    if raw.isdigit():
        return f"http://localhost:{raw}/v1"

    if raw.startswith(("http://", "https://")):
        base_url = raw.rstrip("/")
    elif ":" in raw:
        base_url = f"http://{raw}".rstrip("/")
    else:
        return f"http://localhost:{raw}/v1"

    parsed = urlparse(base_url)
    if parsed.path in {"", "/"}:
        return f"{base_url}/v1"
    return base_url

def _parse_positive_int(name: str, value: str) -> int:
    parsed = int(value)
    if parsed <= 0: raise ValueError(f"{name} must be > 0.")
    return parsed

def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}: return True
    if normalized in {"0", "false", "no", "off", "disabled"}: return False
    raise ValueError(f"{name} must be a boolean.")

def parse_cli_args(argv: Iterable[str]) -> ChatConfig:
    config = ChatConfig()
    for raw_arg in argv:
        arg = raw_arg.strip()
        if not arg:
            continue
        if arg in {"help", "-h", "--help"}:
            raise SystemExit(HELP_TEXT)
        if arg.startswith("--"):
            arg = arg[2:]

        if "=" not in arg:
            # Maybe it's a positional arg or flag without value?
            # Original code strictly required key=value
            raise ValueError(f"Invalid argument '{raw_arg}'. Expected key=value.")

        key, value = arg.split("=", 1)
        key = key.strip().lower().replace("-", "_")
        value = value.strip()

        if key == "client": config.client = value or "http://localhost:4343/v1"
        elif key in {"port", "app_port"}: config.app_port = _parse_positive_int("port", value)
        elif key in {"host", "app_host"}: config.app_host = value or "0.0.0.0"
        elif key == "model": config.model = value or None
        elif key == "api_key": config.api_key = value or "abc"
        elif key == "thinking": config.thinking = _parse_bool("thinking", value)
        else:
            raise ValueError(f"Unknown argument '{key}'. Supported keys: client, port, host, model, api_key, thinking.")
    return config

# --- Chainlit App Logic ---

def _is_running_in_chainlit() -> bool:
    return os.environ.get("SP_CHAT_RUNNING") == "1"

async def _list_models_async(client: Any) -> tuple[List[str], str | None]:
    try:
        resp = await client.models.list()
        return sorted([m.id for m in resp.data]), None
    except Exception as exc:
        return [], str(exc)

def _list_models_sync(base_url: str, api_key: str) -> tuple[List[str], str | None]:
    """Synchronous model listing used during startup info."""
    from openai import OpenAI
    try:
        c = OpenAI(base_url=base_url, api_key=api_key)
        resp = c.models.list()
        return sorted([m.id for m in resp.data]), None
    except Exception as exc:
        return [], str(exc)

def _setup_chainlit():
    import chainlit as cl
    from chainlit.input_widget import Select, Slider, TextInput, Switch
    from openai import AsyncOpenAI

    base_url = os.environ.get("SP_CHAT_CLIENT", "http://localhost:4343/v1")
    api_key = os.environ.get("SP_CHAT_API_KEY", "abc")
    initial_model = os.environ.get("SP_CHAT_MODEL")
    enable_thinking_default = os.environ.get("SP_CHAT_THINKING", "true") == "true"

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    @cl.on_chat_start
    async def start():
        models, error = await _list_models_async(client)

        if models:
            model_options = models
        elif initial_model:
            model_options = [initial_model]
        else:
            model_options = ["default"]

        default_idx = 0
        if initial_model and initial_model in model_options:
            default_idx = model_options.index(initial_model)

        settings = await cl.ChatSettings(
            [
                Select(
                    id="model",
                    label="Model",
                    values=model_options,
                    initial_index=default_idx,
                ),
                Slider(
                    id="temperature",
                    label="Temperature",
                    initial=DEFAULT_TEMPERATURE,
                    min=0,
                    max=2,
                    step=0.05,
                ),
                Slider(
                    id="max_tokens",
                    label="Max Tokens",
                    initial=DEFAULT_MAX_TOKENS,
                    min=1,
                    max=32768,
                    step=128,
                ),
                TextInput(
                    id="system_prompt",
                    label="System Prompt",
                    initial=DEFAULT_SYSTEM_PROMPT,
                    multiline=True,
                ),
                Switch(
                    id="thinking",
                    label="Enable Thinking",
                    initial=enable_thinking_default,
                ),
            ]
        ).send()

        cl.user_session.set("settings", settings)
        cl.user_session.set("client", client)
        if error:
            # Keep startup clean (no hardcoded first message), only surface real errors.
            await cl.Message(content=f"⚠️ Model fetch error: {error}", author="system").send()

    @cl.on_settings_update
    async def on_settings_update(settings):
        cl.user_session.set("settings", settings)

    @cl.on_message
    async def on_message(message: cl.Message):
        settings = cl.user_session.get("settings")
        aclient: AsyncOpenAI = cl.user_session.get("client")

        model = settings["model"]
        temperature = float(settings["temperature"])
        max_tokens = int(settings["max_tokens"])
        system_prompt = settings["system_prompt"]
        enable_thinking = settings["thinking"]

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(cl.chat_context.to_openai())

        call_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if enable_thinking:
            call_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

        start_time = time.time()

        try:
            stream = await aclient.chat.completions.create(**call_kwargs)

            thinking_completed = False
            final_answer = cl.Message(content="")

            async with cl.Step(name="Thinking") as thinking_step:
                async for chunk in stream:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta

                    reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                    content = delta.content

                    if reasoning and not thinking_completed:
                        await thinking_step.stream_token(reasoning)
                    elif not thinking_completed and content:
                        thought_for = round(time.time() - start_time)
                        thinking_step.name = f"Thought for {thought_for}s"
                        await thinking_step.update()
                        thinking_completed = True
                        break

                # If thinking never got content, close the step
                if not thinking_completed:
                    thought_for = round(time.time() - start_time)
                    thinking_step.name = f"Thought for {thought_for}s"
                    await thinking_step.update()

            # Stream the final answer (continues from where we left off)
            if thinking_completed and content:
                # We already got the first content token above
                await final_answer.stream_token(content)

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    await final_answer.stream_token(delta.content)

            await final_answer.send()

        except Exception as e:
            await cl.Message(content=f"Error: {str(e)}").send()

# --- Launcher ---

def _launch_chainlit(config: ChatConfig) -> int:
    try:
        import chainlit  # noqa
    except ImportError:
        print("Install chainlit via: uv pip install chainlit", file=sys.stderr)
        return 1

    # Pass config via environment variables to the subprocess
    env = os.environ.copy()
    env["SP_CHAT_CLIENT"] = normalize_client_base_url(config.client)
    env["SP_CHAT_API_KEY"] = config.api_key
    if config.model:
        env["SP_CHAT_MODEL"] = config.model
    env["SP_CHAT_THINKING"] = "true" if config.thinking else "false"
    # Marker to indicate we are running the app logic
    env["SP_CHAT_RUNNING"] = "1"
    
    # We use "chainlit run" on THIS file
    script_path = str(Path(__file__).resolve())
    
    cmd = [
        sys.executable, "-m", "chainlit", "run", script_path,
        "--port", str(config.app_port),
        "--host", config.app_host,
        "--headless"
    ]
    
    base_url = env["SP_CHAT_CLIENT"]
    # Print detected models at launch for visibility
    models, err = _list_models_sync(base_url, config.api_key)
    if models:
        print(f"Models: {', '.join(models)}")
    elif err:
        print(f"⚠️  Could not list models ({err}). Will retry in browser.")

    host_display = "localhost" if config.app_host in {"0.0.0.0", "::"} else config.app_host
    print(f"Chat UI → http://{host_display}:{config.app_port}")

    return subprocess.run(cmd, env=env, check=False).returncode

def main() -> int:
    try:
        config = parse_cli_args(sys.argv[1:])
    except SystemExit as exc:
        print(exc)
        return 0
    except ValueError as exc:
        print(f"Error: {exc}\n{HELP_TEXT}", file=sys.stderr)
        return 2

    return _launch_chainlit(config)

# --- Module‑level dispatch ---
# When chainlit imports this file, __name__ != "__main__" but our env marker is set.
if _is_running_in_chainlit():
    _setup_chainlit()
elif __name__ == "__main__":
    sys.exit(main())
