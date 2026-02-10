#!/usr/bin/env python3
"""Simple Streamlit chat UI for quickly testing vLLM servers."""

from __future__ import annotations

import html
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from llm_utils.lm.openai_memoize import MOpenAI


HELP_TEXT = """\
sp_chat: Streamlit chat UI for vLLM

Usage:
  sp_chat
  sp_chat client=8000
  sp_chat client=http://10.0.0.3:8000/v1 port=5010 model=Qwen/Qwen2.5-7B-Instruct

Supported key=value args:
  client   vLLM client endpoint or port (default: 8000)
  port     streamlit web port (default: 5009)
  host     streamlit bind host (default: 0.0.0.0)
  model    fixed model id (default: auto-detect from /v1/models)
  api_key  API key for OpenAI-compatible endpoint (default: abc)
"""


@dataclass(slots=True)
class ChatConfig:
    client: str = '8000'
    app_port: int = 5009
    app_host: str = '0.0.0.0'
    model: str | None = None
    api_key: str = 'abc'


def normalize_client_base_url(client: str | int | None) -> str:
    """Normalize client input into an OpenAI-compatible base URL."""
    if client is None:
        client = '8000'

    raw = str(client).strip()
    if not raw:
        raw = '8000'

    if raw.isdigit():
        base_url = f'http://localhost:{raw}'
    elif raw.startswith(('http://', 'https://')):
        base_url = raw.rstrip('/')
    elif ':' in raw:
        base_url = f'http://{raw}'.rstrip('/')
    else:
        base_url = f'http://localhost:{raw}'

    if not base_url.endswith('/v1'):
        base_url = f'{base_url}/v1'
    return base_url


def parse_cli_args(argv: Iterable[str]) -> ChatConfig:
    """Parse key=value arguments, keeping defaults when omitted."""
    config = ChatConfig()

    for raw_arg in argv:
        arg = raw_arg.strip()
        if not arg:
            continue
        if arg in {'help', '-h', '--help'}:
            raise SystemExit(HELP_TEXT)
        if arg.startswith('--'):
            arg = arg[2:]

        if '=' not in arg:
            raise ValueError(
                f"Invalid argument '{raw_arg}'. Expected key=value (example: client=8000)."
            )

        key, value = arg.split('=', 1)
        key = key.strip().lower().replace('-', '_')
        value = value.strip()

        if key == 'client':
            config.client = value or '8000'
            continue
        if key in {'port', 'app_port'}:
            config.app_port = _parse_positive_int('port', value)
            continue
        if key in {'host', 'app_host'}:
            config.app_host = value or '0.0.0.0'
            continue
        if key == 'model':
            config.model = value or None
            continue
        if key == 'api_key':
            config.api_key = value or 'abc'
            continue

        raise ValueError(
            f"Unknown argument '{key}'. Supported keys: client, port, host, model, api_key."
        )

    return config


def _parse_positive_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got '{value}'.") from exc

    if parsed <= 0:
        raise ValueError(f'{name} must be > 0, got {parsed}.')
    return parsed


def _is_running_in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


def _list_models_safe(client: Any) -> tuple[list[str], str | None]:
    try:
        models = client.models.list().data
        model_ids = [model.id for model in models]
        return model_ids, None
    except Exception as exc:
        return [], str(exc)


def _stream_tokens(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, 'content', None)
        if content:
            yield content


def _render_streaming_placeholder(placeholder: Any, text: str) -> None:
    safe_text = html.escape(text).replace('\n', '<br>')
    placeholder.markdown(
        f"""
<div class="sp-live-response">
  {safe_text}
  <span class="sp-stream-cursor"></span>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_thinking_placeholder(placeholder: Any) -> None:
    placeholder.markdown(
        """
<div class="sp-thinking">
  <span class="sp-thinking-dot"></span>
  <span class="sp-thinking-dot"></span>
  <span class="sp-thinking-dot"></span>
  <span class="sp-thinking-label">Thinking</span>
</div>
        """,
        unsafe_allow_html=True,
    )


def _extract_renderable_chunk(buffer: str, force: bool = False) -> tuple[str, str]:
    """Return a natural-looking chunk (word/sentence boundary) and remaining buffer."""
    if not buffer:
        return '', ''
    if force:
        return buffer, ''

    boundary_chars = ('\n', ' ', '\t', '.', '!', '?', ':', ';', ',')
    cut_index = max(buffer.rfind(ch) for ch in boundary_chars)
    if cut_index <= 0:
        return '', buffer

    cut_at = cut_index + 1
    return buffer[:cut_at], buffer[cut_at:]


def _render_app(config: ChatConfig) -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            'sp_chat requires streamlit. Install it with: uv pip install streamlit'
        ) from exc

    st.set_page_config(
        page_title='Speedy Chat',
        page_icon=':speech_balloon:',
        layout='wide',
    )

    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&family=Space+Grotesk:wght@500;700&display=swap');

:root,
[data-theme='light'],
[data-theme='dark'] {
  --sp-bg-0: #060a14;
  --sp-bg-1: #0a1120;
  --sp-bg-2: #11192d;
  --sp-surface: rgba(16, 25, 43, 0.72);
  --sp-surface-strong: rgba(12, 20, 35, 0.88);
  --sp-border: rgba(106, 142, 196, 0.34);
  --sp-border-strong: rgba(122, 178, 255, 0.52);
  --sp-text: #edf4ff;
  --sp-muted: #a4b7d3;
  --sp-accent: #5dc1ff;
  --sp-accent-2: #83a6ff;
  --sp-user-bg: linear-gradient(128deg, rgba(51, 127, 255, 0.25), rgba(84, 200, 255, 0.14));
  --sp-assistant-bg: linear-gradient(128deg, rgba(115, 141, 255, 0.22), rgba(58, 88, 143, 0.16));
}

* {
  font-family: 'Manrope', sans-serif;
}

.stApp {
  background:
    radial-gradient(900px 600px at -8% -12%, rgba(57, 129, 255, 0.22), transparent 62%),
    radial-gradient(950px 700px at 110% -10%, rgba(105, 95, 255, 0.18), transparent 58%),
    linear-gradient(180deg, var(--sp-bg-0), var(--sp-bg-1) 48%, var(--sp-bg-2));
  color: var(--sp-text);
}

.stApp::before,
.stApp::after {
  content: '';
  position: fixed;
  width: 22rem;
  height: 22rem;
  pointer-events: none;
  z-index: 0;
  filter: blur(72px);
  opacity: 0.48;
}

.stApp::before {
  top: 9%;
  right: 13%;
  background: rgba(83, 194, 255, 0.32);
}

.stApp::after {
  bottom: 12%;
  left: 8%;
  background: rgba(117, 124, 255, 0.28);
}

[data-testid='stAppViewContainer'] > .main,
section[data-testid='stSidebar'] {
  position: relative;
  z-index: 1;
}

header[data-testid='stHeader'] {
  background: transparent;
}

[data-testid='stToolbar'],
[data-testid='stDecoration'] {
  visibility: hidden;
}

.block-container {
  max-width: 1040px;
  padding-top: 1rem;
}

.sp-hero {
  padding: 1.2rem 1.3rem;
  border: 1px solid var(--sp-border);
  border-radius: 22px;
  background: linear-gradient(
    145deg,
    rgba(17, 27, 47, 0.84),
    rgba(13, 20, 35, 0.7)
  );
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.04) inset,
    0 24px 46px rgba(2, 6, 16, 0.44);
  backdrop-filter: blur(12px);
  margin-bottom: 1rem;
}

.sp-chat-title {
  font-family: 'Space Grotesk', sans-serif;
  letter-spacing: 0.2px;
  font-size: 2.1rem;
  font-weight: 700;
  color: var(--sp-text);
  margin-bottom: 0.2rem;
}

.sp-chat-subtitle {
  color: var(--sp-muted);
  font-size: 1.02rem;
  margin-bottom: 0.85rem;
}

.sp-meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.55rem;
}

.sp-meta {
  display: inline-flex;
  align-items: center;
  gap: 0.42rem;
  border: 1px solid var(--sp-border);
  border-radius: 999px;
  padding: 0.34rem 0.78rem;
  background: rgba(20, 33, 56, 0.64);
  color: var(--sp-text);
  font-size: 0.82rem;
  white-space: nowrap;
}

.sp-meta-live {
  border-color: rgba(132, 211, 255, 0.48);
  background: rgba(26, 49, 83, 0.73);
}

.sp-meta-live .sp-meta-dot {
  width: 7px;
  height: 7px;
  border-radius: 999px;
  background: #77d5ff;
  box-shadow: 0 0 0 rgba(119, 213, 255, 0.5);
  animation: sp-live-pulse 1.3s ease-out infinite;
}

.sp-meta-label {
  color: var(--sp-muted);
  font-weight: 600;
}

section[data-testid='stSidebar'] {
  background: linear-gradient(
    180deg,
    rgba(10, 16, 30, 0.97),
    rgba(7, 12, 24, 0.97)
  );
  border-right: 1px solid rgba(104, 144, 201, 0.22);
}

section[data-testid='stSidebar'] * {
  color: var(--sp-text) !important;
}

section[data-testid='stSidebar'] [data-baseweb='select'] > div,
section[data-testid='stSidebar'] textarea,
section[data-testid='stSidebar'] input,
section[data-testid='stSidebar'] .stCodeBlock {
  background: var(--sp-surface-strong) !important;
  border-color: var(--sp-border) !important;
  color: var(--sp-text) !important;
}

section[data-testid='stSidebar'] textarea::placeholder,
section[data-testid='stSidebar'] input::placeholder {
  color: var(--sp-muted) !important;
}

section[data-testid='stSidebar'] .stButton > button {
  border-radius: 13px;
  border: 1px solid var(--sp-border);
  background: rgba(20, 33, 56, 0.75);
  color: var(--sp-text);
}

section[data-testid='stSidebar'] .stButton > button:hover {
  border-color: var(--sp-border-strong);
  transform: translateY(-1px);
}

div[data-testid='stChatMessage'] {
  border-radius: 18px !important;
  border: 1px solid var(--sp-border) !important;
  background:
    var(--sp-assistant-bg),
    rgba(13, 21, 38, 0.78) !important;
  box-shadow: 0 14px 30px rgba(3, 7, 20, 0.34);
  padding: 0.22rem 0.38rem;
  margin-bottom: 0.6rem;
  animation: sp-fade-up 0.2s ease-out;
}

div[data-testid='stChatMessage'][aria-label*='user'],
div[data-testid='stChatMessage'][aria-label*='User'] {
  background:
    var(--sp-user-bg),
    rgba(11, 18, 32, 0.85) !important;
  border-color: var(--sp-border-strong) !important;
}

div[data-testid='stChatMessage'] [data-testid='stMarkdownContainer'] p,
div[data-testid='stChatMessage'] [data-testid='stMarkdownContainer'] li,
div[data-testid='stChatMessage'] [data-testid='stMarkdownContainer'] span {
  color: var(--sp-text) !important;
  line-height: 1.6;
}

.sp-live-response {
  color: var(--sp-text);
  line-height: 1.62;
  white-space: normal;
}

.sp-stream-cursor {
  display: inline-block;
  width: 0.55ch;
  height: 1.05em;
  margin-left: 0.2rem;
  vertical-align: -0.12em;
  border-radius: 2px;
  background: linear-gradient(180deg, #8ed6ff, #5f8dff);
  box-shadow: 0 0 12px rgba(114, 185, 255, 0.8);
  animation: sp-cursor-blink 1.05s steps(2, end) infinite;
}

.sp-thinking {
  display: inline-flex;
  align-items: center;
  gap: 0.32rem;
  border: 1px solid var(--sp-border);
  border-radius: 999px;
  background: rgba(18, 30, 51, 0.75);
  padding: 0.34rem 0.66rem;
  color: var(--sp-muted);
}

.sp-thinking-dot {
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: #8ed6ff;
  opacity: 0.6;
  animation: sp-thinking-bounce 1.2s ease-in-out infinite;
}

.sp-thinking-dot:nth-child(2) {
  animation-delay: 0.16s;
}

.sp-thinking-dot:nth-child(3) {
  animation-delay: 0.32s;
}

.sp-thinking-label {
  margin-left: 0.2rem;
  font-size: 0.82rem;
  font-weight: 600;
}

div[data-testid='stChatInput'] textarea,
div[data-testid='stChatInput'] input {
  min-height: 56px !important;
  border-radius: 15px !important;
  background: rgba(12, 21, 37, 0.9) !important;
  border: 1px solid var(--sp-border) !important;
  color: var(--sp-text) !important;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.32);
}

div[data-testid='stChatInput'] textarea::placeholder,
div[data-testid='stChatInput'] input::placeholder {
  color: var(--sp-muted) !important;
}

div[data-testid='stChatInput'] button {
  border-radius: 12px !important;
  border: none !important;
  background: linear-gradient(145deg, #4db5ff, #6f87ff) !important;
  color: #eff6ff !important;
}

a {
  color: #8fd0ff !important;
}

@keyframes sp-fade-up {
  from {
    opacity: 0;
    transform: translateY(4px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

@keyframes sp-cursor-blink {
  0%,
  49% {
    opacity: 1;
  }
  50%,
  100% {
    opacity: 0.25;
  }
}

@keyframes sp-thinking-bounce {
  0%,
  80%,
  100% {
    transform: translateY(0);
    opacity: 0.55;
  }
  40% {
    transform: translateY(-2px);
    opacity: 1;
  }
}

@keyframes sp-live-pulse {
  0% {
    box-shadow: 0 0 0 0 rgba(119, 213, 255, 0.45);
  }
  70% {
    box-shadow: 0 0 0 6px rgba(119, 213, 255, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(119, 213, 255, 0);
  }
}
</style>
        """,
        unsafe_allow_html=True,
    )

    base_url = normalize_client_base_url(config.client)
    client = MOpenAI(base_url=base_url, api_key=config.api_key, cache=False)

    models, model_error = _list_models_safe(client)
    if not models and config.model:
        models = [config.model]

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 1024
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = ''

    st.markdown(
        f"""
<div class="sp-hero">
  <div class="sp-chat-title">Speedy vLLM Chat</div>
  <div class="sp-chat-subtitle">Fast streaming playground for OpenAI-compatible local models.</div>
  <div class="sp-meta-row">
    <span class="sp-meta"><span class="sp-meta-label">Endpoint</span>{base_url}</span>
    <span class="sp-meta sp-meta-live"><span class="sp-meta-dot"></span><span class="sp-meta-label">Transport</span>Real-time stream</span>
    <span class="sp-meta"><span class="sp-meta-label">UX</span>Latency-first</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown('### Connection')
        st.code(base_url, language=None)
        if model_error:
            st.warning(f'Model auto-detect failed: {model_error[:180]}')

        selected_model = ''
        if models:
            default_index = 0
            if config.model and config.model in models:
                default_index = models.index(config.model)
            selected_model = st.selectbox('Model', options=models, index=default_index)
        else:
            selected_model = st.text_input('Model', value=config.model or '')

        st.markdown('### Generation')
        st.session_state.temperature = st.slider(
            'Temperature',
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.05,
        )
        st.session_state.max_tokens = int(
            st.number_input(
                'Max tokens',
                min_value=1,
                max_value=32768,
                value=int(st.session_state.max_tokens),
                step=128,
            )
        )
        st.session_state.system_prompt = st.text_area(
            'System prompt',
            value=st.session_state.system_prompt,
            height=120,
            placeholder='Optional system message for every request.',
        )
        if st.button('Clear chat', use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input('Send a prompt')
    if not prompt:
        return

    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('assistant'):
        placeholder = st.empty()
        chunks: list[str] = []
        _render_thinking_placeholder(placeholder)

        request_messages: list[dict[str, str]] = []
        system_prompt = st.session_state.system_prompt.strip()
        if system_prompt:
            request_messages.append({'role': 'system', 'content': system_prompt})
        request_messages.extend(st.session_state.messages)

        response_text = ''
        try:
            if not selected_model:
                raise ValueError(
                    'No model available. Provide model=... or make /v1/models reachable.'
                )

            displayed_text = ''
            pending_buffer = ''
            last_flush = time.perf_counter()

            for token in _stream_tokens(
                client=client,
                model=selected_model,
                messages=request_messages,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
            ):
                chunks.append(token)
                pending_buffer += token
                now = time.perf_counter()

                should_attempt_flush = (
                    len(pending_buffer) >= 24
                    or '\n' in pending_buffer
                    or pending_buffer.endswith(('.', '!', '?', ':', ';', ','))
                    or (now - last_flush) >= 0.05
                )
                if should_attempt_flush:
                    force_flush = len(pending_buffer) >= 72
                    chunk_to_render, pending_buffer = _extract_renderable_chunk(
                        pending_buffer, force=force_flush
                    )
                    if chunk_to_render:
                        displayed_text += chunk_to_render
                        _render_streaming_placeholder(placeholder, displayed_text)
                        last_flush = now

            if pending_buffer:
                chunk_to_render, pending_buffer = _extract_renderable_chunk(
                    pending_buffer, force=True
                )
                displayed_text += chunk_to_render
                _render_streaming_placeholder(placeholder, displayed_text)

            response_text = ''.join(chunks).strip() or '(empty response)'
            placeholder.markdown(response_text)
        except Exception as exc:
            response_text = f'Request failed: {exc}'
            placeholder.error(response_text)

    st.session_state.messages.append({'role': 'assistant', 'content': response_text})


def _launch_streamlit(config: ChatConfig) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            'sp_chat requires streamlit. Install it with: uv pip install streamlit',
            file=sys.stderr,
        )
        return 1

    script_path = Path(__file__).resolve()
    command = [
        sys.executable,
        '-m',
        'streamlit',
        'run',
        str(script_path),
        '--server.port',
        str(config.app_port),
        '--server.address',
        config.app_host,
        '--server.headless',
        'true',
        '--browser.gatherUsageStats',
        'false',
        '--',
        f'client={config.client}',
        f'api_key={config.api_key}',
    ]
    if config.model:
        command.append(f'model={config.model}')

    display_host = (
        'localhost' if config.app_host in {'0.0.0.0', '::'} else config.app_host
    )
    print(f'Launching chat UI at http://{display_host}:{config.app_port}')
    return subprocess.run(command, check=False).returncode


def main() -> int:
    try:
        config = parse_cli_args(sys.argv[1:])
    except SystemExit as exc:
        print(exc)
        return 0
    except ValueError as exc:
        print(f'Error: {exc}', file=sys.stderr)
        print(HELP_TEXT, file=sys.stderr)
        return 2

    if _is_running_in_streamlit():
        _render_app(config)
        return 0
    return _launch_streamlit(config)


if __name__ == '__main__':
    raise SystemExit(main())
