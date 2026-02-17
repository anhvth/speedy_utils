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
from urllib.parse import urlparse

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
  thinking enable model thinking/reasoning stream (default: false)
"""


@dataclass(slots=True)
class ChatConfig:
    client: str = "8000"
    app_port: int = 5009
    app_host: str = "0.0.0.0"
    model: str | None = None
    api_key: str = "abc"
    thinking: bool = False


def normalize_client_base_url(client: str | int | None) -> str:
    """Normalize client input into an OpenAI-compatible base URL."""
    if client is None:
        client = "8000"

    raw = str(client).strip()
    if not raw:
        raw = "8000"

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


def parse_cli_args(argv: Iterable[str]) -> ChatConfig:
    """Parse key=value arguments, keeping defaults when omitted."""
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
            raise ValueError(
                f"Invalid argument '{raw_arg}'. Expected key=value (example: client=8000)."
            )

        key, value = arg.split("=", 1)
        key = key.strip().lower().replace("-", "_")
        value = value.strip()

        if key == "client":
            config.client = value or "8000"
            continue
        if key in {"port", "app_port"}:
            config.app_port = _parse_positive_int("port", value)
            continue
        if key in {"host", "app_host"}:
            config.app_host = value or "0.0.0.0"
            continue
        if key == "model":
            config.model = value or None
            continue
        if key == "api_key":
            config.api_key = value or "abc"
            continue
        if key == "thinking":
            config.thinking = _parse_bool("thinking", value)
            continue

        raise ValueError(
            "Unknown argument "
            f"'{key}'. Supported keys: client, port, host, model, api_key, thinking."
        )

    return config


def _parse_positive_int(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got '{value}'.") from exc

    if parsed <= 0:
        raise ValueError(f"{name} must be > 0, got {parsed}.")
    return parsed


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disabled"}:
        return False
    raise ValueError(
        f"{name} must be a boolean (true/false/1/0/enabled/disabled), got '{value}'."
    )


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


def _iter_delta_text(value: Any) -> Iterable[str]:
    """Extract text fragments from OpenAI-compatible delta fields."""
    if value is None:
        return
    if isinstance(value, str):
        if value:
            yield value
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                if item:
                    yield item
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text:
                    yield text
                continue
            text = getattr(item, "text", None)
            if isinstance(text, str) and text:
                yield text


def _stream_tokens(
    *,
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
):
    call_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if enable_thinking:
        call_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}

    stream = client.chat.completions.create(
        **call_kwargs,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning_content = getattr(delta, "reasoning_content", None)
        for text in _iter_delta_text(reasoning_content):
            yield "thinking", text

        reasoning = getattr(delta, "reasoning", None)
        for text in _iter_delta_text(reasoning):
            yield "thinking", text

        content = getattr(delta, "content", None)
        for text in _iter_delta_text(content):
            yield "content", text


def _render_streaming_placeholder(placeholder: Any, text: str) -> None:
    safe_text = html.escape(text).replace("\n", "<br>")
    placeholder.markdown(
        f"""
<div class="sp-live-response">
  {safe_text}
  <span class="sp-stream-cursor"></span>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_streaming_blocks(
    placeholder: Any, *, thinking_text: str, answer_text: str, thinking_active: bool
) -> None:
    # Convert markdown-friendly newlines to HTML breaks, preserving code blocks
    def format_text(text: str) -> str:
        if not text:
            return ""
        # Escape HTML first
        safe = html.escape(text)
        # Handle code blocks (preserve them)
        parts = []
        in_code = False
        code_buffer = []

        for line in safe.split("\n"):
            if line.startswith("```"):
                if in_code:
                    # End code block
                    code_content = "\n".join(code_buffer)
                    parts.append(f"<pre><code>{code_content}</code></pre>")
                    code_buffer = []
                    in_code = False
                else:
                    # Start code block
                    in_code = True
            elif in_code:
                code_buffer.append(line)
            else:
                # Regular line - convert newlines to <br> for paragraph breaks
                if line.strip() == "":
                    parts.append("<br>")
                else:
                    parts.append(line + "<br>")

        # Handle any remaining code
        if code_buffer:
            code_content = "\n".join(code_buffer)
            parts.append(f"<pre><code>{code_content}</code></pre>")

        return "".join(parts)

    safe_answer = format_text(answer_text)
    thinking_block = ""

    if thinking_text.strip():
        safe_thinking = format_text(thinking_text)
        if thinking_active:
            thinking_cursor = '<span class="sp-stream-cursor"></span>'
            thinking_block = f"""
<div style="margin-bottom:0.75rem; opacity:0.9;">
  <div style="color:var(--text-muted); font-size:0.8rem; margin-bottom:0.25rem; font-weight:500;">💭 Thinking</div>
  <div class="sp-live-response" style="color:var(--text-secondary); font-size:0.9rem;">{safe_thinking}{thinking_cursor}</div>
</div>
            """
        else:
            thinking_block = f"""
<details class="sp-thinking-details">
  <summary>Thinking ({len(thinking_text.split())} words)</summary>
  <div class="sp-thinking-details-content">{safe_thinking}</div>
</details>
            """

    answer_cursor = '<span class="sp-stream-cursor"></span>' if thinking_active else ""
    placeholder.markdown(
        f"""
{thinking_block}
<div class="sp-live-response">
  {safe_answer}{answer_cursor}
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


def _extract_renderable_chunk(
    buffer: str, force: bool = False, min_chunk_size: int = 8
) -> tuple[str, str]:
    """Return a natural-looking chunk (word/sentence boundary) and remaining buffer.

    For smoother streaming, extracts smaller chunks more frequently.
    """
    if not buffer:
        return "", ""
    if force:
        return buffer, ""

    # If buffer is small, wait for more content
    if len(buffer) < min_chunk_size:
        return "", buffer

    # Find natural break points in priority order
    boundary_chars = ("\n\n", ". ", "! ", "? ", ": ", "; ", ", ", " ", "\n", "\t")

    for boundary in boundary_chars:
        idx = buffer.rfind(boundary, min_chunk_size - len(boundary))
        if idx > 0:
            cut_at = idx + len(boundary)
            return buffer[:cut_at], buffer[cut_at:]

    # If no boundary found but buffer is large enough, cut at min_chunk_size
    if len(buffer) >= min_chunk_size * 2:
        return buffer[:min_chunk_size], buffer[min_chunk_size:]

    return "", buffer


def _render_app(config: ChatConfig) -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "sp_chat requires streamlit. Install it with: uv pip install streamlit"
        ) from exc

    st.set_page_config(
        page_title="Speedy Chat",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg-primary: #0a0a0f;
  --bg-secondary: #12121a;
  --bg-tertiary: #1a1a25;
  --bg-card: rgba(26, 26, 37, 0.6);
  --border-subtle: rgba(255, 255, 255, 0.06);
  --border-strong: rgba(255, 255, 255, 0.12);
  --text-primary: #f0f0f5;
  --text-secondary: #a0a0b0;
  --text-muted: #6b6b7b;
  --accent-primary: #6366f1;
  --accent-secondary: #8b5cf6;
  --accent-glow: rgba(99, 102, 241, 0.3);
  --user-gradient: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.08));
  --assistant-gradient: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(99, 102, 241, 0.06));
  --success: #10b981;
  --warning: #f59e0b;
}

* {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
  background: var(--bg-primary);
  color: var(--text-primary);
}

.stApp::before {
  content: '';
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background:
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.15), transparent),
    radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139, 92, 246, 0.1), transparent);
  pointer-events: none;
  z-index: 0;
}

[data-testid='stAppViewContainer'] {
  position: relative;
  z-index: 1;
}

header[data-testid='stHeader'] {
  background: transparent;
}

[data-testid='stToolbar'],
[data-testid='stDecoration'] {
  display: none !important;
}

.block-container {
  max-width: 1040px;
  padding-top: 1rem;
}

/* Hero Section */
.sp-hero {
  padding: 1.2rem 1.3rem;
  margin: 1rem 0 1.5rem 0;
  background: linear-gradient(
    145deg,
    rgba(17, 27, 47, 0.84),
    rgba(13, 20, 35, 0.7)
  );
  border: 1px solid var(--border-subtle);
  border-radius: 22px;
  backdrop-filter: blur(12px);
  box-shadow:
    0 1px 0 rgba(255, 255, 255, 0.04) inset,
    0 24px 46px rgba(2, 6, 16, 0.44);
  width: 100%;
  box-sizing: border-box;
}

div[data-testid='stChatInput'] textarea,
div[data-testid='stChatInput'] input {
  min-height: 56px !important;
  border-radius: 15px !important;
  background: rgba(12, 21, 37, 0.9) !important;
  border: 1px solid var(--border-subtle) !important;
  color: var(--text-primary) !important;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.32);
}

div[data-testid='stChatInput'] textarea::placeholder,
div[data-testid='stChatInput'] input::placeholder {
  color: var(--text-muted) !important;
}

div[data-testid='stChatInput'] button {
  border-radius: 12px !important;
  border: none !important;
  background: linear-gradient(145deg, #4db5ff, #6f87ff) !important;
  color: #eff6ff !important;
}

@keyframes sp-hero-in {
  from {
    opacity: 0;
    transform: translateY(-10px) scale(0.98);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

.sp-chat-title {
  font-size: 1.75rem;
  font-weight: 700;
  background: linear-gradient(135deg, var(--text-primary), var(--accent-primary));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.25rem;
  letter-spacing: -0.02em;
}

.sp-chat-subtitle {
  color: var(--text-secondary);
  font-size: 0.9rem;
  margin-bottom: 1rem;
}

.sp-meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.sp-meta {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.375rem 0.875rem;
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid var(--border-subtle);
  border-radius: 20px;
  font-size: 0.8rem;
  color: var(--text-secondary);
  transition: all 0.2s ease;
}

.sp-meta:hover {
  background: rgba(255, 255, 255, 0.06);
  border-color: var(--border-strong);
}

.sp-meta-live {
  background: rgba(16, 185, 129, 0.1);
  border-color: rgba(16, 185, 129, 0.3);
  color: var(--success);
}

.sp-meta-live .sp-meta-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--success);
  animation: sp-pulse 2s ease-in-out infinite;
}

@keyframes sp-pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}

.sp-meta-label {
  color: var(--text-muted);
  font-weight: 500;
}

/* Sidebar */
section[data-testid='stSidebar'] {
  background: var(--bg-secondary);
  border-right: 1px solid var(--border-subtle);
}

section[data-testid='stSidebar'] * {
  color: var(--text-primary) !important;
}

section[data-testid='stSidebar'] .stButton > button {
  background: var(--accent-primary) !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 500 !important;
  transition: all 0.2s ease !important;
}

section[data-testid='stSidebar'] .stButton > button:hover {
  background: var(--accent-secondary) !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px var(--accent-glow);
}

/* Chat Messages */
div[data-testid='stChatMessage'] {
  border-radius: 12px !important;
  border: 1px solid var(--border-subtle) !important;
  background: var(--assistant-gradient), var(--bg-card) !important;
  padding: 1rem 1.25rem !important;
  margin-bottom: 0.75rem !important;
  animation: sp-message-in 0.4s cubic-bezier(0.16, 1, 0.3, 1);
  transition: all 0.2s ease;
}

@keyframes sp-message-in {
  from {
    opacity: 0;
    transform: translateY(20px) scale(0.95);
  }
  to {
    opacity: 1;
    transform: translateY(0) scale(1);
  }
}

div[data-testid='stChatMessage']:hover {
  border-color: var(--border-strong) !important;
  transform: translateY(-1px);
}

div[data-testid='stChatMessage'][aria-label*='user'],
div[data-testid='stChatMessage'][aria-label*='User'] {
  background: var(--user-gradient), var(--bg-card) !important;
  border-color: rgba(99, 102, 241, 0.2) !important;
}

div[data-testid='stChatMessage'] [data-testid='stMarkdownContainer'] p,
div[data-testid='stChatMessage'] [data-testid='stMarkdownContainer'] li {
  color: var(--text-primary) !important;
  line-height: 1.7;
  font-size: 0.95rem;
}

div[data-testid='stChatMessage'] [data-testid='stMarkdownContainer'] code {
  font-family: 'JetBrains Mono', monospace !important;
  background: rgba(0, 0, 0, 0.3) !important;
  padding: 0.2rem 0.4rem !important;
  border-radius: 4px !important;
  font-size: 0.85em !important;
}

/* Streaming Display */
.sp-live-response {
  color: var(--text-primary);
  line-height: 1.7;
  font-size: 0.95rem;
}

.sp-stream-cursor {
  display: inline-block;
  width: 2px;
  height: 1.2em;
  margin-left: 2px;
  vertical-align: text-bottom;
  background: var(--accent-primary);
  border-radius: 1px;
  animation: sp-cursor-blink 1s ease-in-out infinite;
  box-shadow: 0 0 8px var(--accent-glow);
}

@keyframes sp-cursor-blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

/* Thinking Block */
.sp-thinking-details {
  margin: 0 0 0.75rem;
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  background: rgba(0, 0, 0, 0.2);
  overflow: hidden;
  animation: sp-thinking-expand 0.3s ease;
}

@keyframes sp-thinking-expand {
  from {
    opacity: 0;
    transform: scaleY(0.95);
  }
  to {
    opacity: 1;
    transform: scaleY(1);
  }
}

.sp-thinking-details summary {
  padding: 0.75rem 1rem;
  color: var(--text-secondary);
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  user-select: none;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  transition: all 0.2s ease;
}

.sp-thinking-details summary:hover {
  color: var(--text-primary);
  background: rgba(255, 255, 255, 0.02);
}

.sp-thinking-details summary::before {
  content: '💭';
  font-size: 0.9rem;
}

.sp-thinking-details[open] summary::before {
  content: '🤔';
}

.sp-thinking-details-content {
  padding: 0 1rem 1rem;
  color: var(--text-secondary);
  font-size: 0.9rem;
  line-height: 1.6;
  border-top: 1px solid var(--border-subtle);
  margin-top: 0;
  padding-top: 0.75rem;
}

/* Thinking Indicator */
.sp-thinking {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: var(--bg-tertiary);
  border: 1px solid var(--border-subtle);
  border-radius: 20px;
  color: var(--text-secondary);
  font-size: 0.85rem;
  animation: sp-thinking-in 0.3s ease;
}

@keyframes sp-thinking-in {
  from {
    opacity: 0;
    transform: scale(0.9);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}

.sp-thinking-dot {
  width: 5px;
  height: 5px;
  border-radius: 50%;
  background: var(--accent-primary);
  animation: sp-thinking-dot 1.4s ease-in-out infinite both;
}

.sp-thinking-dot:nth-child(1) { animation-delay: -0.32s; }
.sp-thinking-dot:nth-child(2) { animation-delay: -0.16s; }
.sp-thinking-dot:nth-child(3) { animation-delay: 0s; }

@keyframes sp-thinking-dot {
  0%, 80%, 100% {
    transform: scale(0.6);
    opacity: 0.4;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

.sp-thinking-label {
  font-weight: 500;
  margin-left: 0.25rem;
}

div[data-testid='stChatInput'] textarea {
  min-height: 52px !important;
  max-height: 200px !important;
  background: var(--bg-tertiary) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 12px !important;
  color: var(--text-primary) !important;
  font-size: 0.95rem !important;
  padding: 0.875rem 1rem !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}

div[data-testid='stChatInput'] textarea:focus {
  border-color: var(--accent-primary) !important;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), 0 0 0 3px var(--accent-glow) !important;
}

div[data-testid='stChatInput'] textarea::placeholder {
  color: var(--text-muted) !important;
}

div[data-testid='stChatInput'] button {
  background: var(--accent-primary) !important;
  border: none !important;
  border-radius: 10px !important;
  transition: all 0.2s ease !important;
}

div[data-testid='stChatInput'] button:hover {
  background: var(--accent-secondary) !important;
  transform: scale(1.05);
  box-shadow: 0 4px 12px var(--accent-glow);
}

/* Links */
a {
  color: var(--accent-primary) !important;
  text-decoration: none !important;
  transition: all 0.2s ease;
}

a:hover {
  color: var(--accent-secondary) !important;
  text-decoration: underline !important;
}

/* Scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: var(--border-strong);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--text-muted);
}

/* Mobile Responsive */
@media (max-width: 768px) {
  .block-container {
    padding: 0 1rem 1.5rem;
  }

  div[data-testid='stChatInput'] {
    padding: 0.75rem 1rem 1rem !important;
  }

  .sp-hero {
    padding: 1rem;
    margin: 0.5rem 0 1rem;
  }

  .sp-chat-title {
    font-size: 1.4rem;
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

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 1024
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = ""
    if "enable_thinking" not in st.session_state:
        st.session_state.enable_thinking = config.thinking

    def _on_clear_chat() -> None:
        """Callback to clear chat history."""
        st.session_state.messages = []

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
        st.markdown("### Connection")
        st.code(base_url, language=None)
        if model_error:
            st.warning(f"Model auto-detect failed: {model_error[:180]}")

        selected_model = ""
        if models:
            default_index = 0
            if config.model and config.model in models:
                default_index = models.index(config.model)
            selected_model = st.selectbox("Model", options=models, index=default_index)
        else:
            selected_model = st.text_input("Model", value=config.model or "")

        st.markdown("### Generation")
        # Use unique keys for widgets to avoid rerender issues
        temp_val = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.05,
            key="temp_slider",
        )
        st.session_state.temperature = temp_val

        max_tokens_val = st.number_input(
            "Max tokens",
            min_value=1,
            max_value=32768,
            value=int(st.session_state.max_tokens),
            step=128,
            key="max_tokens_input",
        )
        st.session_state.max_tokens = int(max_tokens_val)

        system_prompt_val = st.text_area(
            "System prompt",
            value=st.session_state.system_prompt,
            height=120,
            placeholder="Optional system message for every request.",
            key="system_prompt_input",
        )
        st.session_state.system_prompt = system_prompt_val

        enable_thinking_val = st.toggle(
            "Enable thinking stream",
            value=bool(st.session_state.enable_thinking),
            help="Sends extra_body.thinking.type=enabled for providers like Z.AI.",
            key="enable_thinking_toggle",
        )
        st.session_state.enable_thinking = enable_thinking_val

        st.button(
            "Clear chat",
            use_container_width=True,
            key="clear_chat_btn",
            on_click=_on_clear_chat,
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Send a prompt")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        answer_chunks: list[str] = []
        thinking_chunks: list[str] = []
        _render_thinking_placeholder(placeholder)

        request_messages: list[dict[str, str]] = []
        system_prompt = st.session_state.system_prompt.strip()
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(st.session_state.messages)

        response_text = ""
        try:
            if not selected_model:
                raise ValueError(
                    "No model available. Provide model=... or make /v1/models reachable."
                )

            displayed_text = ""
            displayed_thinking = ""
            pending_answer_buffer = ""
            pending_thinking_buffer = ""
            last_flush = time.perf_counter()
            flush_interval = 0.03  # 30ms for smooth 30+ FPS updates
            min_chars_before_flush = 3  # Very small for responsive streaming

            for chunk_kind, token in _stream_tokens(
                client=client,
                model=selected_model,
                messages=request_messages,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                enable_thinking=bool(st.session_state.enable_thinking),
            ):
                if chunk_kind == "thinking":
                    thinking_chunks.append(token)
                    pending_thinking_buffer += token
                else:
                    answer_chunks.append(token)
                    pending_answer_buffer += token
                now = time.perf_counter()
                time_since_flush = now - last_flush

                # Determine if we should flush based on multiple criteria
                total_pending = len(pending_answer_buffer) + len(
                    pending_thinking_buffer
                )
                has_line_break = (
                    "\n" in pending_answer_buffer or "\n" in pending_thinking_buffer
                )
                has_punctuation = any(
                    pending_answer_buffer.endswith(p)
                    or pending_thinking_buffer.endswith(p)
                    for p in (". ", "! ", "? ", ": ", "; ", ", ")
                )
                buffer_large = total_pending >= 48
                time_to_flush = time_since_flush >= flush_interval
                has_min_content = total_pending >= min_chars_before_flush

                should_flush = (
                    (time_to_flush and has_min_content)
                    or buffer_large
                    or (has_line_break and has_min_content)
                    or has_punctuation
                )

                if should_flush:
                    # Extract renderable content from buffers
                    force_flush = buffer_large
                    answer_to_render = ""
                    thinking_to_render = ""

                    if pending_answer_buffer:
                        answer_to_render, pending_answer_buffer = (
                            _extract_renderable_chunk(
                                pending_answer_buffer,
                                force=force_flush,
                                min_chunk_size=4,
                            )
                        )
                        if answer_to_render:
                            displayed_text += answer_to_render

                    if pending_thinking_buffer:
                        thinking_to_render, pending_thinking_buffer = (
                            _extract_renderable_chunk(
                                pending_thinking_buffer,
                                force=force_flush,
                                min_chunk_size=4,
                            )
                        )
                        if thinking_to_render:
                            displayed_thinking += thinking_to_render

                    # Only update UI if there's something new to show
                    if answer_to_render or thinking_to_render:
                        _render_streaming_blocks(
                            placeholder,
                            thinking_text=displayed_thinking,
                            answer_text=displayed_text,
                            thinking_active=True,
                        )
                        last_flush = now

            if pending_thinking_buffer:
                chunk_to_render, pending_thinking_buffer = _extract_renderable_chunk(
                    pending_thinking_buffer, force=True
                )
                displayed_thinking += chunk_to_render
            if pending_answer_buffer:
                chunk_to_render, pending_answer_buffer = _extract_renderable_chunk(
                    pending_answer_buffer, force=True
                )
                displayed_text += chunk_to_render
            if displayed_text or displayed_thinking:
                _render_streaming_blocks(
                    placeholder,
                    thinking_text=displayed_thinking,
                    answer_text=displayed_text,
                    thinking_active=False,
                )

            response_text = "".join(answer_chunks).strip() or "(empty response)"
        except Exception as exc:
            response_text = f"Request failed: {exc}"
            placeholder.error(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})


def _launch_streamlit(config: ChatConfig) -> int:
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "sp_chat requires streamlit. Install it with: uv pip install streamlit",
            file=sys.stderr,
        )
        return 1

    script_path = Path(__file__).resolve()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script_path),
        "--server.port",
        str(config.app_port),
        "--server.address",
        config.app_host,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--",
        f"client={config.client}",
        f"api_key={config.api_key}",
        f"thinking={str(config.thinking).lower()}",
    ]
    if config.model:
        command.append(f"model={config.model}")

    display_host = (
        "localhost" if config.app_host in {"0.0.0.0", "::"} else config.app_host
    )
    print(f"Launching chat UI at http://{display_host}:{config.app_port}")
    return subprocess.run(command, check=False).returncode


def main() -> int:
    try:
        config = parse_cli_args(sys.argv[1:])
    except SystemExit as exc:
        print(exc)
        return 0
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(HELP_TEXT, file=sys.stderr)
        return 2

    if _is_running_in_streamlit():
        _render_app(config)
        return 0
    return _launch_streamlit(config)


if __name__ == "__main__":
    raise SystemExit(main())
