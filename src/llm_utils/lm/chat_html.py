from typing import Any, Optional, cast
from .sync_lm import LM, Messages, LegacyMsgs, RawMsgs
import sys

# Configuration
DEFAULT_FONT_SIZE = 1  # Base font size in pixels
DEFAULT_CODE_FONT_SIZE = 1  # Code font size in pixels
DEFAULT_PADDING = [1] * 4  # Padding [top, right, bottom, left] in pixels
DEFAULT_INNER_PADDING = [1] * 4  # Inner padding [top, right, bottom, left]
thinking_tag = "think"
# Jupyter notebook detection and imports
try:
    from IPython.display import display, HTML
    from IPython import get_ipython

    JUPYTER_AVAILABLE = True
except ImportError:
    JUPYTER_AVAILABLE = False


def _is_jupyter_notebook() -> bool:
    """Check if running in Jupyter notebook environment."""
    if not JUPYTER_AVAILABLE:
        return False
    try:
        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:
        return False


def _parse_thinking_content(content: str) -> tuple[str, str]:
    """Parse content to separate thinking and answer sections during streaming."""
    import re

    # For streaming: detect if we're currently in thinking mode
    think_start_match = re.search(r"<think[^>]*>", content, re.IGNORECASE)
    if not think_start_match:
        return "", content

    think_start_pos = think_start_match.end()

    # Look for closing tag
    think_end_match = re.search(
        r"</think[^>]*>", content[think_start_pos:], re.IGNORECASE
    )

    if think_end_match:
        # We have complete thinking section
        thinking_content = content[
            think_start_pos : think_start_pos + think_end_match.start()
        ].strip()
        # Everything after </think> is answer content
        answer_start = think_start_pos + think_end_match.end()
        answer_content = content[answer_start:].strip()
        return thinking_content, answer_content
    else:
        # Still in thinking mode (streaming), no closing tag yet
        thinking_content = content[think_start_pos:].strip()
        return thinking_content, ""


def _get_chat_html_template(
    content: str,
    font_size: int = DEFAULT_FONT_SIZE,
    padding: list[int] = DEFAULT_PADDING,
    inner_padding: list[int] = DEFAULT_INNER_PADDING,
) -> str:
    """Generate HTML template with improved styling for chat display."""
    code_font_size = max(font_size - 1, 10)  # Code slightly smaller, min 10px

    # Parse thinking and answer content
    thinking_content, answer_content = _parse_thinking_content(content)

    # Format padding as CSS value - reduce outer padding more
    outer_padding_css = f"2px {padding[1]}px 2px {padding[3]}px"
    inner_padding_css = f"2px {inner_padding[1]}px 2px {inner_padding[3]}px"

    # Build thinking section HTML if present
    thinking_html = ""
    if thinking_content:
        # Show as open during streaming, closed when complete
        is_complete = "</think" in content.lower()
        open_attr = "" if is_complete else "open"

        thinking_html = f"""
        <details {open_attr} style="
            margin-bottom: 4px;
            border: 1px solid #d1d9e0;
            border-radius: 4px;
            background-color: #f8f9fa;
        ">
            <summary style="
                padding: 3px 8px;
                background-color: #e9ecef;
                border-radius: 3px 3px 0 0;
                cursor: pointer;
                font-weight: 500;
                color: #495057;
                user-select: none;
                border-bottom: 1px solid #d1d9e0;
                font-size: {font_size - 1}px;
            ">
                🤔 Thinking{'...' if not is_complete else ''}
            </summary>
            <div style="
                padding: 4px 8px;
                background-color: #f8f9fa;
                border-radius: 0 0 3px 3px;
            ">
                <pre style="
                    margin: 0;
                    padding: 0;
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                    font-size: {code_font_size - 1}px;
                    line-height: 1.3;
                    background: transparent;
                    border: none;
                    color: #6c757d;
                ">{thinking_content}</pre>
            </div>
        </details>
        """

    return f"""
    <div style="
        border: 1px solid #d0d7de;
        border-radius: 6px;
        padding: {outer_padding_css};
        margin: 2px 0;
        background-color: #f6f8fa;
        color: #24292f;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', Helvetica, Arial, sans-serif;
        font-size: {font_size}px;
        line-height: 1.4;
        white-space: pre-wrap;
        word-wrap: break-word;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    ">
        <div style="
            background-color: #fff;
            border: 1px solid #d0d7de;
            border-radius: 4px;
            padding: {inner_padding_css};
            color: #24292f;
        ">
            <strong style="color: #0969da;">Assistant:</strong><br>
            {thinking_html}
            <pre style="
                margin: 2px 0 0 0;
                padding: 0;
                white-space: pre-wrap;
                word-wrap: break-word;
                font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
                font-size: {code_font_size}px;
                line-height: 1.4;
                background: transparent;
                border: none;
            ">{answer_content}</pre>
        </div>
    </div>
    """


class LMChatHtml(LM):
    def __init__(
        self,
        *args,
        font_size: int = DEFAULT_FONT_SIZE,
        padding: list[int] = DEFAULT_PADDING,
        inner_padding: list[int] = DEFAULT_INNER_PADDING,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.font_size = font_size
        self.padding = padding
        self.inner_padding = inner_padding

    def chat_stream(
        self,
        prompt: Optional[str] = None,
        messages: Optional[RawMsgs] = None,
        html_mode: bool = False,
        font_size: Optional[int] = None,
        padding: Optional[list[int]] = None,
        inner_padding: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Stream responses from the model with HTML support in Jupyter.
        """
        if prompt is not None:
            messages = [{"role": "user", "content": prompt}]

        assert messages is not None  # for type-checker

        openai_msgs: Messages = (
            self._convert_messages(cast(LegacyMsgs, messages))
            if isinstance(messages[0], dict)  # legacy style
            else cast(Messages, messages)  # already typed
        )
        assert self.model is not None, "Model must be set before streaming."

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_msgs,
            stream=True,
            **kwargs,
        )  # type: ignore

        output_text = ""
        is_jupyter = _is_jupyter_notebook()
        display_font_size = font_size or self.font_size
        display_padding = padding or self.padding
        display_inner_padding = inner_padding or self.inner_padding

        if html_mode and is_jupyter:
            # Create initial display handle
            display_handle = display(HTML(""), display_id=True)

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    chunk_content = chunk.choices[0].delta.content
                    output_text += chunk_content

                    # Update HTML display progressively using improved template
                    html_content = _get_chat_html_template(
                        output_text,
                        font_size=display_font_size,
                        padding=display_padding,
                        inner_padding=display_inner_padding,
                    )
                    if display_handle is not None:
                        display_handle.update(HTML(html_content))
        else:
            # Console streaming mode (original behavior)
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    chunk_content = chunk.choices[0].delta.content
                    print(chunk_content, end="")
                    sys.stdout.flush()
                    output_text += chunk_content

        return output_text
