"""
Error handling utilities for clean, user-focused tracebacks.
"""

import inspect
import linecache
import sys
import traceback
from typing import Any, Callable, TypeVar

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
except ImportError:
    Console = None
    Panel = None
    Text = None

F = TypeVar('F', bound=Callable[..., Any])


class CleanTracebackError(Exception):
    """Exception wrapper that provides clean, user-focused tracebacks."""

    def __init__(
        self,
        original_exception: Exception,
        user_traceback: list[traceback.FrameSummary],
        caller_frame: traceback.FrameSummary | None = None,
        func_name: str | None = None,
    ) -> None:
        self.original_exception = original_exception
        self.user_traceback = user_traceback
        self.caller_frame = caller_frame
        self.func_name = func_name

        # Create a focused error message
        tb_str = ''.join(traceback.format_list(user_traceback))
        func_part = f' in {func_name}' if func_name else ''
        msg = (
            f'Error{func_part}:\n'
            f'\nUser code traceback:\n{tb_str}'
            f'{type(original_exception).__name__}: {original_exception}'
        )
        super().__init__(msg)

    def __str__(self) -> str:
        return super().__str__()

    def format_rich(self) -> None:
        """Format and print error with rich panels and code context."""
        if Console is None or Panel is None or Text is None:
            print(str(self), file=sys.stderr)
            return

        console = Console(stderr=True, force_terminal=True)

        # Build traceback display with code context
        tb_parts: list[str] = []

        # Show caller frame first if available
        if self.caller_frame and self.caller_frame.lineno is not None:
            tb_parts.append(
                f'[cyan]{self.caller_frame.filename}[/cyan]:[yellow]{self.caller_frame.lineno}[/yellow] '
                f'in [green]{self.caller_frame.name}[/green]'
            )
            tb_parts.append('')
            context = _get_code_context_rich(self.caller_frame.filename, self.caller_frame.lineno, 3)
            tb_parts.extend(context)
            tb_parts.append('')

        # Show user code frames with context
        for frame in self.user_traceback:
            if frame.lineno is not None:
                func_name = f' {self.func_name}' if self.func_name else ''
                tb_parts.append(
                    f'[cyan]{frame.filename}[/cyan]:[yellow]{frame.lineno}[/yellow] '
                    f'in [green]{frame.name}{func_name}[/green]'
                )
                tb_parts.append('')
                context = _get_code_context_rich(frame.filename, frame.lineno, 3)
                tb_parts.extend(context)
                tb_parts.append('')

        # Print with rich Panel
        console.print()
        console.print(
            Panel(
                '\n'.join(tb_parts),
                title='[bold red]Traceback (most recent call last)[/bold red]',
                border_style='red',
                expand=False,
            )
        )
        console.print(
            f'[bold red]{type(self.original_exception).__name__}[/bold red]: '
            f'{self.original_exception}'
        )
        console.print()


def _get_code_context(filename: str, lineno: int, context_lines: int = 3) -> list[str]:
    """Get code context around a line with line numbers and highlighting."""
    lines: list[str] = []
    start = max(1, lineno - context_lines)
    end = lineno + context_lines

    for i in range(start, end + 1):
        line = linecache.getline(filename, i)
        if not line:
            continue
        line = line.rstrip()
        marker = '❱' if i == lineno else ' '
        lines.append(f'  {i:4d} {marker} {line}')

    return lines


def _get_code_context_rich(filename: str, lineno: int, context_lines: int = 3) -> list[str]:
    """Get code context with rich formatting (colors)."""
    lines: list[str] = []
    start = max(1, lineno - context_lines)
    end = lineno + context_lines

    for i in range(start, end + 1):
        line = linecache.getline(filename, i)
        if not line:
            continue
        line = line.rstrip()
        num_str = f'{i:4d}'

        if i == lineno:
            # Highlight error line
            lines.append(f'[dim]{num_str}[/dim] [red]❱[/red] {line}')
        else:
            # Normal context line
            lines.append(f'[dim]{num_str} │[/dim] {line}')

    return lines


def _filter_traceback_frames(tb_list: list[traceback.FrameSummary]) -> list[traceback.FrameSummary]:
    """Filter traceback frames to show only user code."""
    user_frames = []
    skip_patterns = [
        'site-packages/',
        'dist-packages/',
        'python3.',
        'lib/python',
        'concurrent/futures/',
        'threading.py',
        'multiprocessing/',
        'urllib/',
        'httpx/',
        'httpcore/',
        'openai/',
        'requests/',
        'aiohttp/',
        'urllib3/',
        'speedy_utils/common/',
        'speedy_utils/multi_worker/',
        'llm_utils/lm/',
        'llm_utils/chat_format/',
        'llm_utils/vector_cache/',
    ]

    skip_functions = [
        'wrapper',  # Our decorator wrapper
        '__call__',  # Internal calls
        '__inner_call__',  # Internal calls
        '_worker',  # Multi-thread worker
        '_run_batch',  # Batch runner
    ]

    for frame in tb_list:
        # Skip frames matching skip patterns
        if any(pattern in frame.filename for pattern in skip_patterns):
            continue
        # Skip specific function names
        if frame.name in skip_functions:
            continue
        # Skip frames that are too deep in the call stack (internal implementation)
        if 'speedy_utils' in frame.filename and any(
            name in frame.name for name in ['__inner_call__', '_worker', '_run_batch']
        ):
            continue
        user_frames.append(frame)

    # If no user frames found, keep frames that are likely user code (not deep library internals)
    if not user_frames:
        for frame in reversed(tb_list):
            # Keep if it's not in site-packages and not in standard library
            if ('site-packages/' not in frame.filename and 
                'dist-packages/' not in frame.filename and
                not frame.filename.startswith('/usr/') and
                not frame.filename.startswith('/opt/') and
                'python3.' not in frame.filename and
                frame.name not in skip_functions):
                user_frames.append(frame)
                if len(user_frames) >= 5:  # Keep up to 5 frames
                    break
        user_frames.reverse()

    return user_frames


def clean_traceback(func: F) -> F:
    """Decorator to wrap function calls with clean traceback handling."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            # Get the current traceback
            exc_tb = sys.exc_info()[2]

            if exc_tb is not None:
                tb_list = traceback.extract_tb(exc_tb)

                # Filter to keep only user code frames
                user_frames = _filter_traceback_frames(tb_list)

                # Get caller frame - walk up the stack to find the original caller
                caller_context = None
                frame = inspect.currentframe()
                while frame:
                    frame = frame.f_back
                    if frame and frame.f_code.co_name not in ['wrapper', '__call__', '__inner_call__']:
                        caller_info = inspect.getframeinfo(frame)
                        if not any(skip in caller_info.filename for skip in [
                            'speedy_utils/common/', 'speedy_utils/multi_worker/', 
                            'llm_utils/lm/', 'site-packages/', 'dist-packages/'
                        ]):
                            caller_context = traceback.FrameSummary(
                                caller_info.filename,
                                caller_info.lineno,
                                caller_info.function,
                            )
                            break

                # If we have user frames, create and format our custom exception
                if user_frames:
                    func_name = getattr(func, '__name__', repr(func))
                    clean_error = CleanTracebackError(
                        exc,
                        user_frames,
                        caller_context,
                        func_name,
                    )
                    clean_error.format_rich()
                    sys.exit(1)  # Exit after formatting

            # Fallback: re-raise original if we couldn't extract frames
            raise

    return wrapper  # type: ignore


def handle_exceptions_with_clean_traceback(func: F) -> F:
    """Alias for clean_traceback decorator."""
    return clean_traceback(func)