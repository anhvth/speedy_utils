"""
Common utilities shared across multi_process backends.

Includes:
- Error formatting and logging
- Log gating (stdout/stderr control)
- Cache helpers
- Process/thread tracking
"""
from __future__ import annotations

import contextlib
import linecache
import os
import pickle
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import psutil

if TYPE_CHECKING:
    from tqdm import tqdm

# ─── error handler types ────────────────────────────────────────
ErrorHandlerType = Literal['raise', 'ignore', 'log']


class ErrorStats:
    """Thread-safe error statistics tracker."""

    def __init__(
        self,
        func_name: str,
        max_error_files: int = 100,
        write_logs: bool = True
    ):
        self._lock = threading.Lock()
        self._success_count = 0
        self._error_count = 0
        self._first_error_shown = False
        self._max_error_files = max_error_files
        self._write_logs = write_logs
        self._error_log_dir = self._get_error_log_dir(func_name)
        if write_logs:
            self._error_log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _get_error_log_dir(func_name: str) -> Path:
        """Generate unique error log directory with run counter."""
        base_dir = Path('.cache/speedy_utils/error_logs')
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Find the next run counter
        counter = 1
        existing = list(base_dir.glob(f'{func_name}_run_*'))
        if existing:
            counters = []
            for p in existing:
                try:
                    parts = p.name.split('_run_')
                    if len(parts) == 2:
                        counters.append(int(parts[1]))
                except (ValueError, IndexError):
                    pass
            if counters:
                counter = max(counters) + 1
        
        return base_dir / f'{func_name}_run_{counter}'

    def record_success(self) -> None:
        with self._lock:
            self._success_count += 1

    def record_error(
        self,
        idx: int,
        error: Exception,
        input_value: Any,
        func_name: str,
        ray_task_error: Exception | None = None,
    ) -> str | None:
        """
        Record an error and write to log file.
        Returns the log file path if written, None otherwise.
        
        Args:
            ray_task_error: Optional RayTaskError for fallback frame extraction
                            when the native traceback is unavailable.
        """
        with self._lock:
            self._error_count += 1
            should_show_first = not self._first_error_shown
            if should_show_first:
                self._first_error_shown = True
            should_write = (
                self._write_logs and self._error_count <= self._max_error_files
            )

        log_path = None
        if should_write:
            log_path = self._write_error_log(
                idx, error, input_value, func_name, ray_task_error
            )

        if should_show_first:
            self._print_first_error(
                error, input_value, func_name, log_path, ray_task_error
            )

        return log_path

    def _write_error_log(
        self,
        idx: int,
        error: Exception,
        input_value: Any,
        func_name: str,
        ray_task_error: Exception | None = None,
    ) -> str:
        """Write error details to a log file."""
        from io import StringIO
        from rich.console import Console
        
        log_path = self._error_log_dir / f'{idx}.log'
        
        output = StringIO()
        console = Console(file=output, width=120, no_color=False)
        
        # Format traceback using unified extraction
        tb_lines = _format_traceback_lines(
            _extract_frames(error, ray_task_error),
            include_locals=False,
        )
        
        console.print(f'{"=" * 60}')
        console.print(f'Error at index: {idx}')
        console.print(f'Function: {func_name}')
        console.print(f'Error Type: {type(error).__name__}')
        console.print(f'Error Message: {error}')
        console.print(f'{"=" * 60}')
        console.print('')
        console.print('Input:')
        console.print('-' * 40)
        try:
            import json
            console.print(json.dumps(input_value, indent=2))
        except Exception:
            console.print(repr(input_value))
        console.print('')
        console.print('Traceback:')
        console.print('-' * 40)
        for line in tb_lines:
            console.print(line)
        
        with open(log_path, 'w') as f:
            f.write(output.getvalue())
        
        return str(log_path)

    def _print_first_error(
        self,
        error: Exception,
        input_value: Any,
        func_name: str,
        log_path: str | None,
        ray_task_error: Exception | None = None,
    ) -> None:
        """Print the first error to screen with Rich formatting."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console(stderr=True)
            
            # Use unified frame extraction
            tb_lines = _format_traceback_lines(
                _extract_frames(error, ray_task_error),
                include_locals=False,
            )
            
            console.print()
            console.print(
                Panel(
                    '\n'.join(tb_lines),
                    title=(
                        '[bold red]First Error '
                        '(continuing with remaining items)[/bold red]'
                    ),
                    border_style='yellow',
                    expand=False,
                )
            )
            console.print(
                f'[bold red]{type(error).__name__}[/bold red]: {error}'
            )
            if log_path:
                console.print(f'[dim]Error log: {log_path}[/dim]')
            console.print()
        except ImportError:
            # Fallback to plain print
            print('\n--- First Error ---', file=sys.stderr)
            print(f'{type(error).__name__}: {error}', file=sys.stderr)
            if log_path:
                print(f'Error log: {log_path}', file=sys.stderr)
            print('', file=sys.stderr)

    @property
    def success_count(self) -> int:
        with self._lock:
            return self._success_count

    @property
    def error_count(self) -> int:
        with self._lock:
            return self._error_count

    def get_postfix_dict(self) -> dict[str, int]:
        """Get dict for pbar postfix."""
        with self._lock:
            return {'ok': self._success_count, 'err': self._error_count}


# ─── Traceback formatting utilities ─────────────────────────────


def _should_skip_frame(filepath: str) -> bool:
    """Check if a frame should be filtered from traceback display."""
    skip_patterns = [
        'ray/_private',
        'ray/worker',
        'site-packages/ray',
        'speedy_utils/multi_worker',
        'concurrent/futures',
        'multiprocessing/',
        'fastcore/parallel',
        'fastcore/foundation',
        'fastcore/basics',
        'site-packages/fastcore',
        '/threading.py',
        '/concurrent/',
    ]
    return any(skip in filepath for skip in skip_patterns)


def _should_show_local(name: str, value: object) -> bool:
    """Check if a local variable should be displayed in traceback."""
    import types
    
    # Skip dunder variables
    if name.startswith('__') and name.endswith('__'):
        return False
    
    # Skip modules
    if isinstance(value, types.ModuleType):
        return False
    
    # Skip type objects and classes
    if isinstance(value, type):
        return False
    
    # Skip functions and methods
    if isinstance(
        value,
        (types.FunctionType, types.MethodType, types.BuiltinFunctionType)
    ):
        return False
    
    # Skip common typing aliases
    value_str = str(value)
    if value_str.startswith('typing.'):
        return False
    
    # Skip large objects that would clutter output
    skip_markers = ['module', 'function', 'method', 'built-in']
    if value_str.startswith('<') and any(x in value_str for x in skip_markers):
        return False
    
    return True


def _format_locals(frame_locals: dict) -> list[str]:
    """Format local variables for display, filtering out noisy imports."""
    from io import StringIO
    from rich.console import Console
    from rich.pretty import Pretty
    
    # Filter locals
    clean_locals = {
        k: v for k, v in frame_locals.items() if _should_show_local(k, v)
    }
    
    if not clean_locals:
        return []
    
    lines = []
    lines.append('[dim]╭─ locals ─╮[/dim]')
    
    # Format each local variable
    for name, value in clean_locals.items():
        # Use Rich's Pretty for nice formatting
        try:
            console = Console(file=StringIO(), width=60)
            console.print(Pretty(value), end='')
            value_str = console.file.getvalue().strip()
            # Limit length
            if len(value_str) > 100:
                value_str = value_str[:97] + '...'
        except Exception:
            value_str = repr(value)
            if len(value_str) > 100:
                value_str = value_str[:97] + '...'
        
        lines.append(f'[dim]│[/dim] {name} = {value_str}')
    
    lines.append('[dim]╰──────────╯[/dim]')
    return lines


def _format_frame_with_context(
    filepath: str,
    lineno: int,
    funcname: str,
    frame_locals: dict | None = None,
) -> list[str]:
    """Format a single frame with context lines and optional locals."""
    lines = []
    # Frame header
    lines.append(
        f'[cyan]{filepath}[/cyan]:[yellow]{lineno}[/yellow] '
        f'in [green]{funcname}[/green]'
    )
    lines.append('')
    
    # Get context lines
    context_size = 3
    start_line = max(1, lineno - context_size)
    end_line = lineno + context_size + 1
    
    for line_num in range(start_line, end_line):
        line_text = linecache.getline(filepath, line_num).rstrip()
        if line_text:
            num_str = str(line_num).rjust(4)
            if line_num == lineno:
                lines.append(f'[dim]{num_str}[/dim] [red]❱[/red] {line_text}')
            else:
                lines.append(f'[dim]{num_str} │[/dim] {line_text}')
    
    # Add locals if available
    if frame_locals:
        locals_lines = _format_locals(frame_locals)
        if locals_lines:
            lines.append('')
            lines.extend(locals_lines)
    
    lines.append('')
    return lines


def _format_traceback_lines(
    frames: list[tuple[str, int, str, dict]],
    *,
    caller_info: dict | None = None,
    include_locals: bool = True,
) -> list[str]:
    """Format frames into a list of Rich-markup lines for display/logging."""
    display_lines: list[str] = []
    if caller_info:
        display_lines.extend(
            _format_frame_with_context(
                caller_info['filename'],
                caller_info['lineno'],
                caller_info['function'],
                None,
            )
        )
    for filepath, lineno, funcname, frame_locals in frames:
        locals_for_frame = frame_locals if include_locals else None
        display_lines.extend(
            _format_frame_with_context(
                filepath,
                lineno,
                funcname,
                locals_for_frame,
            )
        )
    return display_lines


def _extract_frames_from_traceback(
    error: Exception,
) -> list[tuple[str, int, str, dict]]:
    """Extract user frames from exception traceback object with locals."""
    frames = []
    if hasattr(error, '__traceback__') and error.__traceback__ is not None:
        tb = error.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            filename = frame.f_code.co_filename
            lineno = tb.tb_lineno
            funcname = frame.f_code.co_name
            
            if not _should_skip_frame(filename):
                # Get local variables from the frame
                frame_locals = dict(frame.f_locals)
                frames.append((filename, lineno, funcname, frame_locals))
            
            tb = tb.tb_next
    return frames


def _extract_frames_from_ray_error(
    ray_task_error: Exception,
) -> list[tuple[str, int, str, dict]]:
    """Extract user frames from Ray's string traceback representation."""
    import re
    
    frames = []
    error_str = str(ray_task_error)
    lines = error_str.split('\n')
    
    for line in lines:
        # Match: File "path", line N, in func
        file_match = re.match(r'\s*File "([^"]+)", line (\d+), in (.+)', line)
        if file_match:
            filepath, lineno, funcname = file_match.groups()
            if not _should_skip_frame(filepath):
                # Ray doesn't preserve locals, so use empty dict
                frames.append((filepath, int(lineno), funcname, {}))
    
    return frames


def _extract_frames(
    error: Exception,
    ray_task_error: Exception | None = None,
) -> list[tuple[str, int, str, dict]]:
    """
    Unified frame extraction that works for both native exceptions and Ray errors.
    
    First tries to extract frames from the error's __traceback__.
    If that's empty and ray_task_error is provided, falls back to parsing
    the Ray error string representation.
    """
    # Try native traceback extraction first
    frames = _extract_frames_from_traceback(error)
    
    # If empty and we have a Ray error, try string parsing
    if not frames and ray_task_error is not None:
        frames = _extract_frames_from_ray_error(ray_task_error)
    
    return frames


def _display_formatted_error_and_exit(
    exc_type_name: str,
    exc_msg: str,
    frames: list[tuple[str, int, str, dict]],
    caller_info: dict | None,
    backend: str,
    pbar: tqdm | None = None,
) -> None:
    """Display a formatted error and exit the process."""
    # Suppress additional error logs
    os.environ['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'

    # Close progress bar cleanly if provided
    if pbar is not None:
        pbar.close()

    from rich.console import Console
    from rich.panel import Panel
    console = Console(stderr=True)

    if frames or caller_info:
        display_lines = _format_traceback_lines(
            frames,
            caller_info=caller_info,
            include_locals=True,
        )

        # Display the traceback
        console.print()
        console.print(
            Panel(
                '\n'.join(display_lines),
                title=(
                    f'[bold red]Traceback (most recent call last) '
                    f'[{backend}][/bold red]'
                ),
                border_style='red',
                expand=False,
            )
        )
        console.print(f'[bold red]{exc_type_name}[/bold red]: {exc_msg}')
        console.print()
    else:
        # No frames found, minimal output
        console.print()
        console.print(f'[bold red]{exc_type_name}[/bold red]: {exc_msg}')
        console.print()

    # Ensure output is flushed
    sys.stderr.flush()
    sys.stdout.flush()
    sys.exit(1)


def _exit_on_worker_error(
    error: Exception,
    pbar: tqdm | None = None,
    caller_info: dict | None = None,
    backend: str = 'unknown',
) -> None:
    """Display a clean traceback for a worker error and exit the process."""
    frames = _extract_frames_from_traceback(error)
    _display_formatted_error_and_exit(
        exc_type_name=type(error).__name__,
        exc_msg=str(error),
        frames=frames,
        caller_info=caller_info,
        backend=backend,
        pbar=pbar,
    )


def _exit_on_ray_error(
    ray_task_error: Exception,
    pbar: tqdm | None = None,
    caller_info: dict | None = None,
) -> None:
    """Display a clean traceback for a RayTaskError and exit the process."""
    # Get the exception info
    cause = (
        ray_task_error.cause
        if hasattr(ray_task_error, 'cause')
        else None
    )
    if cause is None:
        cause = ray_task_error.__cause__

    exc_type_name = type(cause).__name__ if cause else 'Error'
    exc_msg = str(cause) if cause else str(ray_task_error)
    
    frames = []
    if cause:
        frames = _extract_frames_from_traceback(cause)
    if not frames:
        frames = _extract_frames_from_ray_error(ray_task_error)
    _display_formatted_error_and_exit(
        exc_type_name=exc_type_name,
        exc_msg=exc_msg,
        frames=frames,
        caller_info=caller_info,
        backend='ray',
        pbar=pbar,
    )


# ─── Process/thread tracking ────────────────────────────────────

SPEEDY_RUNNING_PROCESSES: list[psutil.Process] = []
_SPEEDY_PROCESSES_LOCK = threading.Lock()


def _prune_dead_processes() -> None:
    """Remove dead processes from tracking list."""
    with _SPEEDY_PROCESSES_LOCK:
        SPEEDY_RUNNING_PROCESSES[:] = [
            p for p in SPEEDY_RUNNING_PROCESSES if p.is_running()
        ]


def _track_processes(processes: list[psutil.Process]) -> None:
    """Add processes to global tracking list."""
    if not processes:
        return
    with _SPEEDY_PROCESSES_LOCK:
        living = [p for p in SPEEDY_RUNNING_PROCESSES if p.is_running()]
        for candidate in processes:
            if not candidate.is_running():
                continue
            if any(existing.pid == candidate.pid for existing in living):
                continue
            living.append(candidate)
        SPEEDY_RUNNING_PROCESSES[:] = living


def _track_multiprocessing_processes() -> None:
    """Track multiprocessing worker processes."""
    try:
        # Find recently created child processes
        current_pid = os.getpid()
        parent = psutil.Process(current_pid)
        new_processes = []
        for child in parent.children(recursive=False):
            try:
                # Created within last 5 seconds
                if time.time() - child.create_time() < 5:
                    new_processes.append(child)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _track_processes(new_processes)
    except Exception:
        # Don't fail if process tracking fails
        pass


def _track_ray_processes() -> None:
    """Track Ray worker processes when Ray is initialized."""
    try:
        current_pid = os.getpid()
        parent = psutil.Process(current_pid)
        ray_processes = []
        for child in parent.children(recursive=True):
            try:
                name = child.name().lower()
                if 'ray' in name or 'worker' in name:
                    ray_processes.append(child)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _track_processes(ray_processes)
    except Exception:
        pass


# ─── Log gating utilities ───────────────────────────────────────

_LOG_GATE_CACHE: dict[str, bool] = {}


def _should_allow_worker_logs(
    mode: Literal['all', 'zero', 'first'],
    gate_path: Path | None,
) -> bool:
    """Determine if current worker should emit logs for the given mode."""
    if mode == 'all':
        return True
    if mode == 'zero':
        return False
    if mode == 'first':
        if gate_path is None:
            return True
        key = str(gate_path)
        cached = _LOG_GATE_CACHE.get(key)
        if cached is not None:
            return cached
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(key, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            allowed = False
        else:
            os.close(fd)
            allowed = True
        _LOG_GATE_CACHE[key] = allowed
        return allowed
    raise ValueError(f'Unsupported log mode: {mode!r}')


def _cleanup_log_gate(gate_path: Path | None) -> None:
    """Remove the log gate file if it exists."""
    if gate_path is None:
        return
    try:
        gate_path.unlink(missing_ok=True)
    except OSError:
        pass


class _PrefixedWriter:
    """Stream wrapper that prefixes each line with worker id."""

    def __init__(self, stream, prefix: str):
        self._stream = stream
        self._prefix = prefix
        self._at_line_start = True

    def write(self, s):
        if not s:
            return 0
        total = 0
        for chunk in s.splitlines(True):
            if self._at_line_start:
                self._stream.write(self._prefix)
                total += len(self._prefix)
            self._stream.write(chunk)
            total += len(chunk)
            self._at_line_start = chunk.endswith('\n')
        return total

    def flush(self):
        self._stream.flush()


def _call_with_log_control(
    func: Callable,
    x: Any,
    func_kwargs: dict[str, Any],
    log_mode: Literal['all', 'zero', 'first'],
    gate_path: Path | None,
):
    """Call a function, silencing stdout/stderr based on log mode."""
    allow_logs = _should_allow_worker_logs(log_mode, gate_path)
    if allow_logs:
        prefix = f'[worker-{os.getpid()}] '
        # Route worker logs to stderr to reduce clobbering tqdm on stdout
        out = _PrefixedWriter(sys.stderr, prefix)
        err = out
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            return func(x, **func_kwargs)
    with (
        open(os.devnull, 'w') as devnull,
        contextlib.redirect_stdout(devnull),
        contextlib.redirect_stderr(devnull),
    ):
        return func(x, **func_kwargs)


# ─── Cache helpers ──────────────────────────────────────────────


def _build_cache_dir(func: Callable, items: list[Any]) -> Path:
    """Build cache dir with function name + timestamp."""
    import datetime
    func_name = getattr(func, '__name__', 'func')
    now = datetime.datetime.now()
    stamp = now.strftime('%m%d_%Hh%Mm%Ss')
    run_id = f'{func_name}_{stamp}_{uuid.uuid4().hex[:6]}'
    path = Path('.cache') / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def wrap_dump(
    func: Callable,
    cache_dir: Path | None,
    dump_in_thread: bool = True,
):
    """Wrap a function so results are dumped to .pkl when cache_dir is set."""
    if cache_dir is None:
        return func

    def wrapped(x, *args, **kwargs):
        res = func(x, *args, **kwargs)
        p = cache_dir / f'{uuid.uuid4().hex}.pkl'

        def save():
            with open(p, 'wb') as fh:
                pickle.dump(res, fh)

        if dump_in_thread:
            thread = threading.Thread(target=save)
            while threading.active_count() > 16:
                time.sleep(0.1)
            thread.start()
        else:
            save()
        return str(p)

    return wrapped


# ─── Log gate path helper ───────────────────────────────────────


def create_log_gate_path(
    log_worker: Literal['zero', 'first', 'all'],
) -> Path | None:
    """Create a log gate path for first-worker-only logging."""
    if log_worker == 'first':
        return (
            Path(tempfile.gettempdir())
            / f'speedy_utils_log_gate_{os.getpid()}_{uuid.uuid4().hex}.gate'
        )
    elif log_worker not in ('zero', 'all'):
        raise ValueError(f'Unsupported log_worker: {log_worker!r}')
    return None


# ─── Cleanup utility ────────────────────────────────────────────


def cleanup_phantom_workers() -> None:
    """
    Kill all tracked processes and threads (phantom workers).

    Also lists non-daemon threads that remain.
    """
    # Clean up tracked processes first
    _prune_dead_processes()
    killed_processes = 0
    with _SPEEDY_PROCESSES_LOCK:
        for process in SPEEDY_RUNNING_PROCESSES[:]:
            try:
                print(
                    f'🔪 Killing tracked process {process.pid} '
                    f'({process.name()})'
                )
                process.kill()
                killed_processes += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                print(f'⚠️ Could not kill process {process.pid}: {e}')
        SPEEDY_RUNNING_PROCESSES.clear()

    # Also kill any remaining child processes (fallback)
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        try:
            print(f'🔪 Killing child process {child.pid} ({child.name()})')
            child.kill()
        except psutil.NoSuchProcess:
            pass

    # Try to clean up threads using thread module functions if available
    try:
        from .thread import (
            SPEEDY_RUNNING_THREADS,
            _prune_dead_threads,
            kill_all_thread,
        )

        _prune_dead_threads()
        killed_threads = kill_all_thread()
        if killed_threads > 0:
            print(f'🔪 Killed {killed_threads} tracked threads')
    except ImportError:
        # Fallback: just report stray threads
        for t in threading.enumerate():
            if t is threading.current_thread():
                continue
            if not t.daemon:
                print(
                    f'⚠️ Thread {t.name} is still running '
                    f'(cannot be force-killed).'
                )

    print(
        f'✅ Cleaned up {killed_processes} tracked processes and '
        f'child processes (kernel untouched).'
    )


__all__ = [
    # Types
    'ErrorHandlerType',
    'ErrorStats',
    # Process tracking globals
    'SPEEDY_RUNNING_PROCESSES',
    '_SPEEDY_PROCESSES_LOCK',
    # Error utilities
    '_should_skip_frame',
    '_format_traceback_lines',
    '_extract_frames_from_traceback',
    '_extract_frames_from_ray_error',
    '_display_formatted_error_and_exit',
    '_exit_on_worker_error',
    '_exit_on_ray_error',
    # Process tracking
    '_prune_dead_processes',
    '_track_processes',
    '_track_multiprocessing_processes',
    '_track_ray_processes',
    # Log gating
    '_LOG_GATE_CACHE',
    '_should_allow_worker_logs',
    '_cleanup_log_gate',
    '_PrefixedWriter',
    '_call_with_log_control',
    'create_log_gate_path',
    # Cache helpers
    '_build_cache_dir',
    'wrap_dump',
    # Cleanup
    'cleanup_phantom_workers',
]
