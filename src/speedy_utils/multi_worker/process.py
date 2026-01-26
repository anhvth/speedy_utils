import warnings
import os
# Suppress Ray FutureWarnings before any imports
warnings.filterwarnings("ignore", category=FutureWarning, module="ray.*")
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)

# Set environment variables before Ray is imported anywhere
os.environ["RAY_ACCEL_ENV_VAR_OVERRI" \
"DE_ON_ZERO"] = "0"
os.environ["RAY_DEDUP_LOGS"] = "1"
os.environ["RAY_LOG_TO_STDERR"] = "0"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"

from ..__imports import *
import tempfile
import inspect
import linecache
import traceback as tb_module
from .progress import create_progress_tracker, ProgressPoller, get_ray_progress_actor

# Import thread tracking functions if available
try:
    from .thread import _prune_dead_threads, _track_executor_threads
except ImportError:
    _prune_dead_threads = None  # type: ignore[assignment]
    _track_executor_threads = None  # type: ignore[assignment]


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
    ) -> str | None:
        """
        Record an error and write to log file.
        Returns the log file path if written, None otherwise.
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
                idx, error, input_value, func_name
            )

        if should_show_first:
            self._print_first_error(error, input_value, func_name, log_path)

        return log_path

    def _write_error_log(
        self,
        idx: int,
        error: Exception,
        input_value: Any,
        func_name: str,
    ) -> str:
        """Write error details to a log file."""
        log_path = self._error_log_dir / f'{idx}.log'
        
        # Format traceback
        tb_lines = self._format_traceback(error)
        
        content = []
        content.append(f'{"=" * 60}')
        content.append(f'Error at index: {idx}')
        content.append(f'Function: {func_name}')
        content.append(f'Error Type: {type(error).__name__}')
        content.append(f'Error Message: {error}')
        content.append(f'{"=" * 60}')
        content.append('')
        content.append('Input:')
        content.append('-' * 40)
        try:
            content.append(repr(input_value))
        except Exception:
            content.append('<unable to repr input>')
        content.append('')
        content.append('Traceback:')
        content.append('-' * 40)
        content.extend(tb_lines)
        
        with open(log_path, 'w') as f:
            f.write('\n'.join(content))
        
        return str(log_path)

    def _format_traceback(self, error: Exception) -> list[str]:
        """Format traceback with context lines like Rich panel."""
        lines = []
        frames = _extract_frames_from_traceback(error)
        
        for filepath, lineno, funcname, frame_locals in frames:
            lines.append(f'│ {filepath}:{lineno} in {funcname} │')
            lines.append('│' + ' ' * 70 + '│')
            
            # Get context lines
            context_size = 3
            start_line = max(1, lineno - context_size)
            end_line = lineno + context_size + 1
            
            for line_num in range(start_line, end_line):
                line_text = linecache.getline(filepath, line_num).rstrip()
                if line_text:
                    num_str = str(line_num).rjust(4)
                    if line_num == lineno:
                        lines.append(f'│   {num_str} ❱     {line_text}')
                    else:
                        lines.append(f'│   {num_str} │     {line_text}')
            lines.append('')
        
        return lines

    def _print_first_error(
        self,
        error: Exception,
        input_value: Any,
        func_name: str,
        log_path: str | None,
    ) -> None:
        """Print the first error to screen with Rich formatting."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            console = Console(stderr=True)
            
            tb_lines = self._format_traceback(error)
            
            console.print()
            console.print(
                Panel(
                    '\n'.join(tb_lines),
                    title='[bold red]First Error (continuing with remaining items)[/bold red]',
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
            print(f'\n--- First Error ---', file=sys.stderr)
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
    if isinstance(value, (types.FunctionType, types.MethodType, types.BuiltinFunctionType)):
        return False
    
    # Skip common typing aliases
    value_str = str(value)
    if value_str.startswith('typing.'):
        return False
    
    # Skip large objects that would clutter output
    if value_str.startswith('<') and any(x in value_str for x in ['module', 'function', 'method', 'built-in']):
        return False
    
    return True


def _format_locals(frame_locals: dict) -> list[str]:
    """Format local variables for display, filtering out noisy imports."""
    from rich.pretty import Pretty
    from rich.console import Console
    from io import StringIO
    
    # Filter locals
    clean_locals = {k: v for k, v in frame_locals.items() if _should_show_local(k, v)}
    
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


def _format_frame_with_context(filepath: str, lineno: int, funcname: str, frame_locals: dict | None = None) -> list[str]:
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
        import linecache
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


def _display_formatted_error(
    exc_type_name: str,
    exc_msg: str,
    frames: list[tuple[str, int, str, dict]],
    caller_info: dict | None,
    backend: str,
    pbar=None,
) -> None:
    
    # Suppress additional error logs
    os.environ['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'

    # Close progress bar cleanly if provided
    if pbar is not None:
        pbar.close()

    from rich.console import Console
    from rich.panel import Panel
    console = Console(stderr=True)

    if frames or caller_info:
        display_lines = []
        
        # Add caller frame first if available (no locals for caller)
        if caller_info:
            display_lines.extend(_format_frame_with_context(
                caller_info['filename'],
                caller_info['lineno'],
                caller_info['function'],
                None  # Don't show locals for caller frame
            ))
        
        # Add error frames with locals
        for filepath, lineno, funcname, frame_locals in frames:
            display_lines.extend(_format_frame_with_context(
                filepath, lineno, funcname, frame_locals
            ))

        # Display the traceback
        console.print()
        console.print(
            Panel(
                '\n'.join(display_lines),
                title=f'[bold red]Traceback (most recent call last) [{backend}][/bold red]',
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


def _extract_frames_from_traceback(error: Exception) -> list[tuple[str, int, str, dict]]:
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


def _extract_frames_from_ray_error(ray_task_error: Exception) -> list[tuple[str, int, str, dict]]:
    """Extract user frames from Ray's string traceback representation."""
    frames = []
    error_str = str(ray_task_error)
    lines = error_str.split('\n')
    
    import re
    for i, line in enumerate(lines):
        # Match: File "path", line N, in func
        file_match = re.match(r'\s*File "([^"]+)", line (\d+), in (.+)', line)
        if file_match:
            filepath, lineno, funcname = file_match.groups()
            if not _should_skip_frame(filepath):
                # Ray doesn't preserve locals, so use empty dict
                frames.append((filepath, int(lineno), funcname, {}))
    
    return frames


def _reraise_worker_error(error: Exception, pbar=None, caller_info=None, backend: str = 'unknown') -> None:
    """
    Re-raise the original exception from a worker error with clean traceback.
    Works for multiprocessing, threadpool, and other backends with real tracebacks.
    """
    frames = _extract_frames_from_traceback(error)
    _display_formatted_error(
        exc_type_name=type(error).__name__,
        exc_msg=str(error),
        frames=frames,
        caller_info=caller_info,
        backend=backend,
        pbar=pbar,
    )


def _reraise_ray_error(ray_task_error: Exception, pbar=None, caller_info=None) -> None:
    """
    Re-raise the original exception from a RayTaskError with clean traceback.
    Parses Ray's string traceback and displays with full context.
    """
    # Get the exception info
    cause = ray_task_error.cause if hasattr(ray_task_error, 'cause') else None
    if cause is None:
        cause = ray_task_error.__cause__

    exc_type_name = type(cause).__name__ if cause else 'Error'
    exc_msg = str(cause) if cause else str(ray_task_error)
    
    frames = _extract_frames_from_ray_error(ray_task_error)
    _display_formatted_error(
        exc_type_name=exc_type_name,
        exc_msg=exc_msg,
        frames=frames,
        caller_info=caller_info,
        backend='ray',
        pbar=pbar,
    )


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


def _track_ray_processes() -> None:
    """Track Ray worker processes when Ray is initialized."""

    try:
        # Get Ray worker processes
        current_pid = os.getpid()
        parent = psutil.Process(current_pid)
        ray_processes = []
        for child in parent.children(recursive=True):
            try:
                if 'ray' in child.name().lower() or 'worker' in child.name().lower():
                    ray_processes.append(child)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _track_processes(ray_processes)
    except Exception:
        # Don't fail if process tracking fails
        pass


def _track_multiprocessing_processes() -> None:
    """Track multiprocessing worker processes."""
    try:
        # Find recently created child processes that might be multiprocessing workers
        current_pid = os.getpid()
        parent = psutil.Process(current_pid)
        new_processes = []
        for child in parent.children(recursive=False):  # Only direct children
            try:
                # Basic heuristic: if it's a recent child process, it might be a worker
                if (
                    time.time() - child.create_time() < 5
                ):  # Created within last 5 seconds
                    new_processes.append(child)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _track_processes(new_processes)
    except Exception:
        # Don't fail if process tracking fails
        pass


# ─── cache helpers ──────────────────────────────────────────


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
_DUMP_INTERMEDIATE_THREADS = []
def wrap_dump(func: Callable, cache_dir: Path | None, dump_in_thread: bool = True):
    """Wrap a function so results are dumped to .pkl when cache_dir is set."""
    if cache_dir is None:
        return func

    def wrapped(x, *args, **kwargs):
        res = func(x, *args, **kwargs)
        p = cache_dir / f'{uuid.uuid4().hex}.pkl'

        def save():
            with open(p, 'wb') as fh:
                pickle.dump(res, fh)
            # Clean trash to avoid bloating memory
            # print(f'Thread count: {threading.active_count()}')
            # print(f'Saved result to {p}')

        if dump_in_thread:
            thread = threading.Thread(target=save)
            _DUMP_INTERMEDIATE_THREADS.append(thread)
            # count thread
            # print(f'Thread count: {threading.active_count()}')
            while threading.active_count() > 16:
                time.sleep(0.1)
            thread.start()
        else:
            save()
        return str(p)

    return wrapped


_LOG_GATE_CACHE: dict[str, bool] = {}


def _should_allow_worker_logs(mode: Literal['all', 'zero', 'first'], gate_path: Path | None) -> bool:
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


def _cleanup_log_gate(gate_path: Path | None):
    if gate_path is None:
        return
    try:
        gate_path.unlink(missing_ok=True)
    except OSError:
        pass


@contextlib.contextmanager
def _patch_fastcore_progress_bar(*, leave: bool = True):
    """Temporarily force fastcore.progress_bar to keep the bar on screen."""
    try:
        import fastcore.parallel as _fp
    except ImportError:
        yield False
        return

    orig = getattr(_fp, 'progress_bar', None)
    if orig is None:
        yield False
        return

    def _wrapped(*args, **kwargs):
        kwargs.setdefault('leave', leave)
        return orig(*args, **kwargs)

    _fp.progress_bar = _wrapped
    try:
        yield True
    finally:
        _fp.progress_bar = orig


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
        prefix = f"[worker-{os.getpid()}] "
        # Route worker logs to stderr to reduce clobbering tqdm/progress output on stdout
        out = _PrefixedWriter(sys.stderr, prefix)
        err = out
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            return func(x, **func_kwargs)
    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
        return func(x, **func_kwargs)


# ─── ray management ─────────────────────────────────────────

RAY_WORKER = None


def ensure_ray(workers: int | None, pbar: tqdm | None = None, ray_metrics_port: int | None = None):
    """
    Initialize or reinitialize Ray safely for both local and cluster environments.

    1. Tries to connect to an existing cluster (address='auto') first.
    2. If no cluster is found, starts a local Ray instance with 'workers' CPUs.
    """
    import ray as _ray_module
    import logging

    global RAY_WORKER
    requested_workers = workers
    if workers is None:
        workers = os.cpu_count() or 1

    if ray_metrics_port is not None:
        os.environ['RAY_metrics_export_port'] = str(ray_metrics_port)

    allow_restart = os.environ.get("RESTART_RAY", "0").lower() in ("1", "true")
    is_cluster_env = "RAY_ADDRESS" in os.environ or os.environ.get("RAY_CLUSTER") == "1"

    # 1. Handle existing session
    if _ray_module.is_initialized():
        if not allow_restart:
            if pbar:
                pbar.set_postfix_str("Using existing Ray session")
            return

        # Avoid shutting down shared cluster sessions.
        if is_cluster_env:
            if pbar:
                pbar.set_postfix_str("Cluster active: skipping restart to protect connection")
            return

        # Local restart: only if worker count changed
        if workers != RAY_WORKER:
            if pbar:
                pbar.set_postfix_str(f'Restarting local Ray with {workers} workers')
            _ray_module.shutdown()
        else:
            return

    # 2. Initialization logic
    t0 = time.time()
    
    # Try to connect to existing cluster FIRST (address="auto")
    try:
        if pbar:
            pbar.set_postfix_str("Searching for Ray cluster...")
        
        # MUST NOT pass num_cpus/num_gpus here to avoid ValueError on existing clusters
        _ray_module.init(
            address="auto", 
            ignore_reinit_error=True, 
            logging_level=logging.ERROR,
            log_to_driver=False
        )
        
        if pbar:
            resources = _ray_module.cluster_resources()
            cpus = resources.get("CPU", 0)
            pbar.set_postfix_str(f"Connected to Ray Cluster ({int(cpus)} CPUs)")
            
    except Exception:
        # 3. Fallback: Start a local Ray session
        if pbar:
            pbar.set_postfix_str(f"No cluster found. Starting local Ray ({workers} CPUs)...")
            
        _ray_module.init(
            num_cpus=workers,
            ignore_reinit_error=True,
            logging_level=logging.ERROR,
            log_to_driver=False,
        )
        
        if pbar:
            took = time.time() - t0
            pbar.set_postfix_str(f'ray.init local {workers} took {took:.2f}s')

    _track_ray_processes()

    if requested_workers is None:
        try:
            resources = _ray_module.cluster_resources()
            total_cpus = int(resources.get("CPU", 0))
            if total_cpus > 0:
                workers = total_cpus
        except Exception:
            pass

    RAY_WORKER = workers


# TODO: make smarter backend selection, when shared_kwargs is used, and backend != 'ray', do not raise error but change to ray and warning user about this
def multi_process(
    func: Callable[[Any], Any],
    items: Iterable[Any] | None = None,
    *,
    inputs: Iterable[Any] | None = None,
    workers: int | None = None,
    lazy_output: bool = False,
    progress: bool = True,
    backend: Literal['seq', 'ray', 'mp', 'safe'] = 'mp',
    desc: str | None = None,
    shared_kwargs: list[str] | None = None,
    dump_in_thread: bool = True,
    ray_metrics_port: int | None = None,
    log_worker: Literal['zero', 'first', 'all'] = 'first',
    total_items: int | None = None,
    poll_interval: float = 0.3,
    error_handler: ErrorHandlerType = 'raise',
    max_error_files: int = 100,
    **func_kwargs: Any,
) -> list[Any]:
    """
    Multi-process map with selectable backend.

    backend:
        - "seq": run sequentially
        - "ray": run in parallel with Ray
        - "mp": run in parallel with thread pool (uses ThreadPoolExecutor)
        - "safe": run in parallel with thread pool (explicitly safe for tests)

    shared_kwargs:
        - Optional list of kwarg names that should be shared via Ray's
          zero-copy object store
        - Only works with Ray backend
        - Useful for large objects (e.g., models, datasets)
        - Example: shared_kwargs=['model', 'tokenizer']

    dump_in_thread:
        - Whether to dump results to disk in a separate thread (default: True)
        - If False, dumping is done synchronously

    ray_metrics_port:
        - Optional port for Ray metrics export (Ray backend only)

    log_worker:
        - Control worker stdout/stderr noise
        - 'first': only first worker emits logs (default)
        - 'all': allow worker prints
        - 'zero': silence all worker output

    total_items:
        - Optional item-level total for progress tracking (Ray backend only)

    poll_interval:
        - Poll interval in seconds for progress actor updates (Ray only)

    error_handler:
        - 'raise': raise exception on first error (default)
        - 'ignore': continue processing, return None for failed items
        - 'log': same as ignore, but logs errors to files

    max_error_files:
        - Maximum number of error log files to write (default: 100)
        - Error logs are written to .cache/speedy_utils/error_logs/{idx}.log
        - First error is always printed to screen with the log file path

    If lazy_output=True, every result is saved to .pkl and
    the returned list contains file paths.
    """

    # default backend selection
    if backend is None:
        try:
            import ray as _ray_module
            backend = 'ray'
        except ImportError:
            backend = 'mp'

    # Validate shared_kwargs
    if shared_kwargs:
        # Validate that all shared_kwargs are valid kwargs for the function
        sig = inspect.signature(func)
        valid_params = set(sig.parameters.keys())

        for kw in shared_kwargs:
            if kw not in func_kwargs:
                raise ValueError(
                    f"shared_kwargs key '{kw}' not found in provided func_kwargs"
                )
            # Check if parameter exists in function signature or if function accepts **kwargs
            has_var_keyword = any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )
            if kw not in valid_params and not has_var_keyword:
                raise ValueError(
                    f"shared_kwargs key '{kw}' is not a valid parameter for function '{func.__name__}'. "
                    f"Valid parameters: {valid_params}"
                )

    # Prefer Ray backend when shared kwargs are requested
    if shared_kwargs and backend != 'ray':
        warnings.warn(
            "shared_kwargs only supported with 'ray' backend, switching backend to 'ray'",
            UserWarning,
        )
        backend = 'ray'

    # unify items
    # unify items and coerce to concrete list so we can use len() and
    # iterate multiple times. This accepts ranges and other iterables.
    if items is None and inputs is not None:
        items = list(inputs)
    if items is not None and not isinstance(items, list):
        items = list(items)
    if items is None:
        raise ValueError("'items' or 'inputs' must be provided")

    if workers is None and backend != 'ray':
        workers = os.cpu_count() or 1

    # build cache dir + wrap func
    cache_dir = _build_cache_dir(func, items) if lazy_output else None
    f_wrapped = wrap_dump(func, cache_dir, dump_in_thread)

    log_gate_path: Path | None = None
    if log_worker == 'first':
        log_gate_path = Path(tempfile.gettempdir()) / f'speedy_utils_log_gate_{os.getpid()}_{uuid.uuid4().hex}.gate'
    elif log_worker not in ('zero', 'all'):
        raise ValueError(f'Unsupported log_worker: {log_worker!r}')

    total = len(items)
    if desc:
        desc = desc.strip() + f'[{backend}]'
    else:
        desc = f'Multi-process [{backend}]'

    # Initialize error stats for error handling
    func_name = getattr(func, '__name__', repr(func))
    error_stats = ErrorStats(
        func_name=func_name,
        max_error_files=max_error_files,
        write_logs=error_handler == 'log'
    )

    def _update_pbar_postfix(pbar: tqdm) -> None:
        """Update pbar with success/error counts."""
        postfix = error_stats.get_postfix_dict()
        pbar.set_postfix(postfix)

    def _wrap_with_error_handler(
        f: Callable,
        idx: int,
        input_value: Any,
        error_stats_ref: ErrorStats,
        handler: ErrorHandlerType,
    ) -> Callable:
        """Wrap function to handle errors based on error_handler setting."""
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            try:
                result = f(*args, **kwargs)
                error_stats_ref.record_success()
                return result
            except Exception as e:
                if handler == 'raise':
                    raise
                error_stats_ref.record_error(idx, e, input_value, func_name)
                return None
        return wrapper

    # ---- sequential backend ----
    if backend == 'seq':
        results: list[Any] = []
        with tqdm(total=total, desc=desc, disable=not progress, file=sys.stdout) as pbar:
            for idx, x in enumerate(items):
                try:
                    result = _call_with_log_control(
                        f_wrapped,
                        x,
                        func_kwargs,
                        log_worker,
                        log_gate_path,
                    )
                    error_stats.record_success()
                    results.append(result)
                except Exception as e:
                    if error_handler == 'raise':
                        raise
                    error_stats.record_error(idx, e, x, func_name)
                    results.append(None)
                pbar.update(1)
                _update_pbar_postfix(pbar)
        _cleanup_log_gate(log_gate_path)
        return results

    # ---- ray backend ----
    if backend == 'ray':
        import ray as _ray_module

        # Capture caller frame for better error reporting
        caller_frame = inspect.currentframe()
        caller_info = None
        if caller_frame and caller_frame.f_back:
            caller_info = {
                'filename': caller_frame.f_back.f_code.co_filename,
                'lineno': caller_frame.f_back.f_lineno,
                'function': caller_frame.f_back.f_code.co_name,
            }

        results = []
        gate_path_str = str(log_gate_path) if log_gate_path else None
        with tqdm(total=total, desc=desc, disable=not progress, file=sys.stdout) as pbar:
            ensure_ray(workers, pbar, ray_metrics_port)
            shared_refs = {}
            regular_kwargs = {}
            
            # Create progress actor for item-level tracking if total_items specified
            progress_actor = None
            progress_poller = None
            if total_items is not None:
                progress_actor = create_progress_tracker(total_items, desc or "Items")
                shared_refs['progress_actor'] = progress_actor

            if shared_kwargs:
                for kw in shared_kwargs:
                    # Put large objects in Ray's object store (zero-copy)
                    shared_refs[kw] = _ray_module.put(func_kwargs[kw])
                    pbar.set_postfix_str(f'ray: shared `{kw}` via object store')

                # Remaining kwargs are regular
                regular_kwargs = {
                    k: v for k, v in func_kwargs.items()
                    if k not in shared_kwargs
                }
            else:
                regular_kwargs = func_kwargs

            @_ray_module.remote
            def _task(x, shared_refs_dict, regular_kwargs_dict):
                # Dereference shared objects (zero-copy for numpy arrays)
                import ray as _ray_in_task
                gate = Path(gate_path_str) if gate_path_str else None
                dereferenced = {}
                for k, v in shared_refs_dict.items():
                    if k == 'progress_actor':
                        dereferenced[k] = v
                    else:
                        dereferenced[k] = _ray_in_task.get(v)
                all_kwargs = {**dereferenced, **regular_kwargs_dict}
                return _call_with_log_control(
                    f_wrapped,
                    x,
                    all_kwargs,
                    log_worker,
                    gate,
                )

            refs = [
                _task.remote(x, shared_refs, regular_kwargs) for x in items
            ]

            t_start = time.time()
            
            if progress_actor is not None:
                pbar.total = total_items
                pbar.refresh()
                progress_poller = ProgressPoller(progress_actor, pbar, poll_interval)
                progress_poller.start()
            
            for idx, r in enumerate(refs):
                try:
                    result = _ray_module.get(r)
                    error_stats.record_success()
                    results.append(result)
                except _ray_module.exceptions.RayTaskError as e:
                    if error_handler == 'raise':
                        if progress_poller is not None:
                            progress_poller.stop()
                        _reraise_ray_error(e, pbar, caller_info)
                    # Extract original error from RayTaskError
                    cause = e.cause if hasattr(e, 'cause') else e.__cause__
                    original_error = cause if cause else e
                    error_stats.record_error(idx, original_error, items[idx], func_name)
                    results.append(None)
                
                if progress_actor is None:
                    pbar.update(1)
                _update_pbar_postfix(pbar)
            
            if progress_poller is not None:
                progress_poller.stop()
                
            t_end = time.time()
            item_desc = f"{total_items:,} items" if total_items else f"{total} tasks"
            print(f"Ray processing took {t_end - t_start:.2f}s for {item_desc}")
        _cleanup_log_gate(log_gate_path)
        return results

    # ---- fastcore/thread backend (mp) ----
    if backend == 'mp':
        import concurrent.futures
        
        # Capture caller frame for better error reporting
        caller_frame = inspect.currentframe()
        caller_info = None
        if caller_frame and caller_frame.f_back:
            caller_info = {
                'filename': caller_frame.f_back.f_code.co_filename,
                'lineno': caller_frame.f_back.f_lineno,
                'function': caller_frame.f_back.f_code.co_name,
            }
        
        def worker_func(x):
            return _call_with_log_control(
                f_wrapped,
                x,
                func_kwargs,
                log_worker,
                log_gate_path,
            )
        
        results: list[Any] = [None] * total
        with tqdm(total=total, desc=desc, disable=not progress, file=sys.stdout) as pbar:
            try:
                from .thread import _prune_dead_threads, _track_executor_threads
                has_thread_tracking = True
            except ImportError:
                has_thread_tracking = False
            
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                if has_thread_tracking:
                    _track_executor_threads(executor)
                
                # Submit all tasks
                future_to_idx = {
                    executor.submit(worker_func, x): idx
                    for idx, x in enumerate(items)
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        error_stats.record_success()
                        results[idx] = result
                    except Exception as e:
                        if error_handler == 'raise':
                            # Cancel remaining futures
                            for f in future_to_idx:
                                f.cancel()
                            _reraise_worker_error(e, pbar, caller_info, backend='mp')
                        error_stats.record_error(idx, e, items[idx], func_name)
                        results[idx] = None
                    pbar.update(1)
                    _update_pbar_postfix(pbar)
            
            if _prune_dead_threads is not None:
                _prune_dead_threads()
            
        _track_multiprocessing_processes()
        _prune_dead_processes()
        _cleanup_log_gate(log_gate_path)
        return results

    if backend == 'safe':
        # Completely safe backend for tests - no multiprocessing
        import concurrent.futures
        
        # Capture caller frame for better error reporting
        caller_frame = inspect.currentframe()
        caller_info = None
        if caller_frame and caller_frame.f_back:
            caller_info = {
                'filename': caller_frame.f_back.f_code.co_filename,
                'lineno': caller_frame.f_back.f_lineno,
                'function': caller_frame.f_back.f_code.co_name,
            }

        def worker_func(x):
            return _call_with_log_control(
                f_wrapped,
                x,
                func_kwargs,
                log_worker,
                log_gate_path,
            )
        
        results: list[Any] = [None] * total
        with tqdm(total=total, desc=desc, disable=not progress, file=sys.stdout) as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                if _track_executor_threads is not None:
                    _track_executor_threads(executor)
                
                # Submit all tasks
                future_to_idx = {
                    executor.submit(worker_func, x): idx
                    for idx, x in enumerate(items)
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        error_stats.record_success()
                        results[idx] = result
                    except Exception as e:
                        if error_handler == 'raise':
                            for f in future_to_idx:
                                f.cancel()
                            _reraise_worker_error(e, pbar, caller_info, backend='safe')
                        error_stats.record_error(idx, e, items[idx], func_name)
                        results[idx] = None
                    pbar.update(1)
                    _update_pbar_postfix(pbar)
            
            if _prune_dead_threads is not None:
                _prune_dead_threads()
        
        _cleanup_log_gate(log_gate_path)
        return results

    raise ValueError(f'Unsupported backend: {backend!r}')


def cleanup_phantom_workers():
    """
    Kill all tracked processes and threads (phantom workers) without killing the Jupyter kernel itself.
    Also lists non-daemon threads that remain.
    """
    # Clean up tracked processes first
    _prune_dead_processes()
    killed_processes = 0
    with _SPEEDY_PROCESSES_LOCK:
        for process in SPEEDY_RUNNING_PROCESSES[
            :
        ]:  # Copy to avoid modification during iteration
            try:
                print(f'🔪 Killing tracked process {process.pid} ({process.name()})')
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
        from .thread import SPEEDY_RUNNING_THREADS, _prune_dead_threads, kill_all_thread

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
                print(f'⚠️ Thread {t.name} is still running (cannot be force-killed).')

    print(
        f'✅ Cleaned up {killed_processes} tracked processes and child processes (kernel untouched).'
    )


# Usage: run this anytime after cancelling a cell


__all__ = [
    'SPEEDY_RUNNING_PROCESSES',
    'ErrorStats',
    'ErrorHandlerType',
    'multi_process',
    'cleanup_phantom_workers',
    'create_progress_tracker',
    'get_ray_progress_actor',
]
