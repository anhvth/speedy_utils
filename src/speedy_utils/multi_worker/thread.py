from ..__imports import *
import linecache

from .process import ErrorStats, ErrorHandlerType


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.text import Text
except ImportError:  # pragma: no cover
    Console = None  # type: ignore[assignment, misc]
    Panel = None  # type: ignore[assignment, misc]
    Syntax = None  # type: ignore[assignment, misc]
    Text = None  # type: ignore[assignment, misc]

# Sensible defaults
DEFAULT_WORKERS = (os.cpu_count() or 4) * 2

T = TypeVar('T')
R = TypeVar('R')

SPEEDY_RUNNING_THREADS: list[threading.Thread] = []  # cooperative shutdown tracking
_SPEEDY_THREADS_LOCK = threading.Lock()


class UserFunctionError(Exception):
    """Exception wrapper that highlights user function errors."""

    def __init__(
        self,
        original_exception: Exception,
        func_name: str,
        input_value: Any,
        user_traceback: list[traceback.FrameSummary],
        caller_frame: traceback.FrameSummary | None = None,
    ) -> None:
        self.original_exception = original_exception
        self.func_name = func_name
        self.input_value = input_value
        self.user_traceback = user_traceback
        self.caller_frame = caller_frame

        # Create a focused error message
        tb_str = ''.join(traceback.format_list(user_traceback))
        msg = (
            f'\nError in function "{func_name}" with input: {input_value!r}\n'
            f'\nUser code traceback:\n{tb_str}'
            f'{type(original_exception).__name__}: {original_exception}'
        )
        super().__init__(msg)

    def __str__(self) -> str:
        # Return focused error without infrastructure frames
        return super().__str__()

    def format_rich(self) -> None:
        """Format and print error with rich panels and code context."""
        if Console is None or Panel is None or Text is None:
            # Fallback to plain text
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
                tb_parts.append(
                    f'[cyan]{frame.filename}[/cyan]:[yellow]{frame.lineno}[/yellow] '
                    f'in [green]{frame.name}[/green]'
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

_PY_SET_ASYNC_EXC = ctypes.pythonapi.PyThreadState_SetAsyncExc
try:
    _PY_SET_ASYNC_EXC.argtypes = (ctypes.c_ulong, ctypes.py_object)  # type: ignore[attr-defined]
    _PY_SET_ASYNC_EXC.restype = ctypes.c_int  # type: ignore[attr-defined]
except AttributeError:  # pragma: no cover - platform specific
    pass


def _prune_dead_threads() -> None:
    with _SPEEDY_THREADS_LOCK:
        SPEEDY_RUNNING_THREADS[:] = [t for t in SPEEDY_RUNNING_THREADS if t.is_alive()]


def _track_threads(threads: Iterable[threading.Thread]) -> None:
    if not threads:
        return
    with _SPEEDY_THREADS_LOCK:
        living = [t for t in SPEEDY_RUNNING_THREADS if t.is_alive()]
        for candidate in threads:
            if not candidate.is_alive():
                continue
            if any(existing is candidate for existing in living):
                continue
            living.append(candidate)
        SPEEDY_RUNNING_THREADS[:] = living


def _track_executor_threads(pool: ThreadPoolExecutor) -> None:
    thread_set = getattr(pool, '_threads', None)
    if not thread_set:
        return
    _track_threads(tuple(thread_set))


def _group_iter(src: Iterable[T], size: int) -> Iterable[list[T]]:
    """Yield successive chunks from iterable of specified size."""
    it = iter(src)
    while chunk := list(islice(it, size)):
        yield chunk


def _worker(
    item: T,
    func: Callable[[T], R],
    fixed_kwargs: Mapping[str, Any],
    caller_frame: traceback.FrameSummary | None = None,
) -> R:
    """Execute the function with an item and fixed kwargs."""
    # Validate func is callable before attempting to call it
    if not callable(func):
        func_type = type(func).__name__
        raise TypeError(
            f'\nmulti_thread: func parameter must be callable, '
            f'got {func_type}: {func!r}\n'
            f'Hint: Did you accidentally pass a {func_type} instead of a function?'
        )

    try:
        return func(item)
    except Exception as exc:
        # Extract user code traceback (filter out infrastructure)
        exc_tb = sys.exc_info()[2]

        if exc_tb is not None:
            tb_list = traceback.extract_tb(exc_tb)

            # Filter to keep only user code frames
            user_frames = []
            skip_patterns = [
                'multi_worker/thread.py',
                'multi_worker/process.py',
                'concurrent/futures/',
                'threading.py',
                'multiprocessing/',
                'site-packages/ray/',
            ]

            for frame in tb_list:
                if not any(pattern in frame.filename for pattern in skip_patterns):
                    user_frames.append(frame)

            # If we have user frames, wrap in our custom exception
            if user_frames:
                func_name = getattr(func, '__name__', repr(func))
                raise UserFunctionError(
                    exc,
                    func_name,
                    item,
                    user_frames,
                    caller_frame,
                ) from exc

        # Fallback: re-raise original if we couldn't extract frames
        raise


def _run_batch(
    items: Sequence[T],
    func: Callable[[T], R],
    fixed_kwargs: Mapping[str, Any],
    caller_frame: traceback.FrameSummary | None = None,
) -> list[R]:
    return [_worker(item, func, fixed_kwargs, caller_frame) for item in items]


def _attach_metadata(fut: Future[Any], idx: int, logical_size: int) -> None:
    fut._speedy_idx = idx
    fut._speedy_size = logical_size


def _future_meta(fut: Future[Any]) -> tuple[int, int]:
    return (
        fut._speedy_idx,
        fut._speedy_size,
    )


class _ResultCollector(Generic[R]):
    def __init__(self, ordered: bool, logical_total: int | None) -> None:
        self._ordered = ordered
        self._logical_total = logical_total
        self._results: list[R | None]
        self._heap: list[tuple[int, list[R | None]]] | None
        self._next_idx = 0
        if ordered and logical_total is not None:
            self._results = [None] * logical_total
            self._heap = None
        else:
            self._results = []
            self._heap = [] if ordered else None

    def add(self, idx: int, items: Sequence[R | None]) -> None:
        if not items:
            return
        if self._ordered and self._logical_total is not None:
            self._results[idx : idx + len(items)] = list(items)
            return
        if self._ordered:
            assert self._heap is not None
            heappush(self._heap, (idx, list(items)))
            self._flush_ready()
            return
        self._results.extend(items)

    def _flush_ready(self) -> None:
        if self._heap is None:
            return
        while self._heap and self._heap[0][0] == self._next_idx:
            _, chunk = heappop(self._heap)
            self._results.extend(chunk)
            self._next_idx += len(chunk)

    def finalize(self) -> list[R | None]:
        self._flush_ready()
        return self._results


def _resolve_worker_count(workers: int | None) -> int:
    if workers is None:
        return DEFAULT_WORKERS
    if workers <= 0:
        raise ValueError('workers must be a positive integer')
    return workers


def _normalize_batch_result(result: Any, logical_size: int) -> list[Any]:
    if logical_size == 1:
        return [result]
    if result is None:
        raise ValueError('batched callable returned None for a batch result')
    if isinstance(result, (str, bytes, bytearray)):
        raise TypeError('batched callable must not return str/bytes when batching')
    if isinstance(result, (Sequence, Iterable)):
        out = list(result)
    else:
        raise TypeError('batched callable must return an iterable of results')
    if len(out) != logical_size:
        raise ValueError(
            f'batched callable returned {len(out)} items, expected {logical_size}',
        )
    return out


def _cancel_futures(inflight: set[Future[Any]]) -> None:
    for fut in inflight:
        fut.cancel()
    inflight.clear()


# ────────────────────────────────────────────────────────────
# main API
# ────────────────────────────────────────────────────────────
def multi_thread(
    func: Callable[[T], R],
    inputs: Iterable[T],
    *,
    workers: int | None = DEFAULT_WORKERS,
    batch: int = 1,
    ordered: bool = True,
    progress: bool = True,
    progress_update: int = 10,
    prefetch_factor: int = 4,
    timeout: float | None = None,
    stop_on_error: bool | None = None,
    error_handler: ErrorHandlerType = 'raise',
    max_error_files: int = 100,
    n_proc: int = 0,
    store_output_pkl_file: str | None = None,
    **fixed_kwargs: Any,
) -> list[R | None]:
    """Execute ``func`` over ``inputs`` using a managed thread pool.

    The scheduler supports batching, ordered result delivery, progress
    reporting, cooperative error handling, and a whole-run timeout.

    Parameters
    ----------
    func : Callable[[T], R]
        Target callable applied to each logical input.
    inputs : Iterable[T]
        Source iterable with input payloads.
    workers : int | None, optional
        Worker thread count (defaults to ``cpu_count()*2`` when ``None``).
    batch : int, optional
        Logical items grouped per invocation. ``1`` disables batching.
    ordered : bool, optional
        Preserve original ordering when ``True`` (default).
    progress : bool, optional
        Toggle tqdm-based progress reporting.
    progress_update : int, optional
        Minimum logical items between progress refreshes.
    prefetch_factor : int, optional
        Multiplier controlling in-flight items (``workers * prefetch_factor``).
    timeout : float | None, optional
        Overall wall-clock timeout in seconds.
    stop_on_error : bool | None, optional
        Deprecated. Use error_handler instead.
        When True -> error_handler='raise', when False -> error_handler='log'.
    error_handler : 'raise' | 'ignore' | 'log', optional
        - 'raise': raise exception on first error (default)
        - 'ignore': continue, return None for failed items
        - 'log': same as ignore, but logs errors to files
    max_error_files : int, optional
        Maximum number of error log files to write (default: 100).
        Error logs are written to .cache/speedy_utils/error_logs/{idx}.log
    n_proc : int, optional
        Optional process-level fan-out; ``>1`` shards work across processes.
    store_output_pkl_file : str | None, optional
        When provided, persist the results to disk via speedy_utils helpers.
    fixed_kwargs : dict[str, Any]
        Extra kwargs forwarded to every invocation of ``func``.

    Returns
    -------
    list[R | None]
        Collected results, preserving order when requested. Failed tasks yield
        ``None`` entries if ``error_handler`` is not 'raise'.
    """
    from speedy_utils import dump_json_or_pickle, load_by_ext

    # Handle deprecated stop_on_error parameter
    if stop_on_error is not None:
        import warnings
        warnings.warn(
            "stop_on_error is deprecated, use error_handler instead",
            DeprecationWarning,
            stacklevel=2
        )
        error_handler = 'raise' if stop_on_error else 'log'

    if n_proc > 1:
        import tempfile

        from fastcore.all import threaded

        items = list(inputs)
        if not items:
            return []
        n_per_proc = max(len(items) // n_proc, 1)
        chunks = [items[i : i + n_per_proc] for i in range(0, len(items), n_per_proc)]
        procs = []
        in_process_multi_thread = threaded(process=True)(multi_thread)
        results: list[R | None] = []

        for proc_idx, chunk in enumerate(chunks):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix='multi_thread.pkl'
            ) as fh:
                file_pkl = fh.name
            assert isinstance(in_process_multi_thread, Callable)
            proc = in_process_multi_thread(
                func,
                chunk,
                workers=workers,
                batch=batch,
                ordered=ordered,
                progress=proc_idx == 0,
                progress_update=progress_update,
                prefetch_factor=prefetch_factor,
                timeout=timeout,
                error_handler=error_handler,
                max_error_files=max_error_files,
                n_proc=0,
                store_output_pkl_file=file_pkl,
                **fixed_kwargs,
            )
            procs.append((proc, file_pkl))

        for proc, file_pkl in procs:
            proc.join()
            logger.info('process finished: %s', proc)
            try:
                results.extend(load_by_ext(file_pkl))
            finally:
                try:
                    os.unlink(file_pkl)
                except OSError as exc:  # pragma: no cover - best effort cleanup
                    logger.warning('failed to remove temp file %s: %s', file_pkl, exc)
        return results

    try:
        import pandas as pd

        if isinstance(inputs, pd.DataFrame):
            inputs = cast(Iterable[T], inputs.to_dict(orient='records'))
    except ImportError:  # pragma: no cover - optional dependency
        pass

    if batch <= 0:
        raise ValueError('batch must be a positive integer')
    if prefetch_factor <= 0:
        raise ValueError('prefetch_factor must be a positive integer')

    workers_val = _resolve_worker_count(workers)
    progress_update = max(progress_update, 1)
    fixed_kwargs_map: Mapping[str, Any] = MappingProxyType(dict(fixed_kwargs))

    try:
        logical_total = len(inputs)  # type: ignore[arg-type]
    except Exception:  # pragma: no cover - generic iterable
        logical_total = None

    if batch == 1 and logical_total and logical_total / max(workers_val, 1) > 20_000:
        batch = 32

    src_iter: Iterator[Any] = iter(inputs)
    if batch > 1:
        src_iter = iter(_group_iter(src_iter, batch))
    collector: _ResultCollector[Any] = _ResultCollector(ordered, logical_total)

    # Initialize error stats for error handling
    func_name = getattr(func, '__name__', repr(func))
    error_stats = ErrorStats(
        func_name=func_name,
        max_error_files=max_error_files,
        write_logs=error_handler == 'log'
    )

    # Convert inputs to list for index access in error logging
    items_list: list[Any] | None = None
    if error_handler != 'raise':
        try:
            items_list = list(inputs)
            src_iter = iter(items_list)
            if batch > 1:
                src_iter = iter(_group_iter(src_iter, batch))
        except Exception:
            items_list = None

    bar = None
    last_bar_update = 0
    if (
        progress
        and tqdm is not None
        and logical_total is not None
        and logical_total > 0
    ):
        bar = tqdm(
            total=logical_total,
            ncols=128,
            colour='green',
            bar_format=(
                '{l_bar}{bar}| {n_fmt}/{total_fmt} '
                '[{elapsed}<{remaining}, {rate_fmt}{postfix}]'
            ),
        )

    # Capture caller context for error reporting
    caller_frame_obj = inspect.currentframe()
    caller_context: traceback.FrameSummary | None = None
    if caller_frame_obj and caller_frame_obj.f_back:
        caller_info = inspect.getframeinfo(caller_frame_obj.f_back)
        caller_context = traceback.FrameSummary(
            caller_info.filename,
            caller_info.lineno,
            caller_info.function,
        )

    deadline = time.monotonic() + timeout if timeout is not None else None
    max_inflight = max(workers_val * prefetch_factor, 1)
    completed_items = 0
    next_logical_idx = 0

    def items_inflight() -> int:
        return next_logical_idx - completed_items

    inflight: set[Future[Any]] = set()
    pool = ThreadPoolExecutor(
        max_workers=workers_val,
        thread_name_prefix='speedy-thread',
    )
    shutdown_kwargs: dict[str, Any] = {'wait': True}

    try:

        def submit_arg(arg: Any) -> None:
            nonlocal next_logical_idx
            if batch > 1:
                batch_items = list(arg)
                if not batch_items:
                    return
                fut = pool.submit(_run_batch, batch_items, func, fixed_kwargs_map, caller_context)
                logical_size = len(batch_items)
            else:
                fut = pool.submit(_worker, arg, func, fixed_kwargs_map, caller_context)
                logical_size = 1
            _attach_metadata(fut, next_logical_idx, logical_size)
            next_logical_idx += logical_size
            inflight.add(fut)
            _track_executor_threads(pool)

        try:
            while items_inflight() < max_inflight:
                submit_arg(next(src_iter))
        except StopIteration:
            pass

        while inflight:
            wait_timeout = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _cancel_futures(inflight)
                    raise TimeoutError(
                        f'multi_thread timed out after {timeout} seconds',
                    )
                wait_timeout = max(remaining, 0.0)

            done, _ = wait(
                inflight,
                timeout=wait_timeout,
                return_when=FIRST_COMPLETED,
            )

            if not done:
                _cancel_futures(inflight)
                raise TimeoutError(
                    f'multi_thread timed out after {timeout} seconds',
                )

            for fut in done:
                inflight.remove(fut)
                idx, logical_size = _future_meta(fut)
                try:
                    result = fut.result()
                    # Record success for each item in the batch
                    for _ in range(logical_size):
                        error_stats.record_success()
                except UserFunctionError as exc:
                    # User function error
                    if error_handler == 'raise':
                        sys.stderr.flush()
                        sys.stdout.flush()
                        exc.format_rich()
                        sys.stderr.flush()
                        _cancel_futures(inflight)
                        sys.exit(1)
                    
                    # Log error with ErrorStats
                    input_val = None
                    if items_list is not None and idx < len(items_list):
                        input_val = items_list[idx]
                    error_stats.record_error(
                        idx, exc.original_exception, input_val, func_name
                    )
                    out_items = [None] * logical_size
                except Exception as exc:
                    # Other errors (infrastructure, batching, etc.)
                    if error_handler == 'raise':
                        _cancel_futures(inflight)
                        raise
                    
                    input_val = None
                    if items_list is not None and idx < len(items_list):
                        input_val = items_list[idx]
                    error_stats.record_error(idx, exc, input_val, func_name)
                    out_items = [None] * logical_size
                else:
                    try:
                        out_items = _normalize_batch_result(result, logical_size)
                    except Exception as exc:
                        _cancel_futures(inflight)
                        raise RuntimeError(
                            'batched callable returned an unexpected shape',
                        ) from exc

                collector.add(idx, out_items)
                completed_items += len(out_items)

                if bar:
                    delta = completed_items - last_bar_update
                    if delta >= progress_update:
                        bar.update(delta)
                        last_bar_update = completed_items
                        submitted = next_logical_idx
                        pending: int | str = (
                            max(logical_total - submitted, 0)
                            if logical_total is not None
                            else '-'
                        )
                        postfix: dict[str, Any] = error_stats.get_postfix_dict()
                        postfix['pending'] = pending
                        bar.set_postfix(postfix)

            try:
                while items_inflight() < max_inflight:
                    submit_arg(next(src_iter))
            except StopIteration:
                pass

        results = collector.finalize()

    except KeyboardInterrupt:
        shutdown_kwargs = {'wait': False, 'cancel_futures': True}
        _cancel_futures(inflight)
        kill_all_thread(SystemExit)
        raise KeyboardInterrupt() from None
    finally:
        try:
            pool.shutdown(**shutdown_kwargs)
        except TypeError:  # pragma: no cover - Python <3.9 fallback
            pool.shutdown(shutdown_kwargs.get('wait', True))
        if bar:
            delta = completed_items - last_bar_update
            if delta > 0:
                bar.update(delta)
            bar.close()

    results = collector.finalize() if 'results' not in locals() else results
    if store_output_pkl_file:
        dump_json_or_pickle(results, store_output_pkl_file)
    _prune_dead_threads()
    return results


def multi_thread_standard(
    fn: Callable[[T], R], items: Iterable[T], workers: int = 4
) -> list[R]:
    """Execute ``fn`` across ``items`` while preserving submission order."""

    workers_val = _resolve_worker_count(workers)
    with ThreadPoolExecutor(
        max_workers=workers_val,
        thread_name_prefix='speedy-thread',
    ) as executor:
        futures: list[Future[R]] = []
        for item in items:
            futures.append(executor.submit(fn, item))
        _track_executor_threads(executor)
        results = [fut.result() for fut in futures]
    _prune_dead_threads()
    return results


def _async_raise(thread_id: int, exc_type: type[BaseException]) -> bool:
    if thread_id <= 0:
        return False
    if not issubclass(exc_type, BaseException):
        raise TypeError('exc_type must derive from BaseException')
    res = _PY_SET_ASYNC_EXC(ctypes.c_ulong(thread_id), ctypes.py_object(exc_type))
    if res == 0:
        return False
    if res > 1:  # pragma: no cover - defensive branch
        _PY_SET_ASYNC_EXC(ctypes.c_ulong(thread_id), None)
        raise SystemError('PyThreadState_SetAsyncExc failed')
    return True


def kill_all_thread(
    exc_type: type[BaseException] = SystemExit, join_timeout: float = 0.1
) -> int:
    """Forcefully stop tracked worker threads (dangerous; use sparingly).

    Returns
    -------
    int
        Count of threads signalled for termination.
    """
    _prune_dead_threads()
    current = threading.current_thread()
    with _SPEEDY_THREADS_LOCK:
        targets = [t for t in SPEEDY_RUNNING_THREADS if t.is_alive()]

    terminated = 0
    for thread in targets:
        if thread is current:
            continue
        ident = thread.ident
        if ident is None:
            continue
        try:
            if _async_raise(ident, exc_type):
                terminated += 1
                thread.join(timeout=join_timeout)
            else:
                logger.warning('Unable to signal thread %s', thread.name)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error('Failed to stop thread %s: %s', thread.name, exc)
    _prune_dead_threads()
    return terminated


__all__ = [
    'SPEEDY_RUNNING_THREADS',
    'UserFunctionError',
    'multi_thread',
    'multi_thread_standard',
    'kill_all_thread',
]
