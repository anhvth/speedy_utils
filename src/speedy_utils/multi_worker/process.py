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

from ..__imports import *
import tempfile
from .progress import create_progress_tracker, ProgressPoller, get_ray_progress_actor


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
_DUMP_THREADS = []
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
            _DUMP_THREADS.append(thread)
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
    # backend: str = "ray",   # "seq", "ray", or "fastcore"
    backend: Literal['seq', 'ray', 'mp', 'threadpool', 'safe'] = 'mp',
    desc: str | None = None,
    shared_kwargs: list[str] | None = None,
    dump_in_thread: bool = True,
    ray_metrics_port: int | None = None,
    log_worker: Literal['zero', 'first', 'all'] = 'first',
    total_items: int | None = None,
    poll_interval: float = 0.3,
    **func_kwargs: Any,
) -> list[Any]:
    """
    Multi-process map with selectable backend.

    backend:
        - "seq": run sequentially
        - "ray": run in parallel with Ray
        - "mp": run in parallel with multiprocessing (uses threadpool to avoid fork warnings)
        - "threadpool": run in parallel with thread pool
        - "safe": run in parallel with thread pool (explicitly safe for tests)

    shared_kwargs:
        - Optional list of kwarg names that should be shared via Ray's zero-copy object store
        - Only works with Ray backend
        - Useful for large objects (e.g., models, datasets) that should be shared across workers
        - Example: shared_kwargs=['model', 'tokenizer'] for sharing large ML models

    dump_in_thread:
        - Whether to dump results to disk in a separate thread (default: True)
        - If False, dumping is done synchronously, which may block but ensures data is saved before returning

    ray_metrics_port:
        - Optional port for Ray metrics export (Ray backend only)
        - Set to 0 to disable Ray metrics
        - If None, uses Ray's default behavior

    log_worker:
        - Control worker stdout/stderr noise
        - 'first': only first worker emits logs, others are silenced (default)
        - 'all': allow worker prints (may overlap tqdm)
        - 'zero': silence worker stdout/stderr to keep progress bar clean

    total_items:
        - Optional item-level total for progress tracking (Ray backend only)

    poll_interval:
        - Poll interval in seconds for progress actor updates (Ray backend only)

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
    if backend != 'ray':
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

    # ---- sequential backend ----
    if backend == 'seq':
        results: list[Any] = []
        with tqdm(total=total, desc=desc, disable=not progress, file=sys.stdout) as pbar:
            for x in items:
                results.append(
                    _call_with_log_control(
                        f_wrapped,
                        x,
                        func_kwargs,
                        log_worker,
                        log_gate_path,
                    )
                )
                pbar.update(1)
        _cleanup_log_gate(log_gate_path)
        return results

    # ---- ray backend ----
    if backend == 'ray':
        import ray as _ray_module

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
                shared_refs['progress_actor'] = progress_actor  # Pass actor handle directly (not via put)

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
                        # Pass actor handle directly (don't dereference)
                        dereferenced[k] = v
                    else:
                        dereferenced[k] = _ray_in_task.get(v)
                # Merge with regular kwargs
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
            
            # Start progress poller if using item-level progress
            if progress_actor is not None:
                # Update pbar total to show items instead of tasks
                pbar.total = total_items
                pbar.refresh()
                progress_poller = ProgressPoller(progress_actor, pbar, poll_interval)
                progress_poller.start()
            
            for r in refs:
                results.append(_ray_module.get(r))
                if progress_actor is None:
                    # Only update task-level progress if not using item-level
                    pbar.update(1)
            
            # Stop progress poller
            if progress_poller is not None:
                progress_poller.stop()
                
            t_end = time.time()
            item_desc = f"{total_items:,} items" if total_items else f"{total} tasks"
            print(f"Ray processing took {t_end - t_start:.2f}s for {item_desc}")
        _cleanup_log_gate(log_gate_path)
        return results

    # ---- fastcore backend ----
    if backend == 'mp':
        worker_func = functools.partial(
            _call_with_log_control,
            f_wrapped,
            func_kwargs=func_kwargs,
            log_mode=log_worker,
            gate_path=log_gate_path,
        )
        with _patch_fastcore_progress_bar(leave=True):
            results = list(parallel(
                worker_func, items, n_workers=workers, progress=progress, threadpool=False
            ))
        _track_multiprocessing_processes()  # Track multiprocessing workers
        _prune_dead_processes()  # Clean up dead processes
        _cleanup_log_gate(log_gate_path)
        return results

    if backend == 'threadpool':
        worker_func = functools.partial(
            _call_with_log_control,
            f_wrapped,
            func_kwargs=func_kwargs,
            log_mode=log_worker,
            gate_path=log_gate_path,
        )
        with _patch_fastcore_progress_bar(leave=True):
            results = list(parallel(
                worker_func, items, n_workers=workers, progress=progress, threadpool=True
            ))
        _cleanup_log_gate(log_gate_path)
        return results

    if backend == 'safe':
        # Completely safe backend for tests - no multiprocessing, no external progress bars
        import concurrent.futures

        # Import thread tracking from thread module
        try:
            from .thread import _prune_dead_threads, _track_executor_threads

            worker_func = functools.partial(
                _call_with_log_control,
                f_wrapped,
                func_kwargs=func_kwargs,
                log_mode=log_worker,
                gate_path=log_gate_path,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                _track_executor_threads(executor)  # Track threads
                results = list(executor.map(worker_func, items))
            _prune_dead_threads()  # Clean up dead threads
        except ImportError:
            # Fallback if thread module not available
            worker_func = functools.partial(
                _call_with_log_control,
                f_wrapped,
                func_kwargs=func_kwargs,
                log_mode=log_worker,
                gate_path=log_gate_path,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers
            ) as executor:
                results = list(executor.map(worker_func, items))
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
    'multi_process',
    'cleanup_phantom_workers',
    'create_progress_tracker',
    'get_ray_progress_actor',
]
