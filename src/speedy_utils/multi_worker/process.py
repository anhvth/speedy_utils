# ray_multi_process.py
import time, os, pickle, uuid, datetime
from pathlib import Path
from typing import Any, Callable
from tqdm import tqdm
import ray
from fastcore.parallel import parallel

# ─── cache helpers ──────────────────────────────────────────

def _build_cache_dir(func: Callable, items: list[Any]) -> Path:
    """Build cache dir with function name + timestamp."""
    func_name = getattr(func, "__name__", "func")
    now = datetime.datetime.now()
    stamp = now.strftime("%m%d_%Hh%Mm%Ss")
    run_id = f"{func_name}_{stamp}_{uuid.uuid4().hex[:6]}"
    path = Path(".cache") / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path

def wrap_dump(func: Callable, cache_dir: Path | None):
    """Wrap a function so results are dumped to .pkl when cache_dir is set."""
    if cache_dir is None:
        return func

    def wrapped(x, *args, **kwargs):
        res = func(x, *args, **kwargs)
        p = cache_dir / f"{uuid.uuid4().hex}.pkl"
        with open(p, "wb") as fh:
            pickle.dump(res, fh)
        return str(p)
    return wrapped

# ─── ray management ─────────────────────────────────────────

RAY_WORKER = None

def ensure_ray(workers: int, pbar: tqdm | None = None):
    """Initialize or reinitialize Ray with a given worker count, log to bar postfix."""
    global RAY_WORKER
    if not ray.is_initialized() or RAY_WORKER != workers:
        if ray.is_initialized() and pbar:
            pbar.set_postfix_str(f"Restarting Ray {workers} workers")
            ray.shutdown()
        t0 = time.time()
        ray.init(num_cpus=workers, ignore_reinit_error=True)
        took = time.time() - t0
        if pbar:
            pbar.set_postfix_str(f"ray.init {workers} took {took:.2f}s")
        RAY_WORKER = workers

# ─── main API ───────────────────────────────────────────────
from typing import Literal

def multi_process(
    func: Callable[[Any], Any],
    items: list[Any] | None = None,
    *,
    inputs: list[Any] | None = None,
    workers: int | None = None,
    lazy_output: bool = False,
    progress: bool = True,
    # backend: str = "ray",   # "seq", "ray", or "fastcore"
    backend: Literal["seq", "ray", "mp", "threadpool"] = "ray",
    # Additional optional knobs (accepted for compatibility)
    batch: int | None = None,
    ordered: bool | None = None,
    process_update_interval: int | None = None,
    stop_on_error: bool | None = None,
    **func_kwargs: Any,
) -> list[Any]:
    """
    Multi-process map with selectable backend.

    backend:
        - "seq": run sequentially
        - "ray": run in parallel with Ray
        - "fastcore": run in parallel with fastcore.parallel

    If lazy_output=True, every result is saved to .pkl and
    the returned list contains file paths.
    """

    # unify items
    if items is None and inputs is not None:
        items = list(inputs)
    if items is None:
        raise ValueError("'items' or 'inputs' must be provided")

    if workers is None:
        workers = os.cpu_count() or 1

    # build cache dir + wrap func
    cache_dir = _build_cache_dir(func, items) if lazy_output else None
    f_wrapped = wrap_dump(func, cache_dir)

    total = len(items)
    with tqdm(total=total, desc=f"multi_process [{backend}]", disable=not progress) as pbar:

        # ---- sequential backend ----
        if backend == "seq":
            pbar.set_postfix_str("backend=seq")
            results = []
            for x in items:
                results.append(f_wrapped(x, **func_kwargs))
                pbar.update(1)
            return results

        # ---- ray backend ----
        if backend == "ray":
            pbar.set_postfix_str("backend=ray")
            ensure_ray(workers, pbar)

            @ray.remote
            def _task(x):
                return f_wrapped(x, **func_kwargs)

            refs = [_task.remote(x) for x in items]

            results = []
            for r in refs:
                results.append(ray.get(r))
                pbar.update(1)
            return results

        # ---- fastcore backend ----
        if backend == "mp":
            results = parallel(f_wrapped, items, n_workers=workers, progress=progress, threadpool=False)
            return list(results)
        if backend == "threadpool":
            results = parallel(f_wrapped, items, n_workers=workers, progress=progress, threadpool=True)
            return list(results)

        raise ValueError(f"Unsupported backend: {backend!r}")
