from __future__ import annotations

import inspect, os, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from typing import Any, Callable, Iterable, List, Sequence

try:
    from tqdm import tqdm
except ImportError:          # pragma: no cover
    tqdm = None              # type: ignore[assignment]


# ──── internal helpers ────────────────────────────────────────────────────
def _sig_kwargs(func: Callable, arg) -> dict[str, Any]:
    "Smartly convert *arg* into **kwargs** for *func* (same logic as multi_thread)."
    params = list(inspect.signature(func).parameters)

    # mapping input → maybe‑kwargs
    if isinstance(arg, dict):
        return arg if set(arg).issubset(params) else {params[0]: arg}

    # 1‑element Sequence → keep wrapper; >1 → positional unpack
    if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes, bytearray)):
        return dict(zip(params, arg)) if len(arg) > 1 else {params[0]: arg}

    # scalar fallback
    return {params[0]: arg}


def _group_iter(src: Iterable[Any], size: int) -> Iterable[List[Any]]:
    "Yield *size*-sized chunks from *src*."
    it = iter(src)
    while chunk := list(islice(it, size)):
        yield chunk


def _short_tb() -> str:
    "Trim internal frames from a traceback for compact logs."
    tb = "".join(traceback.format_exc())
    return "\n".join(ln for ln in tb.splitlines() if "multi_process" not in ln)


def _safe_call(func: Callable, arg, fixed):
    try:
        return func(**_sig_kwargs(func, arg), **fixed)
    except Exception as exc:
        raise RuntimeError(f"{func.__name__}({arg!r}) failed: {exc}\n{_short_tb()}") from exc


# Define _worker at the module level to ensure it's picklable
def _worker_process(func: Callable, item_batch: Any, fixed_kwargs: dict, batch_size: int):
    """Worker function executed in each process."""
    if batch_size > 1:
        results = []
        for itm in item_batch:
            try:
                results.append(_safe_call(func, itm, fixed_kwargs))
            except Exception:
                # If an error occurs for an item in the batch, append None
                # This ensures the output list length matches the input batch length
                results.append(None)
        return results
    # For batch_size == 1, _safe_call handles the exception and raises if needed
    # If stop_on_error=False, the main loop will catch it and set res=None
    return _safe_call(func, item_batch, fixed_kwargs)


# ──── public API ──────────────────────────────────────────────────────────
def multi_process(
    func: Callable[[Any], Any],
    inputs: Iterable[Any],
    *,
    workers: int | None = None,
    batch: int = 1,
    ordered: bool = True,
    progress: bool = False,
    inflight: int | None = None,
    timeout: float | None = None,
    stop_on_error: bool = True,
    **fixed_kwargs,
) -> List[Any]:
    """
    Cross‑platform **multi‑processing** parallel map that returns a *list*.

    Parameters
    ----------
    func          – target callable executed in separate processes.
    inputs        – iterable with the arguments.
    workers       – process pool size (defaults to :pyfunc:`os.cpu_count()`).
    batch         – package *batch* inputs into one call to reduce IPC overhead.
    ordered       – keep original order; if ``False`` results stream as finished.
    progress      – show a tqdm bar (requires *tqdm*).
    inflight      – max logical items concurrently submitted  
                    *(default: ``workers × 4``)*.
    timeout       – overall timeout for the mapping (seconds).
    stop_on_error – raise immediately on first exception (default) or
                    substitute failing result with ``None``.
    **fixed_kwargs – static keyword args forwarded to every ``func()`` call.
    """
    if workers is None:
        workers = os.cpu_count() or 1
    if inflight is None:
        inflight = workers * 4
    if batch < 1:
        raise ValueError("batch must be ≥ 1")

    # figure out total length if cheaply available
    try:
        n_inputs = len(inputs)  # type: ignore[arg-type]
    except Exception:
        n_inputs = None

    # heuristic: auto‑batch tiny tasks if user left batch=1
    if batch == 1 and n_inputs and n_inputs / workers > 20_000:
        batch = 32

    # wrap source iterator
    src_iter: Iterable[Any] = iter(inputs)
    if batch > 1:
        src_iter = _group_iter(src_iter, batch)

    logical_total = n_inputs            # for progress & ordered pre‑alloc
    bar = None
    if progress and tqdm is not None and logical_total is not None:
        bar = tqdm(
            total=logical_total,
            ncols=80,
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
            " [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        last_bar = 0

    # pre‑allocate result list if we must keep order
    if ordered and logical_total is not None:
        results: List[Any] = [None] * logical_total
    else:
        results = []

    completed = 0
    next_idx = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = set()

        # prime pool
        for _ in range(min(inflight, workers)):
            try:
                arg = next(src_iter)
            except StopIteration:
                break
            # Use the top-level _worker_process function
            fut = pool.submit(_worker_process, func, arg, fixed_kwargs, batch)
            fut.idx = next_idx  # type: ignore[attr-defined]
            futures.add(fut)
            next_idx += len(arg) if batch > 1 else 1

        while futures:
            for fut in as_completed(futures, timeout=timeout):
                futures.remove(fut)
                idx = fut.idx  # type: ignore[attr-defined]
                try:
                    res = fut.result()
                except Exception:
                    if stop_on_error:
                        raise
                    # Determine the number of items expected for this future
                    num_items = batch if batch > 1 else 1
                    # If an exception occurred fetching the result (e.g., worker process died)
                    # and stop_on_error is False, fill with None
                    res = [None] * num_items if batch > 1 else None

                # flatten batched outputs if necessary
                out_items = res if batch > 1 else [res]

                if ordered and logical_total is not None:
                    # Ensure out_items has the correct length even if res was None (for batch > 1)
                    if len(out_items) != (batch if batch > 1 else 1) and batch > 1:
                         # This case might happen if the worker process died unexpectedly
                         # We already handled this by setting res = [None] * batch above
                         pass # Should be correctly sized now
                    results[idx : idx + len(out_items)] = out_items
                else:
                    results.extend(out_items)

                completed += len(out_items)

                # progress
                if bar and completed - last_bar >= 500:
                    bar.update(completed - last_bar)
                    last_bar = completed

                # keep pool saturated
                try:
                    while next_idx - completed < inflight:
                        arg = next(src_iter)
                        # Use the top-level _worker_process function
                        fut2 = pool.submit(_worker_process, func, arg, fixed_kwargs, batch)
                        fut2.idx = next_idx  # type: ignore[attr-defined]
                        futures.add(fut2)
                        next_idx += len(arg) if batch > 1 else 1
                except StopIteration:
                    pass
                break      # process one future per outer loop iteration

    if bar:
        bar.update(completed - last_bar)
        bar.close()

    return results

__all__ = ["multi_process"]