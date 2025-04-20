"""
multi_thread.py ― fast, low‑overhead threaded map helper
========================================================
• Streams results internally, but finally **returns a list** to match the
  original signature.
• Batch very small tasks (``batch``) to remove micro‑task overhead.
• Sane default worker count, optional ordering, optional tqdm progress bar.
"""
from __future__ import annotations

import inspect
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from itertools import islice
from typing import Any, Iterable, List, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None                                       # type: ignore[assignment]

# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _signature_kwargs(func, arg) -> dict[str, Any]:
    """Turn *arg* into **kwargs** that match *func*’s signature."""
    params = list(inspect.signature(func).parameters)
    if len(params) == 1:                   # single positional param
        return {params[0]: arg}
    if isinstance(arg, (list, tuple, Sequence)):
        return dict(zip(params, arg))
    if isinstance(arg, dict):
        return arg
    return {}

def _group_iter(src: Iterable[Any], size: int) -> Iterable[list[Any]]:
    it = iter(src)
    while chunk := list(islice(it, size)):
        yield chunk

def _flatten(x):
    if isinstance(x, list):
        return x
    return [x]

def _short_tb() -> str:
    tb = "".join(traceback.format_exc())
    return "\n".join(ln for ln in tb.splitlines() if "multi_thread.py" not in ln)

def _safe_call(func, arg, fixed_kwargs):
    try:
        return func(**_signature_kwargs(func, arg), **fixed_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"{func.__name__}({repr(arg)}) failed: {e}\n{_short_tb()}"
        ) from e

# ──────────────────────────────────────────────────────────────────────────────
# main API
# ──────────────────────────────────────────────────────────────────────────────
def multi_thread(
    func: callable,
    inputs: Iterable[Any],
    *,
    workers: int | None = None,
    batch: int = 1,
    ordered: bool = True,
    progress: bool = False,
    prefetch_factor: int = 4,
    timeout: float | None = None,
    stop_on_error: bool = True,
    **fixed_kwargs,
) -> List[Any]:
    """
    High‑throughput ThreadPool **map** that returns a *list*.

    Parameters
    ----------
    func         – target function.
    inputs       – iterable with the inputs.
    workers      – defaults to `min(os.cpu_count()*2, 32)`.
    batch        – package *batch* inputs into one call to amortise overhead.
    ordered      – keep original order (costs memory); if `False`, outputs
                   arrive as soon as they complete.
    progress     – use `tqdm` if installed.
    """
    t0 = time.perf_counter()

    src_list = list(inputs) if not hasattr(inputs, "__len__") else inputs
    total = len(src_list)
    if batch > 1:
        src_iter = _group_iter(src_list, batch)
        total_batches = (total + batch - 1) // batch
    else:
        src_iter = iter(src_list)
        total_batches = total

    workers = workers or min(os.cpu_count() * 2, 32)
    max_inflight = workers * max(prefetch_factor, 1)

    bar = None
    if progress and tqdm is not None:
        bar = tqdm(
            total=total,
            ncols=80,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )

    results: list[Any] = []
    buffer: dict[int, Any] = {}
    next_emit_idx = 0

    def _worker(item_batch):
        if batch > 1:
            return [_safe_call(func, itm, fixed_kwargs) for itm in item_batch]
        return _safe_call(func, item_batch, fixed_kwargs)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        # Prime the pool
        for idx, item in enumerate(islice(src_iter, max_inflight)):
            futures[pool.submit(_worker, item)] = idx
        next_submit_idx = len(futures)

        try:
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED, timeout=timeout)
                for fut in done:
                    idx = futures.pop(fut)
                    try:
                        res = fut.result()
                    except Exception:
                        pool.shutdown(cancel_futures=True)
                        if stop_on_error:
                            raise
                        res = None

                    if ordered:
                        buffer[idx] = res
                        while next_emit_idx in buffer:
                            results.extend(_flatten(buffer.pop(next_emit_idx)))
                            if bar:
                                bar.update(batch if batch > 1 else 1)
                            next_emit_idx += 1
                    else:
                        results.extend(_flatten(res))
                        if bar:
                            bar.update(batch if batch > 1 else 1)

                    # keep queue full
                    try:
                        new_item = next(src_iter)
                        futures[pool.submit(_worker, new_item)] = next_submit_idx
                        next_submit_idx += 1
                    except StopIteration:
                        pass
        finally:
            if bar:
                bar.close()

    if __debug__ and not progress:
        print(f"[multi_thread] completed {len(results)} items in {time.perf_counter() - t0:.4f}s")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# internals
# ──────────────────────────────────────────────────────────────────────────────
def _flatten(x):
    """Yield x if list‑like else yield [x]."""
    if isinstance(x, list):
        yield from x
    else:
        yield x


def _safe_call(func, arg, fixed_kwargs):
    try:
        return func(**_signature_kwargs(func, arg), **fixed_kwargs)
    except Exception as e:  # attach minimal traceback
        raise RuntimeError(
            f"{func.__name__}({repr(arg)}) failed: {e}\n{_short_tb()}"
        ) from e


def _short_tb() -> str:
    tb = "".join(traceback.format_exc())
    # remove frames inside this helper to keep logs tidy
    lines = [ln for ln in tb.splitlines() if "multi_thread.py" not in ln]
    return "\n".join(lines)


__all__ = ["multi_thread"]

# ──────────────────────────────────────────────────────────────────────────────
# self‑test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    def f(x):
        time.sleep(random.random())
        return x

    inputs = range(1000)
    t = time.time()
    outs = list(multi_thread(
        f,
        inputs,
        workers=1000,
        progress=True,
        prefetch_factor=4,
    ))
    mt_time = time.time() - t
    print(f"[multi_thread] time: {mt_time:.4f}s")
    print(f"[multi_thread] results: {outs[:10]}...{outs[-10:]}")