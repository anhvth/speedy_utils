from __future__ import annotations

import inspect
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from typing import Any, Iterable, List, Sequence
from fastcore.foundation import defaults

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

# ── choose sensible defaults ──────────────────────────────────────────

DEFAULT_WORKERS = os.cpu_count() * 2# ────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────
# ─── helpers ─────────────────────────────────────────────────────────────
from collections.abc import Sequence  # (typing.Sequence misses str checks)


def _sig_kwargs(func, arg) -> dict[str, Any]:
    """
    Convert *arg* into **kwargs** for *func*.

    • dict  – unpack **only if** keys ⊆ func params, else bind whole dict.
    • Sequence (≠ str/bytes) with **> 1 elements** – unpack positionally.
      • 1‑element sequences are treated like scalars to preserve the container.
    • scalar / string – bind to the first parameter.
    """
    params = list(inspect.signature(func).parameters)

    # dict input --------------------------------------------------------
    if isinstance(arg, dict):
        if set(arg).issubset(params):
            return arg  # looks like genuine **kwargs
        return {params[0]: arg}  # dict is a single logical value

    # tuple / list / etc. ----------------------------------------------
    if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes, bytearray)):
        if len(arg) > 1:  # real positional unpacking
            return dict(zip(params, arg))
        return {params[0]: arg}  # 1‑element → keep wrapper intact

    # scalar fallback ---------------------------------------------------
    return {params[0]: arg}


def _group_iter(src: Iterable[Any], size: int) -> Iterable[list[Any]]:
    it = iter(src)
    while chunk := list(islice(it, size)):
        yield chunk


def _short_tb() -> str:
    tb = "".join(traceback.format_exc())
    # hide frames inside this helper to keep logs short
    return "\n".join(ln for ln in tb.splitlines() if "multi_thread.py" not in ln)


def _safe_call(func, arg, fixed):
    try:
        return func(**_sig_kwargs(func, arg), **fixed)
    except Exception as exc:
        raise RuntimeError(
            f"{func.__name__}({arg!r}) failed: {exc}\n{_short_tb()}"
        ) from exc


# ────────────────────────────────────────────────────────────
# main API
# ────────────────────────────────────────────────────────────
def multi_thread(
    func: callable,
    inputs: Iterable[Any],
    *,
    workers: int | None = DEFAULT_WORKERS,
    batch: int = 1,
    ordered: bool = True,
    progress: bool = False,
    progress_update: int = 500,
    prefetch_factor: int = 4,
    timeout: float | None = None,
    stop_on_error: bool = True,
    **fixed_kwargs,
) -> List[Any]:
    """
    ThreadPool **map** that returns a *list*.

    Parameters
    ----------
    func            – target callable.
    inputs          – iterable with the arguments.
    workers         – defaults to ``os.cpu_count()*2``.
    batch           – package *batch* inputs into one call for low‑overhead.
    ordered         – keep original order (costs memory); if ``False`` results
                      are yielded as soon as they finish.
    progress        – show a tqdm bar (requires *tqdm* installed).
    progress_update – bar redraw frequency (logical items, *not* batches).
    prefetch_factor – in‑flight tasks ≈ ``workers * prefetch_factor``.
    timeout         – overall timeout (seconds) for the mapping.
    stop_on_error   – raise immediately on first exception (default).  If
                      ``False`` the failing task’s result becomes ``None``.
    **fixed_kwargs  – static keyword args forwarded to every ``func()`` call.
    """
    t0 = time.perf_counter()



    # Tiny‑task heuristic: if we have *lots* of inputs and the user left batch=1
    # we gently pick a small auto‑batch to avoid pure overhead.
    try:
        n_inputs = len(inputs)  # type: ignore[arg-type]
    except Exception:
        n_inputs = None
    if batch == 1 and n_inputs and n_inputs / max(workers, 1) > 20_000:
        batch = 32  # empirically good for sub‑ms tasks

    # ── build (maybe‑batched) source iterator ────────────────────────────
    src_iter: Iterable[Any] = iter(inputs)
    if batch > 1:
        src_iter = _group_iter(src_iter, batch)

    # total logical items (for bar & ordered pre‑allocation)
    logical_total = n_inputs
    if logical_total is not None and batch > 1:
        logical_total = n_inputs  # still number of *items*, not batches

    # ── progress bar ─────────────────────────────────────────────────────
    bar = None
    if progress and tqdm is not None and logical_total is not None:
        bar = tqdm(
            total=logical_total,
            ncols=80,
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
            " [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        last_bar_update = 0

    # ── prepare result container ─────────────────────────────────────────
    if ordered and logical_total is not None:
        results: list[Any] = [None] * logical_total
    else:
        results = []

    # ── worker wrapper ───────────────────────────────────────────────────
    def _worker(item_batch):
        if batch > 1:
            return [_safe_call(func, itm, fixed_kwargs) for itm in item_batch]
        return _safe_call(func, item_batch, fixed_kwargs)

    # ── main execution loop ──────────────────────────────────────────────────
    max_inflight = workers * max(prefetch_factor, 1)
    completed_items = 0
    next_logical_idx = 0  # index assigned to the next submission

    with ThreadPoolExecutor(max_workers=workers) as pool:
        inflight = set()

        # prime the pool
        for _ in range(max_inflight):
            try:
                arg = next(src_iter)
            except StopIteration:
                break
            fut = pool.submit(_worker, arg)
            fut.idx = next_logical_idx  # type: ignore[attr-defined]
            inflight.add(fut)
            next_logical_idx += len(arg) if batch > 1 else 1

        try:
            # Process futures as they complete and add new ones to keep the pool busy
            while inflight:  # Continue until all in-flight tasks are processed
                for fut in as_completed(inflight, timeout=timeout):
                    inflight.remove(fut)
                    idx = fut.idx  # type: ignore[attr-defined]
                    try:
                        res = fut.result()
                    except Exception:
                        if stop_on_error:
                            raise
                        res = None

                    # flatten res to list of logical outputs
                    out_items = res if batch > 1 else [res]

                    # store outputs
                    if ordered and logical_total is not None:
                        results[idx : idx + len(out_items)] = out_items
                    else:
                        results.extend(out_items)

                    completed_items += len(out_items)

                    # progress bar update
                    if bar and completed_items - last_bar_update >= progress_update:
                        bar.update(completed_items - last_bar_update)
                        last_bar_update = completed_items
                        speed = completed_items / max(time.perf_counter() - t0, 1e-6)
                        bar.set_postfix_str(f"{speed:,.0f} items/s")

                    # keep queue full
                    try:
                        while next_logical_idx - completed_items < max_inflight:
                            arg = next(src_iter)
                            fut2 = pool.submit(_worker, arg)
                            fut2.idx = next_logical_idx  # type: ignore[attr-defined]
                            inflight.add(fut2)
                            next_logical_idx += len(arg) if batch > 1 else 1
                    except StopIteration:
                        pass

                    # Break the inner loop as we've processed one future
                    break

                # If we've exhausted the inner loop without processing anything,
                # and there are still in-flight tasks, we need to wait for them
                if inflight and timeout is not None:
                    # Use a small timeout to avoid hanging indefinitely
                    time.sleep(min(0.01, timeout / 10))

        finally:
            if bar:
                bar.update(completed_items - last_bar_update)
                bar.close()

    return results


def multi_threaad_standard(fn, items, workers=4):
    """
    A standard implementation of multi-threading using ThreadPoolExecutor.
    Ensures the order of results matches the input order.
    """
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fn, item) for item in items]
        results = [fut.result() for fut in futures]
    return results


__all__ = ["multi_thread"]
