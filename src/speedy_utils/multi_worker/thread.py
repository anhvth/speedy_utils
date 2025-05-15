"""Provides thread-based parallel execution utilities."""

import inspect
import os
import time
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from typing import Any, TypeVar, cast

from loguru import logger

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

# Sensible defaults
DEFAULT_WORKERS = (os.cpu_count() or 4) * 2

T = TypeVar("T")
R = TypeVar("R")


def _group_iter(src: Iterable[T], size: int) -> Iterable[list[T]]:
    """Yield successive chunks from iterable of specified size."""
    it = iter(src)
    while chunk := list(islice(it, size)):
        yield chunk


def _short_tb() -> str:
    """Return a shortened traceback, excluding internal frames."""
    tb = "".join(traceback.format_exc())
    # hide frames inside this helper to keep logs short
    return "\n".join(ln for ln in tb.splitlines() if "multi_thread.py" not in ln)


def _worker(item: T, func: Callable[[T], R], fixed_kwargs: dict[str, Any]) -> R:
    """Execute the function with an item and fixed kwargs."""
    return func(item, **fixed_kwargs)


# ────────────────────────────────────────────────────────────
# main API
# ────────────────────────────────────────────────────────────
def multi_thread(
    func: Callable,
    inputs: Iterable[Any],
    *,
    workers: int | None = DEFAULT_WORKERS,
    batch: int = 1,
    ordered: bool = True,
    progress: bool = True,
    progress_update: int = 10,
    prefetch_factor: int = 4,
    timeout: float | None = None,
    stop_on_error: bool = True,
    n_proc=0,
    store_output_pkl_file: str | None = None,
    **fixed_kwargs,
) -> list[Any]:
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
    from speedy_utils import dump_json_or_pickle, identify, load_by_ext

    if n_proc > 1:
        import tempfile

        from fastcore.all import threaded

        # split the inputs by nproc
        inputs = list(inputs)
        n_per_proc = max(len(inputs) // n_proc, 1)
        proc_inputs_list = []
        for i in range(0, len(inputs), n_per_proc):
            proc_inputs_list.append(inputs[i : i + n_per_proc])
        procs = []
        in_process_multi_thread = threaded(process=True)(multi_thread)

        for proc_id, proc_inputs in enumerate(proc_inputs_list):
            with tempfile.NamedTemporaryFile(
                delete=False, suffix="multi_thread.pkl"
            ) as tmp_file:
                file_pkl = tmp_file.name
            assert isinstance(in_process_multi_thread, Callable)
            proc = in_process_multi_thread(
                func,
                proc_inputs,
                workers=workers,
                batch=batch,
                ordered=ordered,
                progress=proc_id == 0,
                progress_update=progress_update,
                prefetch_factor=prefetch_factor,
                timeout=timeout,
                stop_on_error=stop_on_error,
                n_proc=0,  # prevent recursion
                store_output_pkl_file=file_pkl,
                **fixed_kwargs,
            )
            procs.append([proc, file_pkl])
        # join
        results = []

        for proc, file_pkl in procs:
            proc.join()
            logger.info(f"Done proc {proc=}")
            results.extend(load_by_ext(file_pkl))
        return results

    try:
        import pandas as pd

        if isinstance(inputs, pd.DataFrame):
            inputs = inputs.to_dict(orient="records")
    except ImportError:
        pass

    try:
        n_inputs = len(inputs)  # type: ignore[arg-type]
    except Exception:
        n_inputs = None
    workers_val = workers if workers is not None else DEFAULT_WORKERS

    if batch == 1 and n_inputs and n_inputs / max(workers_val, 1) > 20_000:
        batch = 32  # empirically good for sub‑ms tasks

    # ── build (maybe‑batched) source iterator ────────────────────────────
    src_iter: Iterable[Any] = iter(inputs)
    if batch > 1:
        src_iter = _group_iter(src_iter, batch)
    # Ensure src_iter is always an iterator
    src_iter = iter(src_iter)

    # total logical items (for bar & ordered pre‑allocation)
    logical_total = n_inputs
    if logical_total is not None and batch > 1:
        logical_total = n_inputs  # still number of *items*, not batches

    # ── progress bar ─────────────────────────────────────────────────────
    bar = None
    last_bar_update = 0
    if progress and tqdm is not None and logical_total is not None:
        bar = tqdm(
            total=logical_total,
            ncols=128,
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
            " [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    # ── prepare result container ─────────────────────────────────────────
    if ordered and logical_total is not None:
        results: list[Any] = [None] * logical_total
    else:
        results = []

    # ── main execution loop ──────────────────────────────────────────────────
    workers_val = workers if workers is not None else DEFAULT_WORKERS
    max_inflight = workers_val * max(prefetch_factor, 1)
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
            if batch > 1:
                fut = pool.submit(
                    lambda items: [_worker(item, func, fixed_kwargs) for item in items],
                    arg,
                )
                fut.idx = next_logical_idx  # type: ignore[attr-defined]
                inflight.add(fut)
                next_logical_idx += len(arg)
            else:
                fut = pool.submit(_worker, arg, func, fixed_kwargs)
                fut.idx = next_logical_idx  # type: ignore[attr-defined]
                inflight.add(fut)
                next_logical_idx += 1

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

                    # Ensure out_items is a list (and thus Sized)
                    if out_items is None:
                        out_items = [None]
                    elif not isinstance(out_items, list):
                        out_items = (
                            list(out_items)
                            if isinstance(out_items, Iterable)
                            else [out_items]
                        )

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
                        # Show pending, submitted, processing in the bar postfix
                        submitted = next_logical_idx
                        processing = min(len(inflight), workers_val)
                        pending = (
                            (logical_total - submitted)
                            if logical_total is not None
                            else None
                        )
                        postfix = {
                            "pending": pending if pending is not None else "-",
                            # 'submitted': submitted,
                            "processing": processing,
                        }
                        bar.set_postfix(postfix)

                    # keep queue full
                    try:
                        while next_logical_idx - completed_items < max_inflight:
                            arg = next(src_iter)
                            if batch > 1:
                                fut2 = pool.submit(
                                    lambda items: [
                                        _worker(item, func, fixed_kwargs)
                                        for item in items
                                    ],
                                    arg,
                                )
                                fut2.idx = next_logical_idx  # type: ignore[attr-defined]
                                inflight.add(fut2)
                                next_logical_idx += len(arg)
                            else:
                                fut2 = pool.submit(_worker, arg, func, fixed_kwargs)
                                fut2.idx = next_logical_idx  # type: ignore[attr-defined]
                                inflight.add(fut2)
                                next_logical_idx += 1
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
    if store_output_pkl_file:
        dump_json_or_pickle(results, store_output_pkl_file)
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


__all__ = ["multi_thread", "multi_threaad_standard"]
