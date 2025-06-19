import multiprocessing
import os
import traceback
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from typing import Any, TypeVar, cast

T = TypeVar("T")

if hasattr(multiprocessing, "set_start_method"):
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]


# ──── internal helpers ────────────────────────────────────────────────────


def _group_iter(src: Iterable[Any], size: int) -> Iterable[list[Any]]:
    "Yield *size*-sized chunks from *src*."
    it = iter(src)
    while chunk := list(islice(it, size)):
        yield chunk


def _short_tb() -> str:
    tb = "".join(traceback.format_exc())
    return "\n".join(ln for ln in tb.splitlines() if "multi_process" not in ln)


def _safe_call(func: Callable, obj, fixed):
    try:
        return func(obj, **fixed)
    except Exception as exc:
        func_name = getattr(func, "__name__", str(func))
        raise RuntimeError(
            f"{func_name}({obj!r}) failed: {exc}\n{_short_tb()}"
        ) from exc


def _worker_process(
    func: Callable, item_batch: Any, fixed_kwargs: dict, batch_size: int
):
    """Worker function executed in each process."""
    if batch_size > 1:
        results = []
        for itm in item_batch:
            try:
                results.append(_safe_call(func, itm, fixed_kwargs))
            except Exception:
                results.append(None)
        return results
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
    process_update_interval=10,
    for_loop: bool = False,
    **fixed_kwargs,
) -> list[Any]:
    """
    Simple multi‑processing parallel map that returns a *list*.

    Parameters
    ----------
    func          – target callable executed in separate processes, must be of the form f(obj, ...).
    inputs        – iterable with the objects.
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
    if for_loop:
        ret = []
        for arg in inputs:
            ret.append(func(arg, **fixed_kwargs))
        return ret

    if workers is None:
        workers = os.cpu_count() or 1
    if inflight is None:
        inflight = workers * 4
    if batch < 1:
        raise ValueError("batch must be ≥ 1")

    try:
        n_inputs = len(inputs)  # type: ignore[arg-type]
    except Exception:
        n_inputs = None

    src_iter: Iterator[Any] = iter(inputs)
    if batch > 1:
        src_iter = cast(Iterator[Any], _group_iter(src_iter, batch))

    logical_total = n_inputs
    bar = None
    last_bar = 0
    if progress and tqdm is not None and logical_total is not None:
        bar = tqdm(
            total=logical_total,
            ncols=80,
            colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"
            " [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    if ordered and logical_total is not None:
        results: list[Any] = [None] * logical_total
    else:
        results = []

    completed = 0
    next_idx = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = set()

        for _ in range(min(inflight, workers)):
            try:
                arg = next(src_iter)
            except StopIteration:
                break
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
                    num_items = batch if batch > 1 else 1
                    res = [None] * num_items if batch > 1 else None

                out_items = res if batch > 1 else [res]
                if out_items is None:
                    out_items = []

                if ordered and logical_total is not None:
                    if isinstance(out_items, list) and len(out_items) > 0:
                        for i, item in enumerate(out_items):
                            if idx + i < len(results):
                                results[idx + i] = item
                else:
                    if isinstance(out_items, list):
                        results.extend(out_items)

                completed += len(out_items)

                if bar and completed - last_bar >= process_update_interval:
                    bar.update(completed - last_bar)
                    last_bar = completed

                try:
                    while next_idx - completed < inflight:
                        arg = next(src_iter)
                        fut2 = pool.submit(
                            _worker_process, func, arg, fixed_kwargs, batch
                        )
                        fut2.idx = next_idx  # type: ignore[attr-defined]
                        futures.add(fut2)
                        next_idx += len(arg) if batch > 1 else 1
                except StopIteration:
                    pass
                break

    if bar:
        bar.update(completed - last_bar)
        bar.close()

    return results


__all__ = ["multi_process"]
