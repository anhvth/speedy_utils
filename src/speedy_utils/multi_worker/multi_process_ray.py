"""Small ordered ``f(item)`` mapper backed by an existing Ray cluster."""

from __future__ import annotations

import contextlib
import importlib
from numbers import Integral
from typing import Any, Callable, Iterable, Mapping

from tqdm import tqdm

from .common import ErrorHandlerType, ErrorStats


def _call_importable(
    module_name: str, qualname: str, item: Any, func_kwargs: Mapping[str, Any]
) -> Any:
    target: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target(item, **func_kwargs)


def _importable_reference(func: Callable[[Any], Any]) -> tuple[str, str] | None:
    module_name = getattr(func, "__module__", "")
    qualname = getattr(func, "__qualname__", "")
    if not module_name or module_name == "__main__" or not qualname or "<locals>" in qualname:
        return None
    return module_name, qualname


def _require_importable(func: Callable[[Any], Any]) -> tuple[str, str]:
    """Return a stable worker import path instead of cloudpickling project code."""
    if not callable(func):
        raise TypeError("func must be callable")
    reference = _importable_reference(func)
    if reference is None:
        raise TypeError(
            "multi_process_ray requires a top-level function from an importable "
            "module; move lambdas, nested functions, and __main__ functions into "
            "a module."
        )
    return reference


def _default_max_in_flight(ray: Any, num_cpus: float, num_gpus: float) -> int:
    """Bound queued tasks by the resource they actually consume."""
    resources = ray.cluster_resources()
    limits: list[int] = []
    if num_cpus > 0:
        limits.append(max(1, int(float(resources.get("CPU", 1.0)) / num_cpus) * 4))
    if num_gpus > 0:
        limits.append(max(1, int(float(resources.get("GPU", 0.0)) / num_gpus) * 2))
    return min(limits) if limits else 1


def _progress_delta(
    item: Any,
    result: Any,
    progress_increment: Callable[[Any, Any], int] | None,
) -> int:
    delta = progress_increment(item, result) if progress_increment else 1
    if isinstance(delta, bool) or not isinstance(delta, Integral) or delta < 0:
        raise ValueError("progress_increment must return a non-negative integer")
    return int(delta)


def multi_process_ray(
    func: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    address: str = "auto",
    num_cpus: float = 1,
    num_gpus: float = 0,
    max_in_flight: int | None = None,
    progress: bool = True,
    desc: str | None = None,
    error_handler: ErrorHandlerType = "log",
    max_error_files: int = 100,
    runtime_env: Mapping[str, Any] | None = None,
    progress_total: int | None = None,
    progress_increment: Callable[[Any, Any], int] | None = None,
    **func_kwargs: Any,
) -> list[Any]:
    """Run ``func(item, **func_kwargs)`` on an existing Ray cluster in input order.

    ``func`` must be a top-level function in an importable module. This avoids
    Ray cloudpickling the caller's project or creating an implicit runtime
    environment. This task mapper is for independent, stateless work; GPU
    models that should remain loaded between inputs need a Ray actor pool.

    The helper only connects to an existing cluster. Start the local topology
    first with ``./ray_gpu_topology.sh up``.
    """

    if num_cpus < 0 or num_gpus < 0:
        raise ValueError("num_cpus and num_gpus must be non-negative")
    if error_handler not in {"raise", "ignore", "log"}:
        raise ValueError(f"unsupported error_handler: {error_handler!r}")
    importable = _require_importable(func)
    values = list(items)
    if not values:
        return []

    try:
        import ray
    except ImportError as error:  # pragma: no cover - environment boundary
        raise RuntimeError("multi_process_ray requires the optional 'ray' package") from error
    if not ray.is_initialized():
        try:
            ray.init(address=address)
        except ConnectionError as error:
            raise RuntimeError(
                "Ray is unavailable; run ./ray_gpu_topology.sh up before "
                "calling multi_process_ray."
            ) from error

    if max_in_flight is None:
        max_in_flight = _default_max_in_flight(ray, num_cpus, num_gpus)
    if max_in_flight <= 0:
        raise ValueError("max_in_flight must be positive")

    call = ray.remote(num_cpus=num_cpus, num_gpus=num_gpus, runtime_env=runtime_env)(
        _call_importable
    )

    def submit(item: Any) -> Any:
        return call.remote(importable[0], importable[1], item, func_kwargs)

    results: list[Any] = [None] * len(values)
    pending: dict[Any, int] = {}
    next_index = 0
    func_name = getattr(func, "__name__", "ray_task")
    errors = ErrorStats(
        func_name,
        max_error_files=max_error_files,
        write_logs=error_handler == "log",
    )

    def submit_until_full() -> None:
        nonlocal next_index
        while next_index < len(values) and len(pending) < max_in_flight:
            ref = submit(values[next_index])
            pending[ref] = next_index
            next_index += 1

    submit_until_full()
    if progress_total is not None and progress_total <= 0:
        raise ValueError("progress_total must be positive when provided")
    try:
        with tqdm(
            total=progress_total if progress_total is not None else len(values),
            disable=not progress,
            desc=desc or "Ray map",
        ) as bar:
            while pending:
                ready, _remaining = ray.wait(list(pending), num_returns=1)
                ref = ready[0]
                index = pending.pop(ref)
                try:
                    results[index] = ray.get(ref)
                    errors.record_success()
                except Exception as error:  # noqa: BLE001 - match multi_process contract.
                    if error_handler == "raise":
                        raise
                    errors.record_error(index, error, values[index], func_name)
                bar.update(_progress_delta(values[index], results[index], progress_increment))
                submit_until_full()
    except BaseException:
        cancel = getattr(ray, "cancel", None)
        if cancel is not None:
            for ref in pending:
                with contextlib.suppress(Exception):  # pragma: no cover - cleanup only.
                    cancel(ref, force=True)
        raise
    return results


__all__ = ["multi_process_ray"]
