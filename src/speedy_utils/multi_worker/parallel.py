from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import pickle
import queue
import sys
import shutil
import tempfile
import threading
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None

__all__ = ["parallel", "cleanup_parallel_cache"]

_AUTO_VALUES = {None, "auto", 0}

_INPUT_PAYLOAD = "input.pkl"
_INPUT_META = "input_meta.json"
_OUTPUT_PAYLOAD = "output.pkl"
_OUTPUT_META = "output_meta.json"
_STATUS = "status.json"
_ERROR = "error.txt"
_MANIFEST = "manifest.json"


@dataclass(frozen=True)
class _InputRecord:
    input_hash: str
    indices: tuple[int, ...]
    input_dir: Path
    input_payload_path: Path
    input_meta_path: Path
    output_payload_path: Path
    output_meta_path: Path
    status_path: Path
    error_path: Path


@dataclass(frozen=True)
class _Task:
    input_hash: str
    indices: tuple[int, ...]
    input_dir: str
    input_payload_path: str
    output_payload_path: str
    output_meta_path: str
    status_path: str
    error_path: str


@dataclass(frozen=True)
class _Result:
    input_hash: str
    indices: tuple[int, ...]
    ok: bool
    output_payload_path: Optional[str] = None
    error: Optional[str] = None
    traceback_text: Optional[str] = None
    worker_pid: Optional[int] = None
    worker_thread: Optional[int] = None


@dataclass(frozen=True)
class _ProgressEvent:
    kind: str
    count: int = 0
    message: str = ""
    ok: bool = True
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class _ExecutionPlan:
    num_procs: int
    num_threads: int
    total_worker_slots: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parallel(
    func,
    inputs: Iterable[Any],
    num_procs,
    num_threads,
    progress: bool = True,
) -> list[Any]:
    """
    Execute `func` across `inputs` using processes containing threads.

    Core behavior:
    - cache/output root is under /tmp by default
    - per-function cache directory uses a source-based function signature
    - per-input directory uses a stable hash of the serialized input
    - repeated inputs are deduplicated automatically
    - finished outputs are reused automatically across repeated runs
    - work is split across processes, and each process runs a local thread pool
    - a dedicated reporter thread updates tqdm without blocking scheduling

    `num_procs` and `num_threads` accept positive integers, 0, None, or "auto".
    Auto mode chooses a balanced mixed process/thread plan and also shrinks
    oversubscribed plans when the work queue is small.
    """
    func = _validate_func(func)
    input_list = list(inputs)
    if not input_list:
        return []

    func_name, func_signature = _get_function_identity(func)
    tmp_root = _get_tmp_root()
    func_dir = tmp_root / f"{func_name}_{func_signature}"
    run_id = f"{int(time.time() * 1000)}_{os.getpid()}_{threading.get_ident()}"
    run_dir = func_dir / "runs" / run_id

    _ensure_dir(tmp_root)
    _ensure_dir(func_dir)
    _ensure_dir(run_dir)

    records = _register_inputs(func, func_name, func_signature, func_dir, input_list)
    _write_run_manifest(
        run_dir=run_dir,
        func_name=func_name,
        func_signature=func_signature,
        num_inputs=len(input_list),
        requested_num_procs=num_procs,
        requested_num_threads=num_threads,
        progress=progress,
        records=records,
    )

    cached_records: list[_InputRecord] = []
    pending_records: list[_InputRecord] = []
    for record in records:
        if _is_valid_cached_output(record):
            cached_records.append(record)
        else:
            pending_records.append(record)

    total_inputs = len(input_list)
    plan = _choose_execution_plan(
        total_items=len(pending_records),
        requested_num_procs=num_procs,
        requested_num_threads=num_threads,
    )
    _update_run_plan(run_dir, plan)

    progress_queue = None
    reporter = None
    reporter_stop = None

    try:
        if progress and tqdm is not None:
            ctx = _get_mp_context(func)
            progress_queue = ctx.Queue()
            reporter_stop = threading.Event()
            reporter = threading.Thread(
                target=_report_progress,
                args=(progress_queue, total_inputs, reporter_stop),
                daemon=True,
                name="parallel-progress-reporter",
            )
            reporter.start()
        else:
            progress_queue = None

        for record in cached_records:
            _emit_progress(progress_queue, kind="completed", count=len(record.indices), ok=True, message="cached")

        results: dict[str, _Result] = {}
        for record in cached_records:
            results[record.input_hash] = _Result(
                input_hash=record.input_hash,
                indices=record.indices,
                ok=True,
                output_payload_path=str(record.output_payload_path),
            )

        if pending_records:
            computed = _execute_pending(
                func=func,
                pending_records=pending_records,
                plan=plan,
                progress_queue=progress_queue,
            )
            results.update(computed)

        errors = [result for result in results.values() if not result.ok]
        if errors:
            summary = _format_error_summary(errors)
            raise RuntimeError(summary)

        outputs = _materialize_outputs(input_list, records, results)
        return outputs
    finally:
        if reporter_stop is not None:
            reporter_stop.set()
        if reporter is not None:
            reporter.join(timeout=2.0)


# ---------------------------------------------------------------------------
# Registration / hashing / layout
# ---------------------------------------------------------------------------

def _validate_func(func):
    if not callable(func):
        raise TypeError("func must be callable")
    return func


def _get_tmp_root() -> Path:
    return Path(os.environ.get("PARALLEL_TMP_ROOT", "/tmp/parallel_cache")).expanduser()


def _hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_pickle_dumps(obj: Any) -> bytes:
    return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)


def _hash_input(value: Any) -> str:
    return _hash_bytes(_stable_pickle_dumps(value))


def _get_function_identity(func) -> tuple[str, str]:
    func_name = getattr(func, "__name__", None) or getattr(func, "__qualname__", "anonymous")
    signature_bits: list[Any] = []

    try:
        source = inspect.getsource(func)
    except Exception:
        source = None

    if source is not None:
        signature_bits.append(("source", inspect.cleandoc(source)))
    else:
        code = getattr(func, "__code__", None)
        if code is None:
            signature_bits.append(("repr", repr(func)))
        else:
            signature_bits.append(
                (
                    "code",
                    {
                        "co_code": code.co_code,
                        "co_consts": code.co_consts,
                        "co_names": code.co_names,
                        "co_varnames": code.co_varnames,
                        "co_argcount": code.co_argcount,
                        "co_kwonlyargcount": code.co_kwonlyargcount,
                    },
                )
            )

    signature_bits.append(("module", getattr(func, "__module__", None)))
    signature_bits.append(("qualname", getattr(func, "__qualname__", None)))
    signature_bits.append(("defaults", getattr(func, "__defaults__", None)))
    signature_bits.append(("kwdefaults", getattr(func, "__kwdefaults__", None)))
    signature_bits.append(("annotations", getattr(func, "__annotations__", None)))

    closure = getattr(func, "__closure__", None)
    if closure:
        closure_values = []
        for cell in closure:
            try:
                closure_values.append(cell.cell_contents)
            except Exception:
                closure_values.append(repr(cell))
        signature_bits.append(("closure", closure_values))

    payload = _stable_pickle_dumps(signature_bits)
    return func_name, _hash_bytes(payload)


def _input_dir_for(func_dir: Path, input_hash: str) -> Path:
    return func_dir / input_hash


def _register_inputs(
    func,
    func_name: str,
    func_signature: str,
    func_dir: Path,
    inputs: list[Any],
) -> list[_InputRecord]:
    grouped: dict[str, dict[str, Any]] = {}

    for index, item in enumerate(inputs):
        input_hash = _hash_input(item)
        entry = grouped.get(input_hash)
        if entry is None:
            input_dir = _input_dir_for(func_dir, input_hash)
            input_payload_path = input_dir / _INPUT_PAYLOAD
            input_meta_path = input_dir / _INPUT_META
            output_payload_path = input_dir / _OUTPUT_PAYLOAD
            output_meta_path = input_dir / _OUTPUT_META
            status_path = input_dir / _STATUS
            error_path = input_dir / _ERROR
            entry = {
                "input_hash": input_hash,
                "indices": [],
                "input_dir": input_dir,
                "input_payload_path": input_payload_path,
                "input_meta_path": input_meta_path,
                "output_payload_path": output_payload_path,
                "output_meta_path": output_meta_path,
                "status_path": status_path,
                "error_path": error_path,
                "value": item,
            }
            grouped[input_hash] = entry
        entry["indices"].append(index)

    records: list[_InputRecord] = []
    for input_hash, entry in grouped.items():
        input_dir = entry["input_dir"]
        _ensure_dir(input_dir)

        if not entry["input_payload_path"].exists():
            _atomic_pickle_dump(entry["input_payload_path"], entry["value"])

        if not entry["input_meta_path"].exists():
            _atomic_json_dump(
                entry["input_meta_path"],
                {
                    "func_name": func_name,
                    "func_signature": func_signature,
                    "input_hash": input_hash,
                    "registered_at": time.time(),
                },
            )

        if not entry["status_path"].exists():
            _atomic_json_dump(
                entry["status_path"],
                {
                    "state": "registered",
                    "updated_at": time.time(),
                },
            )

        records.append(
            _InputRecord(
                input_hash=input_hash,
                indices=tuple(entry["indices"]),
                input_dir=input_dir,
                input_payload_path=entry["input_payload_path"],
                input_meta_path=entry["input_meta_path"],
                output_payload_path=entry["output_payload_path"],
                output_meta_path=entry["output_meta_path"],
                status_path=entry["status_path"],
                error_path=entry["error_path"],
            )
        )

    records.sort(key=lambda record: record.indices[0])
    return records


def _is_valid_cached_output(record: _InputRecord) -> bool:
    if not record.output_payload_path.exists():
        return False
    if not record.output_meta_path.exists():
        return False
    try:
        meta = json.loads(record.output_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(meta.get("ok", False))


def _write_run_manifest(
    run_dir: Path,
    func_name: str,
    func_signature: str,
    num_inputs: int,
    requested_num_procs,
    requested_num_threads,
    progress: bool,
    records: list[_InputRecord],
) -> None:
    _atomic_json_dump(
        run_dir / _MANIFEST,
        {
            "func_name": func_name,
            "func_signature": func_signature,
            "num_inputs": num_inputs,
            "requested_num_procs": requested_num_procs,
            "requested_num_threads": requested_num_threads,
            "progress": progress,
            "records": [
                {
                    "input_hash": record.input_hash,
                    "indices": list(record.indices),
                    "input_dir": str(record.input_dir),
                }
                for record in records
            ],
            "created_at": time.time(),
        },
    )


def _update_run_plan(run_dir: Path, plan: _ExecutionPlan) -> None:
    manifest_path = run_dir / _MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    manifest["resolved_num_procs"] = plan.num_procs
    manifest["resolved_num_threads"] = plan.num_threads
    manifest["resolved_total_worker_slots"] = plan.total_worker_slots
    _atomic_json_dump(manifest_path, manifest)


# ---------------------------------------------------------------------------
# Automatic planning
# ---------------------------------------------------------------------------

def _choose_execution_plan(total_items: int, requested_num_procs, requested_num_threads) -> _ExecutionPlan:
    if total_items <= 0:
        return _ExecutionPlan(num_procs=0, num_threads=0, total_worker_slots=0)

    cpu_count = os.cpu_count() or 1
    proc_auto = requested_num_procs in _AUTO_VALUES
    thread_auto = requested_num_threads in _AUTO_VALUES

    if not proc_auto:
        if not isinstance(requested_num_procs, int) or requested_num_procs < 1:
            raise ValueError("num_procs must be a positive integer, 0, None, or 'auto'")
    if not thread_auto:
        if not isinstance(requested_num_threads, int) or requested_num_threads < 1:
            raise ValueError("num_threads must be a positive integer, 0, None, or 'auto'")

    # Mixed workload default: enough total workers to exploit I/O overlap,
    # but balance processes and threads to avoid extreme oversubscription.
    recommended_total_workers = min(total_items, max(1, cpu_count * 4))

    if proc_auto and thread_auto:
        if cpu_count == 1:
            num_procs = 1
            num_threads = min(total_items, 32)
        else:
            num_procs = min(cpu_count, max(1, int(round(math.sqrt(recommended_total_workers)))))
            num_threads = max(1, math.ceil(recommended_total_workers / num_procs))
    elif proc_auto:
        num_threads = requested_num_threads
        num_procs = max(1, min(cpu_count, math.ceil(recommended_total_workers / num_threads)))
    elif thread_auto:
        num_procs = requested_num_procs
        num_threads = max(1, math.ceil(recommended_total_workers / max(1, num_procs)))
    else:
        num_procs = requested_num_procs
        num_threads = requested_num_threads

    num_procs, num_threads = _rebalance_worker_shape(total_items, num_procs, num_threads)
    return _ExecutionPlan(
        num_procs=num_procs,
        num_threads=num_threads,
        total_worker_slots=num_procs * num_threads,
    )


def _rebalance_worker_shape(total_items: int, num_procs: int, num_threads: int) -> tuple[int, int]:
    if total_items <= 0:
        return 0, 0

    num_procs = max(1, int(num_procs))
    num_threads = max(1, int(num_threads))

    total_slots = num_procs * num_threads
    if total_slots <= total_items:
        return num_procs, num_threads

    # Prefer shrinking threads first. Processes are more expensive to start,
    # but keeping some process parallelism still helps CPU-bound work.
    num_threads = max(1, min(num_threads, math.ceil(total_items / num_procs)))
    total_slots = num_procs * num_threads
    if total_slots <= total_items:
        return num_procs, num_threads

    num_procs = max(1, min(num_procs, total_items))
    num_threads = max(1, min(num_threads, math.ceil(total_items / num_procs)))
    return num_procs, num_threads


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _can_pickle_callable(func) -> bool:
    try:
        pickle.dumps(func, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception:
        return False


def _callable_is_reimportable(func) -> bool:
    module_name = getattr(func, "__module__", None)
    if not module_name or module_name == "__main__":
        return False

    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file:
        module_parent = str(Path(module_file).resolve().parent)
        sys_path_dirs = {str(Path(p or os.getcwd()).resolve()) for p in sys.path}
        if module_parent not in sys_path_dirs:
            return False

    try:
        spec = importlib.util.find_spec(module_name)
    except Exception:
        return False
    return spec is not None


def _get_mp_context(func=None):
    import multiprocessing as mp

    if not hasattr(mp, "get_context"):
        return mp

    if os.name == "nt":
        return mp.get_context("spawn")

    if func is not None and _can_pickle_callable(func) and _callable_is_reimportable(func):
        for method in ("forkserver", "spawn"):
            try:
                return mp.get_context(method)
            except Exception:
                continue

    # Fallback keeps support for dynamic or locally-defined callables that
    # cannot be safely re-imported in spawn/forkserver workers.
    for method in ("fork", "forkserver", "spawn"):
        try:
            return mp.get_context(method)
        except Exception:
            continue
    return mp


def _execute_pending(
    func,
    pending_records: list[_InputRecord],
    plan: _ExecutionPlan,
    progress_queue,
) -> dict[str, _Result]:
    if not pending_records:
        return {}

    ctx = _get_mp_context(func)
    result_queue = ctx.Queue()
    proc_queues = [ctx.Queue(maxsize=max(8, plan.num_threads * 4)) for _ in range(plan.num_procs)]

    workers = []
    for proc_id in range(plan.num_procs):
        worker = ctx.Process(
            target=_process_worker_main,
            kwargs={
                "proc_id": proc_id,
                "func": func,
                "num_threads": plan.num_threads,
                "task_queue": proc_queues[proc_id],
                "result_queue": result_queue,
                "progress_queue": progress_queue,
                    "env_snapshot": dict(os.environ),
            },
            daemon=False,
        )
        worker.start()
        workers.append(worker)

    try:
        for task_index, record in enumerate(pending_records):
            queue_index = task_index % plan.num_procs
            proc_queues[queue_index].put(
                _Task(
                    input_hash=record.input_hash,
                    indices=record.indices,
                    input_dir=str(record.input_dir),
                    input_payload_path=str(record.input_payload_path),
                    output_payload_path=str(record.output_payload_path),
                    output_meta_path=str(record.output_meta_path),
                    status_path=str(record.status_path),
                    error_path=str(record.error_path),
                )
            )

        for proc_queue in proc_queues:
            proc_queue.put(None)

        results: dict[str, _Result] = {}
        expected = len(pending_records)
        while len(results) < expected:
            try:
                result: _Result = result_queue.get(timeout=0.5)
            except queue.Empty:
                dead = [worker for worker in workers if worker.exitcode not in (None, 0) and not worker.is_alive()]
                if dead:
                    raise RuntimeError(
                        "Worker process exited unexpectedly: "
                        + ", ".join(f"pid={worker.pid} exitcode={worker.exitcode}" for worker in dead)
                    )
                continue
            results[result.input_hash] = result

        for worker in workers:
            worker.join(timeout=10.0)
            if worker.exitcode not in (0, None):
                raise RuntimeError(f"Worker process exited unexpectedly: pid={worker.pid} exitcode={worker.exitcode}")

        return results
    finally:
        for proc_queue in proc_queues:
            try:
                proc_queue.close()
            except Exception:
                pass
        try:
            result_queue.close()
        except Exception:
            pass
        for worker in workers:
            if worker.is_alive():
                worker.join(timeout=0.2)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)


def _process_worker_main(proc_id: int, func, num_threads: int, task_queue, result_queue, progress_queue, env_snapshot) -> None:
    os.environ.update(env_snapshot)
    _emit_progress(progress_queue, kind="worker-started", count=0, ok=True, message=f"proc={proc_id}")

    max_in_flight = max(1, num_threads * 4)
    done_reading = False
    futures = set()

    with ThreadPoolExecutor(max_workers=num_threads, thread_name_prefix=f"parallel-p{proc_id}") as executor:
        while True:
            while not done_reading and len(futures) < max_in_flight:
                task = task_queue.get()
                if task is None:
                    done_reading = True
                    break
                future = executor.submit(_execute_one_task, func, task, progress_queue)
                futures.add(future)

            if not futures and done_reading:
                break

            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                result_queue.put(future.result())

    _emit_progress(progress_queue, kind="worker-finished", count=0, ok=True, message=f"proc={proc_id}")


def _execute_one_task(func, task: _Task, progress_queue) -> _Result:
    status_path = Path(task.status_path)
    error_path = Path(task.error_path)
    output_payload_path = Path(task.output_payload_path)
    output_meta_path = Path(task.output_meta_path)
    input_payload_path = Path(task.input_payload_path)

    worker_pid = os.getpid()
    worker_thread = threading.get_ident()

    _atomic_json_dump(
        status_path,
        {
            "state": "running",
            "worker_pid": worker_pid,
            "worker_thread": worker_thread,
            "updated_at": time.time(),
        },
    )
    _emit_progress(progress_queue, kind="started", count=0, ok=True, message=task.input_hash)

    try:
        input_value = _load_pickle(input_payload_path)
        output_value = _invoke_func(func, input_value)

        _atomic_pickle_dump(output_payload_path, output_value)
        _atomic_json_dump(
            output_meta_path,
            {
                "ok": True,
                "input_hash": task.input_hash,
                "indices": list(task.indices),
                "worker_pid": worker_pid,
                "worker_thread": worker_thread,
                "finished_at": time.time(),
            },
        )
        _atomic_json_dump(
            status_path,
            {
                "state": "done",
                "updated_at": time.time(),
            },
        )
        if error_path.exists():
            try:
                error_path.unlink()
            except Exception:
                pass
        _emit_progress(progress_queue, kind="completed", count=len(task.indices), ok=True, message=task.input_hash)
        return _Result(
            input_hash=task.input_hash,
            indices=task.indices,
            ok=True,
            output_payload_path=str(output_payload_path),
            worker_pid=worker_pid,
            worker_thread=worker_thread,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        _atomic_text_dump(error_path, tb)
        _atomic_json_dump(
            output_meta_path,
            {
                "ok": False,
                "input_hash": task.input_hash,
                "indices": list(task.indices),
                "error": str(exc),
                "worker_pid": worker_pid,
                "worker_thread": worker_thread,
                "finished_at": time.time(),
            },
        )
        _atomic_json_dump(
            status_path,
            {
                "state": "error",
                "updated_at": time.time(),
            },
        )
        _emit_progress(progress_queue, kind="completed", count=len(task.indices), ok=False, message=task.input_hash)
        return _Result(
            input_hash=task.input_hash,
            indices=task.indices,
            ok=False,
            error=str(exc),
            traceback_text=tb,
            worker_pid=worker_pid,
            worker_thread=worker_thread,
        )


# ---------------------------------------------------------------------------
# Calling convention helpers
# ---------------------------------------------------------------------------

def _invoke_func(func, item: Any) -> Any:
    """
    Adaptive calling convention:
    - dict-like inputs bind as kwargs when possible
    - tuple/list inputs bind as positional args when possible
    - otherwise the input is passed as a single argument

    The binding check keeps the automatic behavior predictable instead of
    blindly unpacking every tuple or mapping.
    """
    try:
        sig = inspect.signature(func)
    except Exception:
        sig = None

    if sig is not None:
        if isinstance(item, Mapping):
            try:
                sig.bind_partial(**item)
            except TypeError:
                pass
            else:
                return func(**item)

        if isinstance(item, tuple):
            try:
                sig.bind_partial(*item)
            except TypeError:
                pass
            else:
                return func(*item)

        if isinstance(item, list):
            try:
                sig.bind_partial(*item)
            except TypeError:
                pass
            else:
                return func(*item)

    return func(item)


# ---------------------------------------------------------------------------
# Result materialization
# ---------------------------------------------------------------------------

def _materialize_outputs(inputs: list[Any], records: list[_InputRecord], results: dict[str, _Result]) -> list[Any]:
    outputs: list[Any] = [None] * len(inputs)
    for record in records:
        result = results.get(record.input_hash)
        if result is None:
            raise RuntimeError(f"Missing result for input_hash={record.input_hash}")
        if not result.ok:
            continue
        output_value = _load_pickle(Path(result.output_payload_path))
        for index in record.indices:
            outputs[index] = output_value
    return outputs


def _format_error_summary(errors: list[_Result]) -> str:
    head = [f"parallel() failed for {len(errors)} unique input(s)"]
    for result in errors[:3]:
        head.append(f"- input_hash={result.input_hash}: {result.error}")
    if len(errors) > 3:
        head.append(f"- ... {len(errors) - 3} more")
    first_tb = errors[0].traceback_text or ""
    return "\n".join(head) + ("\n\n" + first_tb if first_tb else "")


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

def _emit_progress(progress_queue, kind: str, count: int, ok: bool, message: str) -> None:
    if progress_queue is None:
        return
    try:
        progress_queue.put_nowait(_ProgressEvent(kind=kind, count=count, ok=ok, message=message))
    except Exception:
        try:
            progress_queue.put(_ProgressEvent(kind=kind, count=count, ok=ok, message=message), timeout=0.1)
        except Exception:
            pass


def _report_progress(progress_queue, total_inputs: int, stop_event: threading.Event) -> None:
    bar = tqdm(total=total_inputs, desc="parallel", dynamic_ncols=True)
    completed = 0
    successes = 0
    failures = 0

    try:
        while not stop_event.is_set() or completed < total_inputs:
            try:
                event: _ProgressEvent = progress_queue.get(timeout=0.2)
            except queue.Empty:
                if stop_event.is_set() and completed >= total_inputs:
                    break
                continue
            except Exception:
                if stop_event.is_set():
                    break
                continue

            if event.kind == "completed" and event.count:
                completed += event.count
                if event.ok:
                    successes += event.count
                else:
                    failures += event.count
                bar.update(event.count)
                if failures:
                    bar.set_postfix(done=completed, ok=successes, failed=failures)
                else:
                    bar.set_postfix(done=completed, ok=successes)
    finally:
        if completed < total_inputs:
            bar.update(total_inputs - completed)
        bar.close()


# ---------------------------------------------------------------------------
# File IO helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _atomic_pickle_dump(path: Path, value: Any) -> None:
    _atomic_write(path, _stable_pickle_dumps(value), binary=True)


def _atomic_json_dump(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write(path, payload, binary=True)


def _atomic_text_dump(path: Path, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"), binary=True)


def _atomic_write(path: Path, payload: bytes, binary: bool = True) -> None:
    _ensure_dir(path.parent)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        mode = "wb" if binary else "w"
        with os.fdopen(fd, mode) as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_parallel_cache(older_than_seconds: Optional[float] = None) -> None:
    root = _get_tmp_root()
    if not root.exists():
        return

    now = time.time()
    for child in root.iterdir():
        try:
            if older_than_seconds is None:
                shutil.rmtree(child)
            else:
                age = now - child.stat().st_mtime
                if age >= older_than_seconds:
                    shutil.rmtree(child)
        except FileNotFoundError:
            continue

