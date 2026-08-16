"""Persistent, checkpointable parallel jobs backed by ``llm_utils.LLM``."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar, cast


ItemT = TypeVar("ItemT")
OutputT = TypeVar("OutputT")
JobStatus = Literal["complete", "incomplete"]
_STATE_VERSION = 1


@dataclass(frozen=True)
class JobSummary:
    """Final counters and paths for one job run."""

    status: JobStatus
    attempted: int
    succeeded: int
    rejected: int
    failed: int
    written_rows: int
    target_rows: int | None
    elapsed_seconds: float
    resumed: bool
    output_path: Path
    state_path: Path
    error_path: Path

    @property
    def complete(self) -> bool:
        return self.status == "complete"


class TargetNotReachedError(RuntimeError):
    """Raised when an input iterable ends before the requested row target."""

    def __init__(self, summary: JobSummary) -> None:
        self.summary = summary
        super().__init__(
            f"only {summary.written_rows}/{summary.target_rows} output rows completed"
        )


@dataclass(frozen=True)
class _Outcome:
    sequence: int
    kind: Literal["result", "rejected", "error"]
    payload: Any = None


def _error_payload(item_id: str, exc: BaseException) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
    }


def _process_item(
    job: "ParallelLLMJob[Any, Any]", sequence: int, item: Any
) -> _Outcome:
    try:
        result = job.process(item)
    except Exception as exc:
        return _Outcome(sequence, "error", _error_payload(job.item_id(item), exc))
    if result is None:
        return _Outcome(sequence, "rejected")
    return _Outcome(sequence, "result", result)


def _process_worker(
    serialized_job: bytes,
    task_queue: Any,
    result_queue: Any,
    threads_per_process: int,
) -> None:
    """Own one process-local job/LLM and a persistent worker thread pool."""
    import dill

    job = cast(ParallelLLMJob[Any, Any], dill.loads(serialized_job))

    def worker() -> None:
        while True:
            task = task_queue.get()
            if task is None:
                return
            sequence, serialized_item = task
            try:
                item = dill.loads(serialized_item)
                outcome = _process_item(job, sequence, item)
                serialized_outcome = dill.dumps(outcome)
            except Exception as exc:
                fallback = _Outcome(
                    sequence,
                    "error",
                    _error_payload(f"sequence:{sequence}", exc),
                )
                serialized_outcome = dill.dumps(fallback)
            result_queue.put(serialized_outcome)

    threads = [
        threading.Thread(target=worker, name=f"parallel-llm-{index}", daemon=True)
        for index in range(threads_per_process)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


class _ThreadJobExecutor:
    def __init__(self, job: "ParallelLLMJob[Any, Any]", workers: int) -> None:
        self.job = job
        self.pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="parallel-llm",
        )
        self.futures: dict[Future[_Outcome], int] = {}

    def submit(self, sequence: int, item: Any) -> None:
        future = self.pool.submit(_process_item, self.job, sequence, item)
        self.futures[future] = sequence

    def receive(self) -> _Outcome:
        done, _ = wait(self.futures, return_when=FIRST_COMPLETED)
        future = next(iter(done))
        self.futures.pop(future)
        return future.result()

    def close(self, *, cancel: bool) -> None:
        self.pool.shutdown(wait=True, cancel_futures=cancel)


class _ProcessJobExecutor:
    def __init__(
        self,
        job: "ParallelLLMJob[Any, Any]",
        *,
        processes: int,
        threads_per_process: int,
        max_inflight: int,
    ) -> None:
        import dill

        context = mp.get_context("spawn")
        self.task_queue = context.Queue(maxsize=max_inflight)
        self.result_queue = context.Queue(maxsize=max_inflight)
        self.threads_per_process = threads_per_process
        serialized_job = dill.dumps(job)
        self.processes = [
            context.Process(
                target=_process_worker,
                args=(
                    serialized_job,
                    self.task_queue,
                    self.result_queue,
                    threads_per_process,
                ),
                name=f"parallel-llm-process-{index}",
            )
            for index in range(processes)
        ]
        for process in self.processes:
            process.start()

    def submit(self, sequence: int, item: Any) -> None:
        import dill

        self.task_queue.put((sequence, dill.dumps(item)))

    def receive(self) -> _Outcome:
        import dill

        while True:
            try:
                return cast(_Outcome, dill.loads(self.result_queue.get(timeout=0.2)))
            except queue.Empty:
                dead = [
                    process
                    for process in self.processes
                    if process.exitcode is not None
                ]
                if dead:
                    statuses = ", ".join(
                        f"{process.name}={process.exitcode}" for process in dead
                    )
                    raise RuntimeError(
                        f"parallel LLM worker exited early: {statuses}"
                    ) from None

    def close(self, *, cancel: bool) -> None:
        if cancel:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for _ in range(len(self.processes) * self.threads_per_process):
                self.task_queue.put(None)
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        self.task_queue.close()
        self.result_queue.close()


class ParallelLLMJob(Generic[ItemT, OutputT], ABC):
    """Run application-owned LLM logic with library-owned orchestration."""

    def __init__(
        self,
        client: Any,
        model: str | None = None,
        *,
        processes: int = 1,
        threads_per_process: int = 32,
        prefetch_factor: int = 2,
        llm: Any | None = None,
        **llm_defaults: Any,
    ) -> None:
        if processes <= 0:
            raise ValueError("processes must be a positive integer")
        if threads_per_process <= 0:
            raise ValueError("threads_per_process must be a positive integer")
        if prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be a positive integer")
        if llm is not None and processes != 1:
            raise ValueError("an injected LLM can only be used with processes=1")
        self.client = client
        self.model = model
        self.processes = processes
        self.threads_per_process = threads_per_process
        self.prefetch_factor = prefetch_factor
        self.llm_defaults = llm_defaults
        self._llm: Any | None = llm
        self._llm_lock = threading.Lock()

    @property
    def llm(self) -> Any:
        """Return the one lazily-created LLM pool owned by this process."""
        if self._llm is None:
            with self._llm_lock:
                if self._llm is None:
                    from llm_utils import LLM

                    self._llm = LLM(
                        client=self.client,
                        model=self.model,
                        **self.llm_defaults,
                    )
        return self._llm

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_llm"] = None
        state["_llm_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._llm = None
        self._llm_lock = threading.Lock()

    @abstractmethod
    def process(self, item: ItemT) -> OutputT | None:
        """Implement the application-specific work for one input item."""

    def item_id(self, item: ItemT) -> str:
        """Return the stable identifier used in error records."""
        if isinstance(item, Mapping):
            return str(item["id"])
        return str(cast(Any, item).id)

    def iter_outputs(self, item: ItemT, result: OutputT) -> Iterable[Any]:
        """Expand one successful result into JSONL rows."""
        del item
        yield result

    def _job_for_processes(self) -> "ParallelLLMJob[ItemT, OutputT]":
        import copy

        from llm_utils.lm.ssh_tunnel import resolve_ssh_endpoint

        def resolve(value: Any) -> Any:
            if isinstance(value, (int, str)):
                if isinstance(value, str):
                    return resolve_ssh_endpoint(value)
                return value
            raise TypeError(
                "process-mode ParallelLLMJob clients must be ports or endpoint strings"
            )

        worker_job = copy.copy(self)
        if isinstance(self.client, list):
            worker_job.client = [resolve(value) for value in self.client]
        else:
            worker_job.client = resolve(self.client)
        worker_job._llm = None
        worker_job._llm_lock = threading.Lock()
        return worker_job

    @staticmethod
    def _json_value(value: Any) -> Any:
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json")
        return value

    @staticmethod
    def _paths(
        output: str | Path, error_log: str | Path | None
    ) -> tuple[Path, Path, Path]:
        output_path = Path(output)
        state_path = output_path.with_suffix(output_path.suffix + ".state.json")
        error_path = (
            Path(error_log)
            if error_log is not None
            else output_path.with_suffix(output_path.suffix + ".errors.jsonl")
        )
        return output_path, state_path, error_path

    @staticmethod
    def _truncate(path: Path, offset: int) -> None:
        if not path.exists():
            if offset:
                raise RuntimeError(
                    f"state expects {offset} bytes but {path} is missing"
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        with path.open("r+b") as handle:
            if handle.seek(0, os.SEEK_END) < offset:
                raise RuntimeError(
                    f"state expects {offset} bytes but {path} is shorter"
                )
            handle.truncate(offset)

    @staticmethod
    def _write_state(path: Path, state: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(
                json.dumps(state, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)

    @staticmethod
    def _line_count(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("rb") as handle:
            return sum(1 for line in handle if line.strip())

    def _summary(
        self,
        *,
        state: dict[str, Any],
        started: float,
        resumed: bool,
        output_path: Path,
        state_path: Path,
        error_path: Path,
    ) -> JobSummary:
        return JobSummary(
            status=cast(JobStatus, state["status"]),
            attempted=int(state["attempted"]),
            succeeded=int(state["succeeded"]),
            rejected=int(state["rejected"]),
            failed=int(state["failed"]),
            written_rows=int(state["written_rows"]),
            target_rows=state["target_rows"],
            elapsed_seconds=time.monotonic() - started,
            resumed=resumed,
            output_path=output_path,
            state_path=state_path,
            error_path=error_path,
        )

    def run_jsonl(
        self,
        items: Iterable[ItemT],
        output: str | Path,
        *,
        target_rows: int | None = None,
        resume: bool = True,
        checkpoint_every: int = 10_000,
        error_log: str | Path | None = None,
        progress: bool = True,
        progress_total: int | None = None,
    ) -> JobSummary:
        """Run until the iterable ends or exactly ``target_rows`` are committed."""
        if target_rows is not None and target_rows < 0:
            raise ValueError("target_rows must be non-negative")
        if checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be a positive integer")
        if progress_total is not None and progress_total < 0:
            raise ValueError("progress_total must be non-negative")

        started = time.monotonic()
        output_path, state_path, error_path = self._paths(output, error_log)
        for path in (output_path, state_path, error_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        resumed = bool(resume and state_path.exists())
        if resumed:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("version") != _STATE_VERSION:
                raise RuntimeError(f"unsupported ParallelLLMJob state: {state_path}")
            existing_target = state.get("target_rows")
            extending_target = (
                state.get("status") == "complete"
                and isinstance(existing_target, int)
                and isinstance(target_rows, int)
                and target_rows > existing_target
            )
            if existing_target != target_rows and not extending_target:
                raise RuntimeError("target_rows does not match the existing job state")
            self._truncate(output_path, int(state["output_bytes"]))
            self._truncate(error_path, int(state["error_bytes"]))
            if extending_target:
                state["target_rows"] = target_rows
                state["status"] = "incomplete"
                self._write_state(state_path, state)
            if state.get("status") == "complete":
                return self._summary(
                    state=state,
                    started=started,
                    resumed=True,
                    output_path=output_path,
                    state_path=state_path,
                    error_path=error_path,
                )
        else:
            if resume and output_path.exists() and output_path.stat().st_size:
                existing_rows = self._line_count(output_path)
                if target_rows is not None and existing_rows == target_rows:
                    state = {
                        "version": _STATE_VERSION,
                        "status": "complete",
                        "target_rows": target_rows,
                        "next_input_index": existing_rows,
                        "attempted": existing_rows,
                        "succeeded": existing_rows,
                        "rejected": 0,
                        "failed": 0,
                        "written_rows": existing_rows,
                        "output_bytes": output_path.stat().st_size,
                        "error_bytes": error_path.stat().st_size
                        if error_path.exists()
                        else 0,
                    }
                    self._write_state(state_path, state)
                    return self._summary(
                        state=state,
                        started=started,
                        resumed=True,
                        output_path=output_path,
                        state_path=state_path,
                        error_path=error_path,
                    )
                raise RuntimeError(
                    f"cannot resume non-empty {output_path} without {state_path}"
                )
            output_path.write_bytes(b"")
            error_path.write_bytes(b"")
            state_path.unlink(missing_ok=True)
            state = {
                "version": _STATE_VERSION,
                "status": "incomplete",
                "target_rows": target_rows,
                "next_input_index": 0,
                "attempted": 0,
                "succeeded": 0,
                "rejected": 0,
                "failed": 0,
                "written_rows": 0,
                "output_bytes": 0,
                "error_bytes": 0,
            }
            self._write_state(state_path, state)

        if target_rows == 0:
            state["status"] = "complete"
            self._write_state(state_path, state)
            return self._summary(
                state=state,
                started=started,
                resumed=resumed,
                output_path=output_path,
                state_path=state_path,
                error_path=error_path,
            )

        start_index = int(state["next_input_index"])
        source: Iterator[ItemT] = islice(iter(items), start_index, None)
        concurrency = self.processes * self.threads_per_process
        max_inflight = max(1, concurrency * self.prefetch_factor)
        if self.processes == 1:
            executor: Any = _ThreadJobExecutor(self, self.threads_per_process)
        else:
            executor = _ProcessJobExecutor(
                self._job_for_processes(),
                processes=self.processes,
                threads_per_process=self.threads_per_process,
                max_inflight=max_inflight,
            )

        pending_items: dict[int, ItemT] = {}
        completed: dict[int, _Outcome] = {}
        submitted = start_index
        commit_sequence = start_index
        inflight = 0
        source_exhausted = False
        target_reached = False
        run_finished = False
        since_checkpoint = 0

        progress_bar: Any = None
        if progress:
            from tqdm import tqdm

            progress_bar = tqdm(
                total=progress_total if progress_total is not None else target_rows,
                initial=int(
                    state["attempted"]
                    if progress_total is not None
                    else state["written_rows"]
                ),
                desc=type(self).__name__,
                dynamic_ncols=True,
            )

        def commit_state(output_handle: Any, error_handle: Any, complete: bool) -> None:
            nonlocal since_checkpoint
            output_handle.flush()
            error_handle.flush()
            os.fsync(output_handle.fileno())
            os.fsync(error_handle.fileno())
            state["status"] = "complete" if complete else "incomplete"
            state["next_input_index"] = commit_sequence
            state["output_bytes"] = output_handle.tell()
            state["error_bytes"] = error_handle.tell()
            self._write_state(state_path, state)
            since_checkpoint = 0

        try:
            with (
                output_path.open("ab") as output_handle,
                error_path.open("ab") as error_handle,
            ):
                while True:
                    while not source_exhausted and inflight < max_inflight:
                        try:
                            item = next(source)
                        except StopIteration:
                            source_exhausted = True
                            break
                        executor.submit(submitted, item)
                        pending_items[submitted] = item
                        submitted += 1
                        inflight += 1

                    if inflight == 0:
                        break

                    outcome = executor.receive()
                    completed[outcome.sequence] = outcome
                    inflight -= 1

                    while commit_sequence in completed:
                        current = completed.pop(commit_sequence)
                        item = pending_items.pop(commit_sequence)
                        state["attempted"] += 1
                        since_checkpoint += 1

                        if current.kind == "error":
                            state["failed"] += 1
                            record = {
                                "input_index": commit_sequence,
                                **cast(dict[str, Any], current.payload),
                            }
                            error_handle.write(
                                json.dumps(record, ensure_ascii=False).encode("utf-8")
                                + b"\n"
                            )
                        elif current.kind == "rejected":
                            state["rejected"] += 1
                        else:
                            rows = list(
                                self.iter_outputs(item, cast(OutputT, current.payload))
                            )
                            if not rows:
                                state["rejected"] += 1
                            else:
                                state["succeeded"] += 1
                                for row in rows:
                                    if (
                                        target_rows is not None
                                        and state["written_rows"] >= target_rows
                                    ):
                                        break
                                    encoded = json.dumps(
                                        self._json_value(row),
                                        ensure_ascii=False,
                                        separators=(",", ":"),
                                    ).encode("utf-8")
                                    output_handle.write(encoded + b"\n")
                                    state["written_rows"] += 1
                                    if (
                                        progress_bar is not None
                                        and progress_total is None
                                    ):
                                        progress_bar.update(1)

                        if progress_bar is not None and progress_total is not None:
                            progress_bar.update(1)

                        commit_sequence += 1
                        if (
                            target_rows is not None
                            and state["written_rows"] >= target_rows
                        ):
                            target_reached = True
                            break
                        if since_checkpoint >= checkpoint_every:
                            commit_state(output_handle, error_handle, False)

                    if target_reached:
                        break

                complete = target_reached or (target_rows is None and source_exhausted)
                commit_state(output_handle, error_handle, complete)
                run_finished = True
        finally:
            executor.close(cancel=target_reached or not run_finished)
            if progress_bar is not None:
                progress_bar.close()

        summary = self._summary(
            state=state,
            started=started,
            resumed=resumed,
            output_path=output_path,
            state_path=state_path,
            error_path=error_path,
        )
        if not summary.complete:
            raise TargetNotReachedError(summary)
        return summary


__all__ = ["JobSummary", "ParallelLLMJob", "TargetNotReachedError"]
