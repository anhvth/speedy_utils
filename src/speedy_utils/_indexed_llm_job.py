"""Periodic atomic snapshots of one result slot per finite input item."""

import json
import time
from pathlib import Path

from .job_context import atomic_write


def run_indexed(job, items, output, *, resume, checkpoint_every,
                error_log, progress, flush_interval):
    from .parallel_llm_job import (
        JobSummary,
        _ProcessJobExecutor,
        _ThreadJobExecutor,
    )

    started = time.monotonic()
    items = list(items)
    ids = [job.item_id(item) for item in items]
    if len(set(ids)) != len(ids):
        raise ValueError("Indexed inputs must have unique IDs")
    output, state_path, error_path = job._paths(output, error_log)
    resumed = resume and output.exists()
    rows = [json.loads(line) for line in output.read_text().splitlines()] if resumed else []
    if len(rows) > len(items):
        raise ValueError("Existing output has more slots than inputs")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Indexed output requires JSON objects; {} denotes unfinished")
    state = json.loads(state_path.read_text()) if resumed and state_path.exists() else {}
    if state.get("indexed"):
        if ids[:len(state["input_ids"])] != state["input_ids"]:
            raise ValueError("Input order changed; use a new output")
    elif any(rows):
        # Import older compact JSONL by ID, including gaps caused by errors.
        by_id = {str(row["id"]): row for row in rows if row}
        if len(by_id) != sum(bool(row) for row in rows) or not by_id.keys() <= set(ids):
            raise ValueError("Existing row IDs do not match indexed inputs")
        rows = [by_id.get(item_id, {}) for item_id in ids]
    rows.extend({} for _ in range(len(items) - len(rows)))
    job._use_context = False
    job._materialized_root = None
    if state.get("materialization_id"):
        job._materialized_root = (
            job._workspace(output) / "materialized" / state["materialization_id"]
        ).resolve()

    errors = []
    changed = 0
    last_flush = time.monotonic()

    def flush():
        nonlocal changed, last_flush
        atomic_write(output, b"".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"
            for row in rows))
        atomic_write(state_path, json.dumps({
            "indexed": True, "input_ids": ids,
            "written_rows": sum(bool(row) for row in rows),
            **({"materialization_id": state["materialization_id"]}
               if state.get("materialization_id") else {}),
        }).encode())
        if errors:
            error_path.parent.mkdir(parents=True, exist_ok=True)
            with error_path.open("a") as handle:
                for record in errors:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            errors.clear()
        changed = 0
        last_flush = time.monotonic()

    flush()
    missing = iter(i for i, row in enumerate(rows) if not row)
    concurrency = job.processes * job.threads_per_process
    limit = max(1, concurrency * job.prefetch_factor)
    executor = (_ThreadJobExecutor(job, job.threads_per_process)
                if job.processes == 1 else _ProcessJobExecutor(
                    job._job_for_processes(), processes=job.processes,
                    threads_per_process=job.threads_per_process, max_inflight=limit))
    bar = None
    if progress:
        from tqdm import tqdm
        bar = tqdm(total=len(items), initial=sum(bool(row) for row in rows),
                   desc=type(job).__name__, unit="row")
    inflight = failed = rejected = attempted = 0
    exhausted = False
    finished = False
    try:
        while True:
            while not exhausted and inflight < limit:
                index = next(missing, None)
                if index is None:
                    exhausted = True
                    break
                executor.submit(index, items[index])
                inflight += 1
            if not inflight:
                break
            outcome = executor.receive()
            if outcome is not None:
                inflight -= 1
                attempted += 1
                index = outcome.sequence
                if outcome.kind == "result":
                    values = list(job.iter_outputs(items[index], outcome.payload))
                    if len(values) != 1:
                        raise ValueError("Indexed dumping requires one output per item")
                    value = job._json_value(values[0])
                    if not isinstance(value, dict) or not value:
                        raise ValueError("Indexed results must be nonempty JSON objects")
                    rows[index] = value
                    if bar is not None:
                        bar.update(1)
                else:
                    failed += outcome.kind == "error"
                    rejected += outcome.kind == "rejected"
                    if outcome.payload:
                        errors.append({"input_index": index, **outcome.payload})
                changed += 1
            if changed and (changed >= checkpoint_every
                            or time.monotonic() - last_flush >= flush_interval):
                flush()
        finished = True
    finally:
        try:
            flush()
        finally:
            executor.close(cancel=not finished)
            if bar is not None:
                bar.close()
    written = sum(bool(row) for row in rows)
    return JobSummary(
        status="complete" if written == len(items) else "incomplete",
        attempted=attempted, succeeded=written, rejected=rejected, failed=failed,
        written_rows=written, target_rows=len(items),
        elapsed_seconds=time.monotonic() - started, resumed=bool(resumed),
        output_path=output, state_path=state_path, error_path=error_path)
