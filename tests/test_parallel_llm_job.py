import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import BaseModel

import speedy_utils
from speedy_utils import JobSummary, ParallelLLMJob, TargetNotReachedError


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class _Output(BaseModel):
    id: str
    value: int


class _BasicJob(ParallelLLMJob[dict, _Output]):
    def process(self, item):
        if item.get("error"):
            raise ValueError("bad item")
        if item.get("reject"):
            return None
        time.sleep(item.get("delay", 0))
        return _Output(id=str(item["id"]), value=item["value"])


class _FanoutJob(ParallelLLMJob[dict, list[dict]]):
    def process(self, item):
        return [
            {"id": f"{item['id']}-a"},
            {"id": f"{item['id']}-b"},
        ]

    def iter_outputs(self, item, result):
        del item
        return result


class _PidJob(ParallelLLMJob[dict, dict]):
    def process(self, item):
        time.sleep(0.01)
        return {"id": item["id"], "pid": os.getpid()}


class _BufferedJob(ParallelLLMJob[dict, dict]):
    def process(self, item):
        if item["id"]:
            threading.Event().wait()
        return {"id": item["id"]}


def test_process_target_exits_with_unconsumed_prefetched_tasks(tmp_path):
    script = """
from test_parallel_llm_job import _BufferedJob
if __name__ == '__main__':
    job = _BufferedJob(1, processes=2, threads_per_process=1,
                       prefetch_factor=4)
    result = job.run_jsonl(
        ({'id': i, 'payload': b'x' * 1_000_000} for i in range(8)),
        'rows.jsonl', target_rows=1, progress=False,
    )
    assert result.complete
    print('completed', flush=True)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).parent), str(Path(speedy_utils.__file__).parent.parent)]
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "completed"
    assert _rows(tmp_path / "rows.jsonl") == [{"_input_index": 0, "id": 0}]


class _StopRun(BaseException):
    pass


def test_default_writes_later_result_before_first_input_finishes(tmp_path):
    output = tmp_path / "arrival.jsonl"
    release_first = threading.Event()

    class Job(ParallelLLMJob):
        def process(self, item):
            if item["id"] == 0:
                assert release_first.wait(5)
            return item

    def inputs():
        yield {"id": 0}
        yield {"id": 1}
        try:
            # Refill occurs only after the coordinator receives input 1.
            assert _rows(output) == [{"_input_index": 1, "id": 1}]
        finally:
            release_first.set()

    summary = Job("unused", threads_per_process=2, prefetch_factor=1).run_jsonl(
        inputs(), output, progress=False
    )
    assert summary.complete
    assert _rows(output) == [
        {"_input_index": 1, "id": 1},
        {"_input_index": 0, "id": 0},
    ]


def test_unordered_resume_skips_committed_later_input_and_truncates_tails(tmp_path):
    output = tmp_path / "resume-arrival.jsonl"
    release_first = threading.Event()
    calls = []

    class Job(ParallelLLMJob):
        def process(self, item):
            calls.append(item["id"])
            if item["id"] == 0 and calls.count(0) == 1:
                assert release_first.wait(5)
                raise _StopRun()
            return item

    def inputs():
        yield {"id": 0}
        yield {"id": 1}
        release_first.set()

    job = Job("unused", threads_per_process=2, prefetch_factor=1)
    with pytest.raises(_StopRun):
        job.run_jsonl(inputs(), output, checkpoint_every=1, progress=False)
    assert _rows(output) == [{"_input_index": 1, "id": 1}]
    with output.open("ab") as handle:
        handle.write(b'{"incomplete":')
    journal = job._workspace(output) / "completed.jsonl"
    with journal.open("ab") as handle:
        handle.write(b"0\n")
    summary = job.run_jsonl(
        [{"id": 0}, {"id": 1}], output, checkpoint_every=1, progress=False
    )
    assert summary.resumed
    assert _rows(output) == [
        {"_input_index": 1, "id": 1},
        {"_input_index": 0, "id": 0},
    ]
    assert calls.count(1) == 1
    assert calls.count(0) == 2


@pytest.mark.parametrize("value", [{"text": "hello"}, "hello", [1, 2], 42])
def test_unordered_tags_output_without_mutating_result(tmp_path, value):
    class Job(ParallelLLMJob):
        def process(self, item):
            return value

    output = tmp_path / "tagged.jsonl"
    Job("unused").run_jsonl([{"id": "source"}], output, ordered=False, progress=False)
    expected = value if isinstance(value, dict) else {"output": value}
    assert _rows(output) == [{"_input_index": 0, **expected}]
    if isinstance(value, dict):
        assert "_input_index" not in value


def test_unordered_rejects_reserved_input_index(tmp_path):
    output = tmp_path / "collision.jsonl"
    with pytest.raises(ValueError, match="_input_index is reserved"):
        _InterruptingJob("unused").run_jsonl(
            [{"id": 1, "_input_index": 99}], output, progress=False
        )


@pytest.mark.parametrize("processes", [1, 2])
def test_unordered_fanout_tags_rows_and_stops_at_exact_target(tmp_path, processes):
    output = tmp_path / "fanout-arrival.jsonl"
    summary = _FanoutJob(
        1, processes=processes, threads_per_process=2
    ).run_jsonl(
        [{"id": i} for i in range(10)], output, target_rows=5, progress=False
    )
    rows = _rows(output)
    assert summary.written_rows == len(rows) == 5
    assert all(row["id"].split("-")[0] == str(row["_input_index"]) for row in rows)
    assert rows[0]["_input_index"] == rows[1]["_input_index"]
    assert rows[2]["_input_index"] == rows[3]["_input_index"]
    assert len({row["_input_index"] for row in rows}) == 3


def test_unordered_resume_retains_errors_and_rejections_without_retry(tmp_path):
    class Job(_BasicJob):
        def process(self, item):
            if item.get("interrupt"):
                raise _StopRun()
            return super().process(item)

    output = tmp_path / "terminal.jsonl"
    items = [
        {"id": "bad", "error": True},
        {"id": "reject", "reject": True},
        {"id": "ok", "value": 2},
        {"id": "stop", "value": 3, "interrupt": True},
    ]
    job = Job("unused", threads_per_process=1, prefetch_factor=1)
    with pytest.raises(_StopRun):
        job.run_jsonl(items, output, checkpoint_every=1, progress=False)
    items[-1].pop("interrupt")
    summary = job.run_jsonl(items, output, progress=False)
    assert (summary.attempted, summary.failed, summary.rejected) == (4, 1, 1)
    assert _rows(output) == [
        {"_input_index": 2, "id": "ok", "value": 2},
        {"_input_index": 3, "id": "stop", "value": 3},
    ]
    assert len(_rows(summary.error_path)) == 1


def test_unordered_checkpoints_by_elapsed_time_before_count_limit(tmp_path, monkeypatch):
    from types import SimpleNamespace

    clock = [0.0]
    emitted = []
    monkeypatch.setattr(
        speedy_utils.parallel_llm_job, "time",
        SimpleNamespace(monotonic=lambda: clock[0], time_ns=time.time_ns),
    )

    class Job(_InterruptingJob):
        def process(self, item):
            clock[0] = 11.0
            return super().process(item)

        def iter_outputs(self, item, result):
            emitted.append(item["id"])
            yield result

    output = tmp_path / "timed.jsonl"
    items = [{"id": 0}, {"id": 1}]
    job = Job("unused", threads_per_process=1, prefetch_factor=1, stop_at=1)
    with pytest.raises(_StopRun):
        job.run_jsonl(
            items, output, checkpoint_every=100, flush_interval=10, progress=False
        )
    job.stop_at = None
    summary = job.run_jsonl(items, output, progress=False)
    assert summary.written_rows == 2
    assert emitted == [0, 1]  # Input 0 was checkpointed, not re-emitted on resume.


def test_unordered_target_extension_keeps_original_indices(tmp_path):
    output = tmp_path / "extend-arrival.jsonl"
    job = _BasicJob("unused", threads_per_process=2)
    items = [{"id": str(i), "value": i} for i in range(8)]
    job.run_jsonl(items, output, target_rows=2, progress=False)
    original = _rows(output)
    summary = job.run_jsonl(items, output, target_rows=6, progress=False)
    rows = _rows(output)
    assert summary.written_rows == len(rows) == 6
    assert rows[:2] == original
    assert len({row["_input_index"] for row in rows}) == 6
    assert all(row["value"] == row["_input_index"] for row in rows)


@pytest.mark.parametrize("ordered", [False, True])
def test_resume_refuses_explicit_ordering_change(tmp_path, ordered):
    output = tmp_path / "mode.jsonl"
    job = _BasicJob("unused")
    job.run_jsonl([], output, ordered=ordered, progress=False)
    with pytest.raises(ValueError, match="ordered does not match"):
        job.run_jsonl([], output, ordered=not ordered, progress=False)


@pytest.mark.parametrize("progress_total", [None, 4])
def test_progress_counts_saved_rows_or_completed_inputs(tmp_path, monkeypatch, progress_total):
    import io
    import tqdm

    bars = []
    stream = io.StringIO()

    class RecordingBar(tqdm.tqdm):
        def __init__(self, **kwargs):
            super().__init__(**kwargs, file=stream)
            bars.append(self)

    monkeypatch.setattr(tqdm, "tqdm", RecordingBar)
    summary = _BasicJob("unused", threads_per_process=1).run_jsonl(
        [
            {"id": "a", "value": 1},
            {"id": "b", "error": True},
            {"id": "c", "reject": True},
            {"id": "d", "value": 2},
        ],
        tmp_path / "progress.jsonl",
        progress_total=progress_total,
    )
    bar = bars[0]
    assert bar.n == (summary.attempted if progress_total else summary.written_rows)
    assert bar.unit == ("item" if progress_total else "row")
    assert "pending=0" in bar.postfix
    assert "errors=1" in bar.postfix
    assert "rejected=1" in bar.postfix
    assert "buffered" not in stream.getvalue()
    assert "waiting_for_order" not in stream.getvalue()
    if progress_total:
        assert "saved=2" in bar.postfix
        assert "100%" in stream.getvalue()


class _InterruptingJob(ParallelLLMJob[dict, dict]):
    def __init__(self, *args, stop_at=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_at = stop_at

    def process(self, item):
        if item["id"] == self.stop_at:
            raise _StopRun()
        return item


def test_public_exports_are_direct_and_lightweight():
    assert speedy_utils.ParallelLLMJob is ParallelLLMJob
    assert speedy_utils.JobSummary is JobSummary
    assert "__getattr__" not in speedy_utils.__dict__
    assert "llm_utils" not in vars(speedy_utils.parallel_llm_job)


def test_default_job_sidecars_live_in_temporary_workspace(tmp_path):
    output = tmp_path / "rows.jsonl"

    summary = _BasicJob("unused").run_jsonl(
        [{"id": "one", "value": 1, "error": True}], output, progress=False
    )

    workspace = Path("/tmp/parallel_llm").resolve()
    assert summary.state_path.is_relative_to(workspace)
    assert summary.error_path.is_relative_to(workspace)
    assert summary.state_path.exists()
    assert summary.error_path.exists()
    assert list(tmp_path.iterdir()) == [output]


def test_resume_migrates_legacy_sidecars_to_job_workspace(tmp_path):
    output = tmp_path / "rows.jsonl"
    output.write_text('{"id":"one"}\n')
    legacy_state = output.with_suffix(output.suffix + ".state.json")
    legacy_state.write_text(json.dumps({
        "version": 1,
        "status": "complete",
        "target_rows": 1,
        "next_input_index": 1,
        "attempted": 1,
        "succeeded": 1,
        "rejected": 0,
        "failed": 0,
        "written_rows": 1,
        "output_bytes": output.stat().st_size,
        "error_bytes": 0,
    }))
    legacy_errors = output.with_suffix(output.suffix + ".errors.jsonl")
    legacy_errors.write_text("")

    summary = _BasicJob("unused").run_jsonl(
        [], output, target_rows=1, progress=False
    )

    assert summary.resumed
    assert summary.state_path.exists()
    assert not legacy_state.exists()
    assert not legacy_errors.exists()
    assert list(tmp_path.iterdir()) == [output]


def test_thread_job_is_ordered_and_replaces_rejections_and_errors(tmp_path):
    output = tmp_path / "rows.jsonl"
    items = [
        {"id": "slow", "value": 0, "delay": 0.03},
        {"id": "rejected", "value": 1, "reject": True},
        {"id": "failed", "value": 2, "error": True},
        {"id": "three", "value": 3},
        {"id": "four", "value": 4},
        {"id": "five", "value": 5},
    ]

    summary = _BasicJob("unused", threads_per_process=3).run_jsonl(
        items,
        output,
        target_rows=3,
        checkpoint_every=1,
        progress=False,
        ordered=True,
    )

    assert summary.complete
    assert (summary.attempted, summary.succeeded, summary.rejected, summary.failed) == (
        5,
        3,
        1,
        1,
    )
    assert [row["id"] for row in _rows(output)] == ["slow", "three", "four"]
    errors = _rows(summary.error_path)
    assert errors[0]["item_id"] == "failed"
    assert errors[0]["error_type"] == "ValueError"


def test_slow_first_row_does_not_block_worker_refill(tmp_path):
    reached_next_window = threading.Event()

    class RefillJob(ParallelLLMJob):
        def process(self, item):
            if item["id"] == 0:
                assert reached_next_window.wait(5), "Workers stopped at the first window"
            if item["id"] == 8:
                reached_next_window.set()
            return item

    output = tmp_path / "refill.jsonl"
    summary = RefillJob("unused", threads_per_process=4, prefetch_factor=2).run_jsonl(
        ({"id": i} for i in range(16)), output, progress=False, ordered=True,
    )
    assert summary.failed == 0
    assert summary.written_rows == 16
    assert [row["id"] for row in _rows(output)] == list(range(16))


def test_process_job_uses_persistent_children_and_keeps_order(tmp_path):
    output = tmp_path / "process.jsonl"
    items = [{"id": index} for index in range(30)]

    summary = _PidJob(
        1,
        processes=2,
        threads_per_process=2,
    ).run_jsonl(items, output, progress=False, ordered=True)

    rows = _rows(output)
    assert summary.written_rows == 30
    assert [row["id"] for row in rows] == list(range(30))
    assert 1 <= len({row["pid"] for row in rows}) <= 2
    assert all(row["pid"] != os.getpid() for row in rows)


def test_fanout_stops_at_exact_target(tmp_path):
    output = tmp_path / "fanout.jsonl"
    summary = _FanoutJob("unused", threads_per_process=2).run_jsonl(
        ({"id": index} for index in range(20)),
        output,
        target_rows=5,
        progress=False,
        ordered=True,
    )

    assert summary.written_rows == 5
    assert [row["id"] for row in _rows(output)] == ["0-a", "0-b", "1-a", "1-b", "2-a"]


def test_resume_replays_only_after_last_atomic_checkpoint(tmp_path):
    output = tmp_path / "resume.jsonl"
    items = [{"id": index} for index in range(4)]
    with pytest.raises(_StopRun):
        _InterruptingJob("unused", threads_per_process=1, stop_at=1).run_jsonl(
            items,
            output,
            target_rows=3,
            checkpoint_every=1,
            progress=False,
            ordered=True,
        )

    summary = _InterruptingJob("unused", threads_per_process=1).run_jsonl(
        items,
        output,
        target_rows=3,
        checkpoint_every=1,
        resume=True,
        progress=False,
    )

    assert summary.resumed
    assert [row["id"] for row in _rows(output)] == [0, 1, 2]


def test_completed_job_can_extend_its_exact_target(tmp_path):
    output = tmp_path / "extend.jsonl"
    job = _BasicJob("unused", threads_per_process=2)
    items = ({"id": str(index), "value": index} for index in range(10))
    first = job.run_jsonl(items, output, target_rows=2, progress=False, ordered=True)
    assert first.written_rows == 2

    extended_items = ({"id": str(index), "value": index} for index in range(10))
    second = job.run_jsonl(
        extended_items,
        output,
        target_rows=5,
        resume=True,
        progress=False,
    )

    assert second.resumed
    assert second.written_rows == 5
    assert [row["value"] for row in _rows(output)] == [0, 1, 2, 3, 4]


def test_resume_refuses_nonempty_output_without_state(tmp_path):
    output = tmp_path / "orphan.jsonl"
    output.write_text('{"id":"existing"}\n')
    with pytest.raises(RuntimeError, match="without"):
        _BasicJob("unused").run_jsonl(
            [], output, target_rows=2, resume=True, progress=False
        )


def test_exhausted_source_raises_with_partial_summary(tmp_path):
    output = tmp_path / "short.jsonl"
    with pytest.raises(TargetNotReachedError) as caught:
        _BasicJob("unused").run_jsonl(
            [{"id": "one", "value": 1}],
            output,
            target_rows=2,
            progress=False,
        )

    assert caught.value.summary.status == "incomplete"
    assert caught.value.summary.written_rows == 1


def test_llm_is_constructed_once_under_thread_contention(monkeypatch, tmp_path):
    created = []

    class FakeLLM:
        def __init__(self, client, **kwargs):
            time.sleep(0.01)
            created.append((client, kwargs))

    class LLMJob(ParallelLLMJob[dict, dict]):
        def process(self, item):
            return {"id": item["id"], "client": id(self.llm)}

    monkeypatch.setattr("llm_utils.LLM", FakeLLM)
    output = tmp_path / "llm.jsonl"
    LLMJob("fake", threads_per_process=8).run_jsonl(
        ({"id": index} for index in range(32)), output, progress=False
    )

    assert len(created) == 1
    assert len({row["client"] for row in _rows(output)}) == 1


def test_injected_llm_is_public_and_restricted_to_thread_mode(tmp_path):
    injected = object()

    class LLMJob(ParallelLLMJob[dict, dict]):
        def process(self, item):
            return {"id": item["id"], "injected": self.llm is injected}

    output = tmp_path / "injected.jsonl"
    LLMJob("unused", llm=injected).run_jsonl(
        [{"id": "one"}], output, progress=False, progress_total=1
    )
    assert _rows(output) == [{"_input_index": 0, "id": "one", "injected": True}]

    with pytest.raises(ValueError, match="processes=1"):
        LLMJob("unused", llm=injected, processes=2)


def test_resume_reuses_materialized_result_behind_interrupted_first_item(tmp_path):
    completed = threading.Event()
    calls = []

    class Job(ParallelLLMJob):
        def process(self, item):
            calls.append(item["id"])
            if item["id"] == 0 and calls.count(0) == 1:
                assert completed.wait(5)
                raise _StopRun()
            completed.set()
            return item

    output = tmp_path / "rows.jsonl"
    items = [{"id": 0}, {"id": 1}]
    with pytest.raises(_StopRun):
        Job("unused", threads_per_process=2).run_jsonl(
            items, output, checkpoint_every=100, progress=False, ordered=True)
    assert output.read_text() == ""
    summary = Job("unused", threads_per_process=2).run_jsonl(
        items, output, checkpoint_every=100, progress=False)
    assert summary.resumed
    assert _rows(output) == items
    assert calls.count(0) == 2
    assert calls.count(1) == 1


def test_indexed_flush_saves_later_row_before_first_and_resumes(tmp_path):
    output = tmp_path / "indexed.jsonl"
    calls = []

    class Job(ParallelLLMJob):
        def process(self, item):
            calls.append(item["id"])
            if item["id"] == 0:
                deadline = time.monotonic() + 5
                while _rows(output)[1] == {}:
                    assert time.monotonic() < deadline
                    time.sleep(0.01)
                raise _StopRun()
            return item

    items = [{"id": 0}, {"id": 1}]
    with pytest.raises(_StopRun):
        Job("unused", threads_per_process=2).run_jsonl(
            items, output, indexed=True, checkpoint_every=1, progress=False)
    assert _rows(output) == [{}, {"id": 1}]

    class ResumeJob(ParallelLLMJob):
        def process(self, item):
            calls.append(item["id"])
            return item

    summary = ResumeJob("unused").run_jsonl(
        [*items, {"id": 2}], output, indexed=True, progress=False)
    assert summary.complete
    assert _rows(output) == [*items, {"id": 2}]
    assert calls.count(1) == 1
