import json
import os
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


class _StopRun(BaseException):
    pass


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


def test_process_job_uses_persistent_children_and_keeps_order(tmp_path):
    output = tmp_path / "process.jsonl"
    items = [{"id": index} for index in range(30)]

    summary = _PidJob(
        1,
        processes=2,
        threads_per_process=2,
    ).run_jsonl(items, output, progress=False)

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
    first = job.run_jsonl(items, output, target_rows=2, progress=False)
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
    assert _rows(output) == [{"id": "one", "injected": True}]

    with pytest.raises(ValueError, match="processes=1"):
        LLMJob("unused", llm=injected, processes=2)
