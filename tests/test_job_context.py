import json

import pytest

from speedy_utils import ParallelLLMJob
from speedy_utils.job_context import JobContext


def test_failed_stage_reuses_upstream_and_retries_only_selected_errors(tmp_path):
    calls = []
    ctx = JobContext(tmp_path, {"id": 1}, {"version": 1})
    assert ctx.stage("generate", lambda: calls.append("generate") or {"text": "hello"})
    with pytest.raises(ConnectionError):
        ctx.stage("audio", lambda: (_ for _ in ()).throw(ConnectionError()))
    resumed = JobContext(tmp_path, {"id": 1}, {"version": 1})
    assert resumed.stage("generate", lambda: pytest.fail("regenerated")) == {"text": "hello"}

    def audio():
        calls.append("audio")
        if calls.count("audio") == 1:
            raise ConnectionError()
        return {"path": resumed.save_artifact("audios/sample.wav", b"audio")}

    result = resumed.stage("audio", audio, retries=1,
                           retry_if=lambda e: isinstance(e, ConnectionError), retry_delay=0)
    assert (tmp_path / result["path"]).read_bytes() == b"audio"
    assert calls == ["generate", "audio", "audio"]
    with pytest.raises(ValueError):
        resumed.save_artifact("../escape", b"bad")


class ContextJob(ParallelLLMJob):
    def process(self, item, ctx):
        if item["id"] == 0:
            ctx.reject("quality", "bad candidate")
        return ctx.stage("generate", lambda: item)


@pytest.mark.parametrize("processes", [1, 2])
def test_context_jobs_replace_rejections_and_resume_with_execution_changes(tmp_path, processes):
    path = tmp_path / "rows.jsonl"
    job = ContextJob(1, processes=processes, threads_per_process=1)
    source = [{"id": i} for i in range(4)]
    result = job.run_jsonl(source, path, target_rows=1, progress=False,
                          semantic_config={"version": 1})
    assert result.rejected == 1
    assert "quality: bad candidate" in result.error_path.read_text()
    job.threads_per_process = 2
    job.run_jsonl(source, path, target_rows=2, progress=False,
                  semantic_config={"version": 1})
    assert [json.loads(l)["id"] for l in path.read_text().splitlines()] == [1, 2]
    with pytest.raises(ValueError, match="Semantic"):
        job.run_jsonl(source, path, semantic_config={"version": 2}, progress=False)
