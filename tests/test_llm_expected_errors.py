from types import SimpleNamespace

import pytest
from openai import LengthFinishReasonError
from pydantic import BaseModel, ValidationError

from llm_utils._traceback import clean_traceback
from speedy_utils import ParallelLLMJob


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize("kind", ["length", "validation"])
def test_expected_response_errors_are_quiet_and_replaced(kind, tmp_path, capsys):
    if kind == "length":
        error = LengthFinishReasonError(completion=SimpleNamespace(usage=None))
    else:
        try:
            Answer(value="invalid")
        except ValidationError as exc:
            error = exc

    @clean_traceback
    @clean_traceback
    def request():
        raise error

    class Job(ParallelLLMJob):
        def process(self, item):
            if item["id"] == 0:
                request()
            return item

    summary = Job(1, threads_per_process=1).run_jsonl(
        [{"id": 0}, {"id": 1}], tmp_path / "rows.jsonl",
        target_rows=1, progress=False,
    )
    assert summary.complete and summary.failed == 1
    assert summary.output_path.read_text().strip() == '{"id":1}'
    assert type(error).__name__ in summary.error_path.read_text()
    assert capsys.readouterr().err == ""
