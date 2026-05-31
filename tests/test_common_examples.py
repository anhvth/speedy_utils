"""Examples-as-tests for small speedy_utils public helpers."""

from __future__ import annotations

import json

from speedy_utils import (
    dedup,
    dump_jsonl,
    flatten_list,
    load_by_ext,
    load_jsonl,
)


def test_flatten_list_concatenates_sublists():
    assert flatten_list([[1, 2], [3], []]) == [1, 2, 3]


def test_dedup_keeps_first_occurrence_by_key():
    items = [{"id": 1}, {"id": 2}, {"id": 1}]
    assert dedup(items, key=lambda x: x["id"]) == [{"id": 1}, {"id": 2}]


def test_jsonl_round_trip(tmp_path):
    rows = [{"a": 1}, {"a": 2}]
    path = tmp_path / "rows.jsonl"
    dump_jsonl(rows, str(path))
    assert load_jsonl(str(path)) == rows


def test_load_by_ext_dispatches_on_suffix(tmp_path):
    json_path = tmp_path / "data.json"
    json_path.write_text(json.dumps({"k": "v"}), encoding="utf-8")
    assert load_by_ext(str(json_path)) == {"k": "v"}

    jsonl_path = tmp_path / "data.jsonl"
    dump_jsonl([{"i": 0}, {"i": 1}], str(jsonl_path))
    assert load_by_ext(str(jsonl_path)) == [{"i": 0}, {"i": 1}]
