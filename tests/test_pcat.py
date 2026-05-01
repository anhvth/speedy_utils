from __future__ import annotations

import io
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets_utils.pcat._shared import (
    AppState,
    JsonlRowSource,
    RowView,
    all_container_paths,
    all_scalar_paths,
    build_common_parser,
    main_jsonl,
    run_source_cli,
    smart_expand,
    step_sample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _InMemorySource:
    rows: list[Any]
    display_path: str = "mem"

    @property
    def total_rows(self) -> int:
        return len(self.rows)

    def load_row(self, index: int) -> Any:
        return self.rows[index]

    def reload(self) -> None:
        pass


def _make_app(
    rows: list[Any],
    sample_indices: list[int] | None = None,
    sample_pos: int = 0,
    start_index: int = 0,
) -> AppState:
    source = _InMemorySource(rows=rows)
    if sample_indices is None:
        sample_indices = []
    value = source.load_row(start_index)
    view = RowView(value=value)
    return AppState(
        source=source,
        prog="test",
        yank_path=Path("/tmp/test-yank.txt"),
        row_index=start_index,
        view=view,
        sample_indices=sample_indices,
        sample_pos=sample_pos,
    )


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def test_sample_flag_const_is_1_when_no_n_given() -> None:
    parser = build_common_parser("pcat", "desc", "path")
    args = parser.parse_args(["data.jsonl", "--sample"])
    assert args.sample == 1


def test_sample_flag_accepts_explicit_n() -> None:
    parser = build_common_parser("pcat", "desc", "path")
    args = parser.parse_args(["--sample", "20", "data.jsonl"])
    assert args.sample == 20


def test_short_sample_flag() -> None:
    parser = build_common_parser("pcat", "desc", "path")
    args = parser.parse_args(["-s", "5", "data.jsonl"])
    assert args.sample == 5


def test_sample_flag_absent_is_none() -> None:
    parser = build_common_parser("pcat", "desc", "path")
    args = parser.parse_args(["data.jsonl"])
    assert args.sample is None


# ---------------------------------------------------------------------------
# step_sample — always picks a fresh random row (like [ / ] but random)
# ---------------------------------------------------------------------------


def test_step_sample_always_moves_to_different_row() -> None:
    rows = [{"i": i} for i in range(10)]
    app = _make_app(rows, start_index=5)

    prev = app.row_index
    for _ in range(30):
        step_sample(app, 30)
        assert app.row_index != prev, "step_sample must move to a different row each press"
        prev = app.row_index


def test_step_sample_status_format() -> None:
    rows = [{"i": i} for i in range(10)]
    app = _make_app(rows, start_index=0)

    status = step_sample(app, 30)
    assert status.startswith("random → row ")
    assert "/10" in status


def test_step_sample_single_row_stays_put() -> None:
    rows = [{"i": 0}]
    app = _make_app(rows, start_index=0)
    status = step_sample(app, 30)
    assert app.row_index == 0
    assert status == "only one row"


def test_step_sample_works_regardless_of_sample_indices() -> None:
    # --sample N pre-draws a starting pool, but s always picks a fresh random row.
    rows = [{"i": i} for i in range(20)]
    app = _make_app(rows, sample_indices=[3, 7, 15], sample_pos=0, start_index=3)

    prev = app.row_index
    for _ in range(20):
        step_sample(app, 30)
        assert app.row_index != prev
        prev = app.row_index


# ---------------------------------------------------------------------------
# run_source_cli — plain mode with --sample
# ---------------------------------------------------------------------------


def test_run_source_cli_sample_plain_prints_valid_json(tmp_path: Path) -> None:
    jsonl = tmp_path / "data.jsonl"
    rows = [{"id": i} for i in range(50)]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    source = JsonlRowSource.from_path(jsonl)
    parser = build_common_parser("pcat", "desc", "path")

    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        rc = run_source_cli(parser, source, index=None, plain=True, prog="pcat", sample=5)
    finally:
        sys.stdout = old

    assert rc == 0
    parsed = json.loads(buf.getvalue())
    assert 0 <= parsed["id"] < 50


def test_run_source_cli_sample_n_larger_than_dataset_is_clamped(tmp_path: Path) -> None:
    jsonl = tmp_path / "data.jsonl"
    rows = [{"id": i} for i in range(3)]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    source = JsonlRowSource.from_path(jsonl)
    parser = build_common_parser("pcat", "desc", "path")

    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        rc = run_source_cli(parser, source, index=None, plain=True, prog="pcat", sample=999)
    finally:
        sys.stdout = old

    assert rc == 0
    parsed = json.loads(buf.getvalue())
    assert 0 <= parsed["id"] < 3


# ---------------------------------------------------------------------------
# End-to-end: main_jsonl with --sample --plain
# ---------------------------------------------------------------------------


def test_main_jsonl_sample_plain_returns_valid_row(tmp_path: Path) -> None:
    jsonl = tmp_path / "data.jsonl"
    rows = [{"id": i, "val": f"v{i}"} for i in range(100)]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    buf = io.StringIO()
    old, sys.stdout = sys.stdout, buf
    try:
        rc = main_jsonl(["--sample", "10", "--plain", str(jsonl)])
    finally:
        sys.stdout = old

    assert rc == 0
    parsed = json.loads(buf.getvalue())
    assert "id" in parsed and "val" in parsed
    assert 0 <= parsed["id"] < 100


def test_main_jsonl_sample_indices_are_unique(tmp_path: Path) -> None:
    # Run the sampler multiple times in plain mode; collected first rows should
    # not be all identical (probability of collisions across 10 runs on 1000
    # rows is negligible, so this is a determinism smoke-check).
    jsonl = tmp_path / "data.jsonl"
    rows = [{"id": i} for i in range(1000)]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    seen_ids: set[int] = set()
    for _ in range(10):
        buf = io.StringIO()
        old, sys.stdout = sys.stdout, buf
        try:
            main_jsonl(["--sample", "1", "--plain", str(jsonl)])
        finally:
            sys.stdout = old
        seen_ids.add(json.loads(buf.getvalue())["id"])

    # With 10 draws from 1000 rows the chance all land on the same row is 1/1000^9
    assert len(seen_ids) > 1


# ---------------------------------------------------------------------------
# Original regression test (kept)
# ---------------------------------------------------------------------------


def test_smart_expand_expands_full_multiline_scalars_for_large_rows() -> None:
    value = {
        "title": "example",
        "payload": {
            "body": "line\n" * 200,
            "notes": ["short", "another\nmultiline\nvalue"],
        },
    }

    view = RowView(value=value)

    smart_expand(view, screen_height=5)

    assert view.expanded == set(all_container_paths(value))
    assert view.scalar_expanded == set(all_scalar_paths(value))
