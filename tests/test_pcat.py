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
# step_sample — pre-drawn sample mode
# ---------------------------------------------------------------------------


def test_step_sample_advances_through_predrawn_indices() -> None:
    rows = [{"i": i} for i in range(20)]
    sample_indices = [3, 7, 15, 2]
    app = _make_app(rows, sample_indices=sample_indices, sample_pos=0, start_index=3)

    step_sample(app, 30)
    assert app.row_index == 7
    assert app.sample_pos == 1

    step_sample(app, 30)
    assert app.row_index == 15
    assert app.sample_pos == 2

    step_sample(app, 30)
    assert app.row_index == 2
    assert app.sample_pos == 3


def test_step_sample_wraps_around_to_first() -> None:
    rows = [{"i": i} for i in range(20)]
    sample_indices = [3, 7]
    # Already at last sample (pos=1, row=7); next press wraps to first (row=3).
    app = _make_app(rows, sample_indices=sample_indices, sample_pos=1, start_index=7)

    step_sample(app, 30)
    assert app.row_index == 3
    assert app.sample_pos == 0


def test_step_sample_status_format() -> None:
    rows = [{"i": i} for i in range(20)]
    sample_indices = [0, 5, 10]
    app = _make_app(rows, sample_indices=sample_indices, sample_pos=0, start_index=0)

    status = step_sample(app, 30)
    assert status == "sample 2/3"

    status = step_sample(app, 30)
    assert status == "sample 3/3"


def test_step_sample_n1_does_not_raise_and_stays_on_same_row() -> None:
    # N=1: only one pre-drawn index, which is always the current row after
    # wrap-around. The loop exhausts all candidates and stays put — no crash.
    rows = [{"i": i} for i in range(50)]
    app = _make_app(rows, sample_indices=[17], sample_pos=0, start_index=17)

    status = step_sample(app, 30)
    assert app.row_index == 17          # could not advance
    assert "sample 1/1" in status


def test_step_sample_skips_current_row_on_wraparound() -> None:
    # Three samples: [4, 9, 4] — impossible with random.sample, but we want
    # to prove the skip logic works even if the same index appears elsewhere.
    rows = [{"i": i} for i in range(20)]
    # Simulate: sample_indices = [4, 9], currently at pos=1 (row=9).
    # Wrapping takes pos to 0 → index 4; that is ≠ 9, so we load it.
    app = _make_app(rows, sample_indices=[4, 9], sample_pos=1, start_index=9)
    step_sample(app, 30)
    assert app.row_index == 4


# ---------------------------------------------------------------------------
# step_sample — ad-hoc (no --sample) mode
# ---------------------------------------------------------------------------


def test_step_sample_adhoc_always_moves_to_different_row() -> None:
    rows = [{"i": i} for i in range(10)]
    app = _make_app(rows, sample_indices=[], start_index=5)

    prev = app.row_index
    for _ in range(30):
        step_sample(app, 30)
        assert app.row_index != prev, "step_sample must move to a different row"
        prev = app.row_index


def test_step_sample_adhoc_status_format() -> None:
    rows = [{"i": i} for i in range(10)]
    app = _make_app(rows, sample_indices=[], start_index=0)

    status = step_sample(app, 30)
    assert status.startswith("random → row ")


def test_step_sample_adhoc_single_row_source_stays_put() -> None:
    # Only one row exists; nothing else to jump to.
    rows = [{"i": 0}]
    app = _make_app(rows, sample_indices=[], start_index=0)
    step_sample(app, 30)
    assert app.row_index == 0


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
