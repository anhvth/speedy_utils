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
    Line,
    RowView,
    _display_width,
    _truncate_to_width,
    _wrap_scalar,
    _wrap_text_display_width,
    all_container_paths,
    all_scalar_paths,
    build_common_parser,
    clip_line,
    collapsed_summary,
    main_jsonl,
    render_line,
    run_source_cli,
    scalar_segments,
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


# ---------------------------------------------------------------------------
# Vietnamese / Unicode display-width tests
# ---------------------------------------------------------------------------

_VIETNAMESE_TEXT = "Quốc hội nước ta hoạt động theo nguyên tắc nào?"
_CJK_TEXT = "你好世界"


def test_display_width_precomposed_vietnamese() -> None:
    """Each precomposed Vietnamese char is 1 display column wide."""
    assert _display_width(_VIETNAMESE_TEXT) == len(_VIETNAMESE_TEXT)


def test_display_width_cjk_is_double_width() -> None:
    """CJK wide chars take 2 columns each."""
    assert _display_width(_CJK_TEXT) == 2 * len(_CJK_TEXT)


def test_display_width_mixed() -> None:
    assert _display_width("a好b") == 4  # 1 + 2 + 1
    assert _display_width("ố好") == 3   # 1 + 2


def test_scalar_segments_preserves_vietnamese() -> None:
    segs = scalar_segments(_VIETNAMESE_TEXT)
    assert "ố" in segs[0][0]


def test_collapsed_summary_preserves_vietnamese_keys() -> None:
    d = {"câu_hỏi": _VIETNAMESE_TEXT, "trả_lời": "Hiến pháp 2013"}
    summary = collapsed_summary(d)
    assert "câu_hỏi" in summary
    assert "trả_lời" in summary


def test_wrap_scalar_vietnamese_widths() -> None:
    """_wrap_scalar lines must fit within the requested width."""
    wrapped = _wrap_scalar(_VIETNAMESE_TEXT, 20)
    for line in wrapped:
        assert _display_width(line) <= 20, (
            f"line too wide ({_display_width(line)}): {line!r}"
        )


def test_wrap_text_cjk_widths() -> None:
    """_wrap_text_display_width must honour display width for CJK."""
    cjk = "你好世界" * 5
    wrapped = _wrap_text_display_width(cjk, 10)
    for line in wrapped:
        assert _display_width(line) <= 10, (
            f"line too wide ({_display_width(line)}): {line!r}"
        )


def test_clip_line_preserves_vietnamese() -> None:
    line: Line = [(_VIETNAMESE_TEXT, 2)]
    clipped = clip_line(line, 30)
    result = "".join(t for t, _ in clipped)
    assert _display_width(result) <= 30
    # Should contain the start of the text, not corrupted
    assert result.startswith("Quốc")


def test_clip_line_cjk_does_not_cut_in_middle() -> None:
    """CJK wide chars should never be cut in half."""
    line: Line = [(_CJK_TEXT, 2)]
    clipped = clip_line(line, 3)
    result = "".join(t for t, _ in clipped)
    # With width 3, we can fit at most 1 CJK char (2 cols) + ellipsis (1 col)
    assert _display_width(result) <= 3
    # Should contain exactly one complete CJK char
    for ch in result.rstrip("…"):
        assert ord(ch) < 0x4E00 or ord(ch) > 0x9FFF or _display_width(ch) == 2, (
            f"split CJK char: {ch!r}"
        )


def test_truncate_to_width_handles_tight_budget() -> None:
    text, w = _truncate_to_width(_VIETNAMESE_TEXT, 1)
    assert w <= 1
    # Should get a single ASCII-range char or empty

    text, w = _truncate_to_width(_CJK_TEXT, 1)
    assert w == 0  # CJK char needs 2 cols, but budget is only 1
    assert text == ""


def test_rowview_vietnamese_nodes_render_in_bounds() -> None:
    """Every rendered node must fit within the terminal width."""
    value = {
        "câu_hỏi": _VIETNAMESE_TEXT,
        "trả_lời": (
            "Theo Hiến pháp 2013, Quốc hội hoạt động theo nguyên tắc "
            "tập trung dân chủ, làm việc theo chế độ hội nghị và "
            "quyết định theo đa số."
        ),
    }
    view = RowView(value=value)
    view.expanded = {()}
    view.rebuild(width=60)
    for i, node in enumerate(view.nodes):
        line_rendered = render_line(node, 60, i == view.cursor)
        total_w = sum(_display_width(t) for t, _ in line_rendered)
        assert total_w <= 60, f"node[{i}] too wide: {total_w}"


# ---------------------------------------------------------------------------
# Serve module tests
# ---------------------------------------------------------------------------


def _make_temp_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    return jsonl


def test_serve_json_to_html_vietnamese() -> None:
    from datasets_utils.pcat.serve import render_row

    value = {"câu_hỏi": "Quốc hội nước ta hoạt động theo nguyên tắc nào?"}
    html_out, _ = render_row(value, 0, mode="generic")
    assert "Quốc hội" in html_out
    assert "câu_hỏi" in html_out


def test_serve_render_row_generic() -> None:
    from datasets_utils.pcat.serve import render_row

    value = {"key": "value", "nested": {"a": [1, 2, 3]}}
    rendered, mode = render_row(value, 0, mode="generic")
    assert mode == "generic"
    assert "key" in rendered
    assert "value" in rendered
    assert "nested" in rendered
    # Lazy: nested array is not expanded, just a summary
    assert "3 items" in rendered or "1)" in rendered


def test_serve_render_row_raw() -> None:
    from datasets_utils.pcat.serve import render_row

    value = {"msg": "Xin chào"}
    rendered, _ = render_row(value, 0, mode="raw")
    assert "Xin chào" in rendered
    assert "plain-dump" in rendered


def test_serve_render_row_sdd() -> None:
    from datasets_utils.pcat.serve import render_row

    value = {
        "messages": [{"role": "user", "content": "Hỏi"}],
        "teacher_messages": [{"role": "assistant", "content": "Đáp"}],
    }
    rendered, _ = render_row(value, 0, mode="sdd")
    assert "side-by-side" in rendered
    assert "messages" in rendered
    assert "teacher_messages" in rendered


def test_serve_lazy_node_resolve() -> None:
    """_resolve_json_path should walk a JSON object by path segments."""
    from datasets_utils.pcat.serve import _resolve_json_path

    root = {"a": {"b": [10, 20, 30]}, "c": 42}
    val, breadcrumb = _resolve_json_path(root, "$/a/b")
    assert val == [10, 20, 30]
    assert breadcrumb == "a/b"

    val, breadcrumb = _resolve_json_path(root, "$/a/b/1")
    assert val == 20
    assert "1" in breadcrumb

    val, breadcrumb = _resolve_json_path(root, "$/c")
    assert val == 42

    val, breadcrumb = _resolve_json_path(root, "$/nonexistent")
    assert val is None


def test_serve_lazy_node_children() -> None:
    """_render_node_children returns HTML fragment for a sub-path."""
    from datasets_utils.pcat.serve import _render_node_children

    root = {"messages": [{"role": "user", "content": "hi"}]}
    fragment = _render_node_children(root, "$/messages")
    # The list shows each item's summary: [0] followed by (2) {role, content}
    assert "[0]" in fragment
    # The dict inside shows as a summary, not expanded inline
    assert "(" in fragment

    # Direct scalar access
    fragment = _render_node_children(root, "$/messages/0/role")
    assert "user" in fragment

    # Non-existent path
    fragment = _render_node_children(root, "$/nothing")
    assert "not found" in fragment


def test_serve_scalar_types_no_exceptions() -> None:
    from datasets_utils.pcat.serve import _scalar_html

    for v in [None, True, 42, 3.14, "hello", "multi\nline"]:
        html_out = _scalar_html(v)
        assert len(html_out) > 0


def test_serve_page_template_well_formed(tmp_path: Path) -> None:
    from datasets_utils.pcat.serve import _page, _build_header, _build_footer

    jsonl_path = tmp_path / "data.jsonl"
    jsonl_path.write_text("{}\n")
    source = JsonlRowSource.from_path(jsonl_path)

    header = _build_header(source, 0, "generic")
    assert "row 1" in header

    footer = _build_footer()
    assert "prev/next" in footer

    full = _page("", header + footer)
    assert "<!DOCTYPE html>" in full
    assert "<html" in full
    assert "</html>" in full


def test_serve_sdd_side_by_side_structure() -> None:
    from datasets_utils.pcat.serve import render_row

    value = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Xin chào"},
        ],
        "teacher_messages": [
            {"role": "assistant", "content": "Chào bạn!"},
        ],
    }
    rendered, res_mode = render_row(value, 0, mode="sdd")
    assert rendered.count("<h3>") == 2
    assert "Xin chào" in rendered
    assert "Chào bạn" in rendered
    assert res_mode == "sdd"


def test_serve_http_handler_error_page() -> None:
    from datasets_utils.pcat.serve import _error_page

    html_out = _error_page("Test Error", "Something went wrong")
    assert "Test Error" in html_out
    assert "Something went wrong" in html_out
    assert "text/html" in html_out or "<!DOCTYPE html>" in html_out
    # error page should link back to row 1
    assert "/row/1" in html_out


def test_serve_mode_registry() -> None:
    from datasets_utils.pcat.serve import _MODE_REGISTRY, render_row

    assert "generic" in _MODE_REGISTRY
    assert "raw" in _MODE_REGISTRY
    assert "sdd" in _MODE_REGISTRY

    # Unknown mode falls back to generic JSON tree
    rendered, mode = render_row({"a": 1}, 0, mode="nonexistent")
    assert len(rendered) > 0
    assert "<div" in rendered  # generic path produces divs


def test_serve_auto_detect_sdd_mode() -> None:
    """auto mode should detect sdd when messages + teacher_messages present."""
    from datasets_utils.pcat.serve import render_row, _detect_mode

    # SDD-style row → sdd mode
    sdd_row = {
        "messages": [{"role": "user", "content": "xin chào"}],
        "teacher_messages": [{"role": "assistant", "content": "chào bạn"}],
    }
    assert _detect_mode(sdd_row) == "sdd"
    rendered, res_mode = render_row(sdd_row, 0, mode="auto")
    assert res_mode == "sdd"
    assert "side-by-side" in rendered
    assert "chat-msg" in rendered

    # Generic row → generic mode
    generic_row = {"key": "value", "list": [1, 2, 3]}
    assert _detect_mode(generic_row) == "generic"
    rendered, res_mode = render_row(generic_row, 0, mode="auto")
    assert res_mode == "generic"
    assert "side-by-side" not in rendered

    # Row with only messages (no teacher) → generic
    only_messages = {"messages": [{"role": "user", "content": "hi"}]}
    assert _detect_mode(only_messages) == "generic"

    # Row with only teacher → generic
    only_teacher = {"teacher_messages": [{"role": "assistant", "content": "hi"}]}
    assert _detect_mode(only_teacher) == "generic"


# --------------------------------------------------------------------------
# Glob refresh tests (--serve periodic file discovery)
# ---------------------------------------------------------------------------


def test_glob_source_refresh_discovers_new_file(tmp_path: Path) -> None:
    """JsonlGlobRowSource.refresh() should pick up files created after init."""
    from datasets_utils.pcat._shared import JsonlGlobRowSource

    (tmp_path / "a.jsonl").write_text('{"id": 1}\n')
    gs = JsonlGlobRowSource.from_path(tmp_path, pattern="*.jsonl")
    assert len(gs.files) == 1

    # Create a new file that wasn't there at startup
    (tmp_path / "b.jsonl").write_text('{"id": 2}\n')
    gs.refresh()
    assert len(gs.files) == 2


def test_glob_source_refresh_removes_deleted_file(tmp_path: Path) -> None:
    """JsonlGlobRowSource.refresh() should drop files that have been removed."""
    from datasets_utils.pcat._shared import JsonlGlobRowSource

    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text('{"id": 1}\n')
    b.write_text('{"id": 2}\n')

    gs = JsonlGlobRowSource.from_path(tmp_path, pattern="*.jsonl")
    assert len(gs.files) == 2

    b.unlink()
    gs.refresh()
    assert len(gs.files) == 1
    assert gs.files[0].name == "a.jsonl"


def test_serve_files_json_endpoint(tmp_path: Path) -> None:
    """/files endpoint returns current file list as JSON."""
    import http.client
    import threading
    import time

    from datasets_utils.pcat._shared import JsonlGlobRowSource, JsonlRowSource
    from datasets_utils.pcat.serve import PcatHandler, serve

    # Start server on a random port
    import socket

    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    (tmp_path / "a.jsonl").write_text('{"id": 1}\n')
    (tmp_path / "b.jsonl").write_text('{"id": 2}\n')

    glob_source = JsonlGlobRowSource.from_path(tmp_path, pattern="*.jsonl")
    initial = JsonlRowSource.from_path(glob_source.files[0])
    port = _free_port()

    server_thread = threading.Thread(
        target=lambda: serve(
            initial,
            host="127.0.0.1",
            port=port,
            mode="generic",
            open_browser=False,
            glob_source=glob_source,
            refresh_interval=0.1,
        ),
        daemon=True,
    )
    server_thread.start()
    time.sleep(0.5)

    try:
        # Verify /files returns the current file list
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/files")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["count"] >= 2
        assert any(f["name"] == "a.jsonl" for f in data["files"])
        assert any(f["name"] == "b.jsonl" for f in data["files"])
        conn.close()

        # Create a new file and wait for refresh
        (tmp_path / "c.jsonl").write_text('{"id": 3}\n')
        time.sleep(0.5)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", "/files")
        resp = conn.getresponse()
        assert resp.status == 200
        data2 = json.loads(resp.read())
        assert data2["count"] >= 3
        assert any(f["name"] == "c.jsonl" for f in data2["files"])
        conn.close()
    finally:
        # Kill the server
        import urllib.request
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/")
        except Exception:
            pass


def test_serve_file_picker_page_has_glob_script(tmp_path: Path) -> None:
    """File picker page for glob sources includes GLOB_MODE=1 script tag."""
    from datasets_utils.pcat.serve import _file_picker_page

    file_list = [tmp_path / "a.jsonl", tmp_path / "b.jsonl"]
    for f in file_list:
        f.touch()

    html = _file_picker_page(file_list, glob_mode=True)
    assert "GLOB_MODE = 1" in html
    assert "/files" not in html  # poll URL is in JS, not in the static HTML

    html_no_glob = _file_picker_page(file_list, glob_mode=False)
    assert "GLOB_MODE" not in html_no_glob


def test_serve_file_picker_page_no_glob_single_file(tmp_path: Path) -> None:
    """Single-file sources don't get GLOB_MODE script."""
    from datasets_utils.pcat.serve import _file_picker_page

    file_list = [tmp_path / "single.jsonl"]
    file_list[0].touch()

    html = _file_picker_page(file_list, glob_mode=False)
    assert "GLOB_MODE" not in html


def test_glob_refresh_thread_get_files_thread_safe(tmp_path: Path) -> None:
    """_GlobRefreshThread.get_files() returns a stable snapshot."""
    import time

    from datasets_utils.pcat._shared import JsonlGlobRowSource
    from datasets_utils.pcat.serve import _GlobRefreshThread

    (tmp_path / "a.jsonl").write_text('{"id": 1}\n')
    (tmp_path / "b.jsonl").write_text('{"id": 2}\n')

    gs = JsonlGlobRowSource.from_path(tmp_path, pattern="*.jsonl")
    rt = _GlobRefreshThread(gs, interval=0.01)
    rt.start()

    # Let it run a few cycles
    time.sleep(0.1)

    files = rt.get_files()
    assert len(files) == 2

    # Add a file while the thread is running
    (tmp_path / "c.jsonl").write_text('{"id": 3}\n')
    time.sleep(0.3)

    files = rt.get_files()
    assert len(files) == 3
    assert any(f["name"] == "c.jsonl" for f in files)

    rt.stop()
    rt.join(timeout=2)
