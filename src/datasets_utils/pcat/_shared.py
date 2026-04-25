from __future__ import annotations

import argparse
import curses
import json
import sys
import textwrap
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence


# ---------------------------------------------------------------------------
# Color pairs
# ---------------------------------------------------------------------------

CP_KEY = 1
CP_STRING = 2
CP_NUMBER = 3
CP_BOOL = 4
CP_NULL = 5
CP_PUNCT = 6
CP_SUMMARY = 7
CP_HEADER = 8
CP_PATH = 9
CP_MATCH = 10


# ---------------------------------------------------------------------------
# Row sources
# ---------------------------------------------------------------------------


class RowSource(Protocol):
    display_path: str

    @property
    def total_rows(self) -> int: ...

    def load_row(self, index: int) -> Any: ...

    def reload(self) -> None: ...


@dataclass
class JsonlRowSource:
    path: Path
    offsets: list[tuple[int, int]]
    display_path: str

    @classmethod
    def from_path(cls, path: Path) -> JsonlRowSource:
        offsets = build_jsonl_index(path)
        return cls(path=path, offsets=offsets, display_path=str(path))

    @property
    def total_rows(self) -> int:
        return len(self.offsets)

    def load_row(self, index: int) -> Any:
        offset, length = self.offsets[index]
        return load_jsonl_row(self.path, offset, length)

    def reload(self) -> None:
        self.offsets = build_jsonl_index(self.path)


@dataclass
class JsonSingleRowSource:
    path: Path
    display_path: str

    @classmethod
    def from_path(cls, path: Path) -> JsonSingleRowSource:
        return cls(path=path, display_path=str(path))

    @property
    def total_rows(self) -> int:
        return 1

    def load_row(self, index: int) -> Any:
        if index != 0:
            raise ValueError(f"row index {index} out of range, only 1 row available")
        return load_json_file_row(self.path)

    def reload(self) -> None:
        return


@dataclass
class JsonDirRowSource:
    path: Path
    files: list[Path]
    display_path: str

    @classmethod
    def from_path(cls, path: Path) -> JsonDirRowSource:
        files = build_json_dir_index(path)
        return cls(path=path, files=files, display_path=f"{path}/*.json")

    @property
    def total_rows(self) -> int:
        return len(self.files)

    def load_row(self, index: int) -> Any:
        return load_json_file_row(self.files[index])

    def reload(self) -> None:
        self.files = build_json_dir_index(self.path)


@dataclass
class HFDatasetRowSource:
    path: Path
    dataset: Any
    display_path: str

    @classmethod
    def from_path(cls, path: Path, split: str | None = None) -> HFDatasetRowSource:
        try:
            from datasets import DatasetDict, load_from_disk
        except ModuleNotFoundError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "pcat-hf-dataset requires the 'datasets' package. Install it with: pip install datasets"
            ) from exc

        try:
            loaded = load_from_disk(str(path))
        except Exception as exc:
            raise ValueError(f"failed to load dataset from {path}: {exc}") from exc

        if isinstance(loaded, DatasetDict):
            if split is None:
                if len(loaded) == 1:
                    split = str(next(iter(loaded)))
                else:
                    choices = ", ".join(str(k) for k in loaded)
                    raise ValueError(
                        f"dataset has multiple splits; use --split one of: {choices}"
                    )
            if split not in loaded:
                choices = ", ".join(str(k) for k in loaded)
                raise ValueError(
                    f"unknown split '{split}'; available splits: {choices}"
                )
            dataset = loaded[split]
            display_path = f"{path}::{split}"
        else:
            if split is not None:
                raise ValueError(
                    "--split is only valid when load_from_disk() returns a DatasetDict"
                )
            dataset = loaded
            display_path = str(path)

        return cls(path=path, dataset=dataset, display_path=display_path)

    @property
    def total_rows(self) -> int:
        return len(self.dataset)

    def load_row(self, index: int) -> Any:
        try:
            row = self.dataset[index]
        except Exception as exc:
            raise ValueError(f"failed to load row {index}: {exc}") from exc
        if isinstance(row, dict):
            return row
        return {"value": row}

    def reload(self) -> None:
        return


# ---------------------------------------------------------------------------
# JSONL index — built once, O(1) row access thereafter
# ---------------------------------------------------------------------------


def build_jsonl_index(path: Path) -> list[tuple[int, int]]:
    """Scan the file once and return a list of (byte_offset, byte_length) per
    non-empty JSONL row. Subsequent row loads are O(1) via seek+read."""
    offsets: list[tuple[int, int]] = []
    with path.open("rb") as handle:
        pos = 0
        for raw in handle:
            if raw.strip():
                offsets.append((pos, len(raw)))
            pos += len(raw)
    return offsets


def load_jsonl_row(path: Path, offset: int, length: int) -> Any:
    """O(1) load: seek to byte offset and read exactly `length` bytes."""
    with path.open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(length)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: row at offset {offset} is not valid JSON: {exc}"
        ) from exc


def build_json_dir_index(path: Path) -> list[Path]:
    """Treat a directory of .json files as a JSONL-like row source.

    Each top-level `*.json` file is one row. Files are sorted by name for
    stable navigation and index selection.
    """
    return sorted(
        child
        for child in path.iterdir()
        if child.is_file() and child.suffix.lower() == ".json"
    )


def load_json_file_row(path: Path) -> Any:
    """Load a single JSON file lazily, one file per logical row."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Tree model
# ---------------------------------------------------------------------------


@dataclass
class Node:
    """A position inside the JSON tree of a single row."""

    path: tuple
    key_repr: str
    value: Any
    depth: int
    parent: tuple | None
    is_continuation: bool = False
    continuation_text: str = ""

    @property
    def container_kind(self) -> str | None:
        if isinstance(self.value, dict):
            return "dict"
        if isinstance(self.value, list):
            return "list"
        return None

    @property
    def is_container(self) -> bool:
        return self.container_kind is not None


def _wrap_scalar(value: Any, text_width: int) -> list[str]:
    """Return wrapped lines for a scalar (used when a scalar node is expanded)."""
    text_width = max(4, text_width)
    if isinstance(value, str):
        raw = value.replace("\r\n", "\n").replace("\r", "\n")
    else:
        raw = json.dumps(value, ensure_ascii=False, indent=2)
    result: list[str] = []
    for physical in raw.split("\n"):
        if not physical:
            result.append("")
            continue
        wrapped = textwrap.wrap(
            physical,
            width=text_width,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        result.extend(wrapped)
    return result or [""]


def build_nodes(
    value: Any,
    expanded: set[tuple],
    scalar_expanded: set[tuple] | None = None,
    render_width: int = 120,
) -> list[Node]:
    """DFS-flatten the JSON value, honoring `expanded` and `scalar_expanded` sets."""
    if scalar_expanded is None:
        scalar_expanded = set()

    nodes: list[Node] = []

    def visit(
        value: Any, path: tuple, key_repr: str, depth: int, parent: tuple | None
    ) -> None:
        node = Node(
            path=path, key_repr=key_repr, value=value, depth=depth, parent=parent
        )
        nodes.append(node)
        if node.is_container:
            if path not in expanded:
                return
            if isinstance(value, dict):
                for key in value:
                    visit(value[key], path + (key,), str(key), depth + 1, path)
            else:
                for idx, item in enumerate(value):
                    visit(item, path + (idx,), f"[{idx}]", depth + 1, path)
        elif path in scalar_expanded:
            prefix_width = 2 + (depth + 1) * 2
            text_width = max(8, render_width - prefix_width)
            for line_text in _wrap_scalar(value, text_width):
                nodes.append(
                    Node(
                        path=path,
                        key_repr="",
                        value=value,
                        depth=depth + 1,
                        parent=parent,
                        is_continuation=True,
                        continuation_text=line_text,
                    )
                )

    visit(value, (), "", 0, None)
    return nodes


def all_container_paths(value: Any) -> list[tuple]:
    out: list[tuple] = []

    def visit(value: Any, path: tuple) -> None:
        if isinstance(value, dict):
            out.append(path)
            for key, child in value.items():
                visit(child, path + (key,))
        elif isinstance(value, list):
            out.append(path)
            for idx, child in enumerate(value):
                visit(child, path + (idx,))

    visit(value, ())
    return out


def all_scalar_paths(value: Any) -> list[tuple]:
    """Return paths of string values that contain newlines (deserve full expansion)."""
    out: list[tuple] = []

    def visit(value: Any, path: tuple) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, path + (key,))
        elif isinstance(value, list):
            for idx, child in enumerate(value):
                visit(child, path + (idx,))
        elif isinstance(value, str) and "\n" in value:
            out.append(path)

    visit(value, ())
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


Segment = tuple[str, int]
Line = list[Segment]


def render_line(node: Node, width: int, is_cursor: bool) -> Line:
    indent = "  " * node.depth
    cursor_marker = "▌ " if is_cursor else "  "
    prefix_text = cursor_marker + indent

    if node.is_continuation:
        text = prefix_text + node.continuation_text
        pair = CP_STRING if isinstance(node.value, str) else CP_NUMBER
        return [(text[:width], pair)]

    line: Line = [(prefix_text, 0)]

    if node.key_repr:
        if node.parent is not None and isinstance_at(node.parent, list, node):
            line.append((node.key_repr, CP_PUNCT))
            line.append((" ", 0))
        else:
            line.append((node.key_repr, CP_KEY))
            line.append((": ", CP_PUNCT))

    if node.is_container:
        kind = node.container_kind
        if node.path in EXPANDED_HINT:
            open_brace = "{" if kind == "dict" else "["
            line.append((open_brace, CP_PUNCT))
        else:
            line.append((collapsed_summary(node.value), CP_SUMMARY))
    elif node.path in SCALAR_EXPANDED_HINT:
        line.append(("▾", CP_SUMMARY))
    else:
        line.extend(scalar_segments(node.value))

    return clip_line(line, width)


EXPANDED_HINT: set[tuple] = set()
SCALAR_EXPANDED_HINT: set[tuple] = set()


def isinstance_at(
    parent_path: tuple, kind, node: Node
) -> bool:  # pragma: no cover - tiny helper
    return getattr(node, "_parent_is_list", False)


def collapsed_summary(value: Any) -> str:
    if isinstance(value, dict):
        n = len(value)
        if n == 0:
            return "{}"
        keys = list(value.keys())
        preview = ", ".join(json.dumps(str(k), ensure_ascii=False) for k in keys[:3])
        if n > 3:
            preview += ", …"
        return "{" + preview + f"}}  ({n} keys)"
    if isinstance(value, list):
        n = len(value)
        if n == 0:
            return "[]"
        return f"[…]  ({n} items)"
    return json.dumps(value, ensure_ascii=False)


def scalar_segments(value: Any) -> list[Segment]:
    if value is None:
        return [("null", CP_NULL)]
    if isinstance(value, bool):
        return [("true" if value else "false", CP_BOOL)]
    if isinstance(value, (int, float)):
        return [(json.dumps(value), CP_NUMBER)]
    if isinstance(value, str):
        return [(json.dumps(value, ensure_ascii=False), CP_STRING)]
    return [(json.dumps(value, ensure_ascii=False), 0)]


def clip_line(line: Line, width: int) -> Line:
    if width <= 0:
        return []
    used = 0
    out: Line = []
    for text, attr in line:
        if used >= width:
            break
        remaining = width - used
        if len(text) <= remaining:
            out.append((text, attr))
            used += len(text)
        else:
            if remaining <= 1:
                out.append(("…", CP_PUNCT))
                used += 1
            else:
                out.append((text[: remaining - 1] + "…", attr))
                used = width
            break
    return out


# ---------------------------------------------------------------------------
# View state
# ---------------------------------------------------------------------------


@dataclass
class RowView:
    value: Any
    expanded: set[tuple] = field(default_factory=set)
    scalar_expanded: set[tuple] = field(default_factory=set)
    render_width: int = 120
    cursor: int = 0
    top: int = 0
    nodes: list[Node] = field(default_factory=list)

    def rebuild(self, width: int | None = None) -> None:
        if width is not None:
            self.render_width = width
        nodes = build_nodes(
            self.value, self.expanded, self.scalar_expanded, self.render_width
        )
        for node in nodes:
            if node.parent is None or node.is_continuation:
                continue
            parent_value = resolve_path(self.value, node.parent)
            node._parent_is_list = isinstance(parent_value, list)  # type: ignore[attr-defined]
        self.nodes = nodes
        if self.cursor >= len(nodes):
            self.cursor = max(0, len(nodes) - 1)

    def toggle(self) -> None:
        if not self.nodes:
            return
        node = self.nodes[self.cursor]
        if node.is_continuation or not node.is_container:
            if node.path in self.scalar_expanded:
                self.scalar_expanded.discard(node.path)
            else:
                self.scalar_expanded.add(node.path)
            self.rebuild()
            return
        if node.path in self.expanded:
            prefix = node.path
            self.expanded = {
                p for p in self.expanded if not _starts_with(p, prefix) or p == prefix
            }
            self.expanded.discard(node.path)
        else:
            self.expanded.add(node.path)
        self.rebuild()

    def expand_all(self) -> None:
        self.expanded = set(all_container_paths(self.value))
        self.scalar_expanded = set(all_scalar_paths(self.value))
        self.rebuild()

    def collapse_all(self) -> None:
        self.expanded = set()
        self.scalar_expanded = set()
        self.cursor = 0
        self.rebuild()

    def move(self, delta: int) -> None:
        if not self.nodes:
            return
        self.cursor = max(0, min(len(self.nodes) - 1, self.cursor + delta))


def resolve_path(value: Any, path: tuple) -> Any:
    cur = value
    for step in path:
        cur = cur[step]
    return cur


def _starts_with(path: tuple, prefix: tuple) -> bool:
    return len(path) >= len(prefix) and path[: len(prefix)] == prefix


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def find_matches(view: RowView, query: str) -> list[int]:
    if not query:
        return []
    lowered = query.lower()

    matched_paths: list[tuple] = []

    def visit(value: Any, path: tuple, key_repr: str) -> None:
        haystacks = [str(key_repr).lower(), "/".join(str(p) for p in path).lower()]
        if not isinstance(value, (dict, list)):
            haystacks.append(json.dumps(value, ensure_ascii=False).lower())
        if any(lowered in h for h in haystacks):
            matched_paths.append(path)
        if isinstance(value, dict):
            for k, v in value.items():
                visit(v, path + (k,), k)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, path + (i,), f"[{i}]")

    visit(view.value, (), "")

    if not matched_paths:
        return []

    for path in matched_paths:
        for i in range(len(path)):
            view.expanded.add(path[:i])
    view.rebuild()

    path_to_idx = {
        node.path: idx
        for idx, node in enumerate(view.nodes)
        if not node.is_continuation
    }
    return [path_to_idx[p] for p in matched_paths if p in path_to_idx]


# ---------------------------------------------------------------------------
# Plain mode
# ---------------------------------------------------------------------------


def print_plain(value: Any, output) -> None:
    json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=False)
    output.write("\n")


# ---------------------------------------------------------------------------
# Curses UI
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    source: RowSource
    prog: str
    yank_path: Path
    row_index: int
    view: RowView
    status: str = ""
    last_query: str = ""
    matches: list[int] = field(default_factory=list)
    match_pos: int = -1

    @property
    def total_rows(self) -> int:
        return self.source.total_rows


def build_help_text(prog: str, yank_path: Path) -> list[str]:
    return [
        f"{prog} — keybindings",
        "",
        "  j / ↓        cursor down",
        "  k / ↑        cursor up",
        "  Ctrl-d/Ctrl-u  half page",
        "  PgDn / PgUp  full page",
        "  g / G        first / last node",
        "  Enter/Space  toggle fold (containers) / expand full text (scalars)",
        "  e            toggle expand-all / collapse-all",
        "  c            collapse all",
        "  [ / ]        prev / next row",
        "  { / }        first / last row",
        "  :N           go to row N (1-based; negative from end: :-1)",
        "  /            search (case-insensitive)",
        "  n / N        next / prev match",
        "  r            reload source from disk",
        f"  y            yank current node value (OSC52 + {yank_path})",
        "  ?            this help",
        "  q / Esc      quit",
        "",
        "  press any key to close help",
    ]


def init_colors() -> None:
    if not curses.has_colors():
        return
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    curses.init_pair(CP_KEY, curses.COLOR_CYAN, bg)
    curses.init_pair(CP_STRING, curses.COLOR_GREEN, bg)
    curses.init_pair(CP_NUMBER, curses.COLOR_MAGENTA, bg)
    curses.init_pair(CP_BOOL, curses.COLOR_YELLOW, bg)
    curses.init_pair(CP_NULL, curses.COLOR_RED, bg)
    curses.init_pair(CP_PUNCT, curses.COLOR_WHITE, bg)
    curses.init_pair(CP_SUMMARY, curses.COLOR_BLUE, bg)
    curses.init_pair(CP_HEADER, curses.COLOR_WHITE, bg)
    curses.init_pair(CP_PATH, curses.COLOR_YELLOW, bg)
    curses.init_pair(CP_MATCH, curses.COLOR_BLACK, curses.COLOR_YELLOW)


def addstr_safe(stdscr, y: int, x: int, text: str, attr: int = 0) -> None:
    with suppress(curses.error):
        stdscr.addstr(y, x, text, attr)


def draw(stdscr, app: AppState) -> None:
    height, width = stdscr.getmaxyx()
    if width != app.view.render_width:
        app.view.rebuild(width)
    stdscr.clear()

    header = f" {app.source.display_path}  row {app.row_index + 1}/{app.total_rows} "
    addstr_safe(
        stdscr,
        0,
        0,
        header.ljust(width - 1)[: width - 1],
        curses.A_BOLD | (curses.color_pair(CP_HEADER) if curses.has_colors() else 0),
    )

    crumb = ""
    if app.view.nodes:
        node = app.view.nodes[app.view.cursor]
        crumb = "/" + "/".join(str(p) for p in node.path) if node.path else "/"
    addstr_safe(
        stdscr,
        1,
        0,
        crumb[: width - 1].ljust(width - 1),
        curses.color_pair(CP_PATH) if curses.has_colors() else 0,
    )

    body_top = 2
    body_height = max(1, height - body_top - 1)

    if app.view.cursor < app.view.top:
        app.view.top = app.view.cursor
    elif app.view.cursor >= app.view.top + body_height:
        app.view.top = app.view.cursor - body_height + 1
    app.view.top = max(0, min(app.view.top, max(0, len(app.view.nodes) - body_height)))

    EXPANDED_HINT.clear()
    EXPANDED_HINT.update(app.view.expanded)
    SCALAR_EXPANDED_HINT.clear()
    SCALAR_EXPANDED_HINT.update(app.view.scalar_expanded)

    visible = app.view.nodes[app.view.top : app.view.top + body_height]
    for row in range(body_top, body_top + body_height):
        try:
            stdscr.move(row, 0)
            stdscr.clrtoeol()
        except curses.error:
            pass
    for row, node in enumerate(visible, start=body_top):
        idx = app.view.top + (row - body_top)
        is_cursor = idx == app.view.cursor
        line = render_line(node, width - 1, is_cursor)
        x = 0
        base_attr = curses.A_REVERSE if is_cursor else 0
        if idx in app.matches:
            base_attr |= (
                curses.color_pair(CP_MATCH)
                if curses.has_colors()
                else curses.A_STANDOUT
            )
        for text, pair in line:
            attr = base_attr
            if pair and curses.has_colors() and idx not in app.matches:
                attr |= curses.color_pair(pair)
            addstr_safe(stdscr, row, x, text, attr)
            x += len(text)

    footer = app.status or default_footer(app)
    addstr_safe(
        stdscr, height - 1, 0, footer[: width - 1].ljust(width - 1), curses.A_DIM
    )
    stdscr.refresh()


def default_footer(app: AppState) -> str:
    n = len(app.view.nodes)
    pos = app.view.cursor + 1 if n else 0
    extra = ""
    if app.last_query:
        if app.matches:
            extra = f"  /{app.last_query}  {app.match_pos + 1}/{len(app.matches)}"
        else:
            extra = f"  /{app.last_query}  no matches"
    return f" node {pos}/{n}  ?=help  q=quit{extra}"


def show_help(stdscr, app: AppState) -> None:
    height, width = stdscr.getmaxyx()
    stdscr.erase()
    for i, line in enumerate(build_help_text(app.prog, app.yank_path)):
        if i >= height:
            break
        addstr_safe(stdscr, i, 2, line[: width - 3])
    stdscr.refresh()
    stdscr.getch()


def prompt(stdscr, prefix: str) -> str:
    height, width = stdscr.getmaxyx()
    curses.echo()
    with suppress(curses.error):
        curses.curs_set(1)
    addstr_safe(stdscr, height - 1, 0, " " * (width - 1))
    addstr_safe(stdscr, height - 1, 0, prefix)
    stdscr.refresh()
    try:
        result = stdscr.getstr(height - 1, len(prefix), 256).decode(
            "utf-8", errors="replace"
        )
    except Exception:
        result = ""
    curses.noecho()
    with suppress(curses.error):
        curses.curs_set(0)
    return result


def osc52_copy(text: str) -> None:
    import base64

    payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
    sys.stdout.write(f"\x1b]52;c;{payload}\x07")
    sys.stdout.flush()


def yank_node(view: RowView, out_path: Path) -> str:
    if not view.nodes:
        return "no node to yank"
    node = view.nodes[view.cursor]
    pretty = json.dumps(node.value, ensure_ascii=False, indent=2)
    try:
        out_path.write_text(pretty, encoding="utf-8")
    except OSError as exc:
        return f"yank failed: {exc}"
    with suppress(Exception):
        osc52_copy(pretty)
    label = "/".join(str(p) for p in node.path) or "<root>"
    return f"yanked {label} -> {out_path}"


_EXPAND_CHARS_PER_LINE = 80


def smart_expand(view: RowView, screen_height: int) -> None:
    view.expanded = set(all_container_paths(view.value))
    candidate_scalars = set(all_scalar_paths(view.value))
    body_lines = max(1, screen_height - 4)
    char_budget = body_lines * _EXPAND_CHARS_PER_LINE
    raw_size = len(json.dumps(view.value, ensure_ascii=False))
    if raw_size <= char_budget:
        view.scalar_expanded = candidate_scalars
    else:
        view.scalar_expanded = set()
    view.rebuild()


def load_row(app: AppState, new_index: int, screen_height: int = 30) -> None:
    new_index = max(0, min(app.total_rows - 1, new_index))
    if new_index == app.row_index and app.view.nodes:
        return
    try:
        value = app.source.load_row(new_index)
    except ValueError as exc:
        app.status = f"row {new_index + 1}: {exc}"
        return
    app.row_index = new_index
    app.view = RowView(value=value)
    smart_expand(app.view, screen_height)
    app.matches = []
    app.match_pos = -1
    app.status = ""


def run_search(app: AppState, query: str) -> None:
    app.last_query = query
    app.matches = find_matches(app.view, query)
    if not app.matches:
        app.match_pos = -1
        app.status = f"no matches for /{query}"
        return
    target = next((i for i, n in enumerate(app.matches) if n >= app.view.cursor), 0)
    app.match_pos = target
    app.view.cursor = app.matches[target]
    app.status = ""


def step_match(app: AppState, delta: int) -> None:
    if not app.matches:
        app.status = "no active search"
        return
    app.match_pos = (app.match_pos + delta) % len(app.matches)
    app.view.cursor = app.matches[app.match_pos]


def reload_source(app: AppState, screen_height: int) -> None:
    old_total = app.total_rows
    old_index = app.row_index
    was_on_last_row = old_total > 0 and old_index >= old_total - 1
    try:
        app.source.reload()
    except Exception as exc:
        app.status = f"reload failed: {exc}"
        return

    if app.total_rows <= 0:
        app.status = "reload failed: source has no rows"
        return

    target_index = (
        app.total_rows - 1 if was_on_last_row else min(old_index, app.total_rows - 1)
    )
    load_row(app, target_index, screen_height)
    if app.total_rows == old_total:
        app.status = f"reloaded {app.total_rows} rows"
    else:
        delta = app.total_rows - old_total
        sign = "+" if delta >= 0 else ""
        app.status = f"reloaded {app.total_rows} rows ({sign}{delta})"


def run_curses(stdscr, app: AppState) -> int:
    init_colors()
    with suppress(curses.error):
        curses.curs_set(0)
    stdscr.keypad(True)
    height, width = stdscr.getmaxyx()
    smart_expand(app.view, height)
    app.view.rebuild(width)

    while True:
        draw(stdscr, app)
        try:
            key = stdscr.getch()
        except KeyboardInterrupt:
            return 0
        app.status = ""

        if key in (ord("q"), 27):
            return 0
        if key in (curses.KEY_DOWN, ord("j")):
            app.view.move(1)
        elif key in (curses.KEY_UP, ord("k")):
            app.view.move(-1)
        elif key == 4:
            height, _ = stdscr.getmaxyx()
            app.view.move(max(1, (height - 3) // 2))
        elif key == 21:
            height, _ = stdscr.getmaxyx()
            app.view.move(-max(1, (height - 3) // 2))
        elif key == ord("g"):
            app.view.cursor = 0
        elif key == ord("G"):
            app.view.cursor = max(0, len(app.view.nodes) - 1)
        elif key in (curses.KEY_ENTER, 10, 13, ord(" ")):
            app.view.toggle()
        elif key == ord("e"):
            all_paths = set(all_container_paths(app.view.value))
            if all_paths and all_paths.issubset(app.view.expanded):
                app.view.collapse_all()
                if isinstance(app.view.value, (dict, list)):
                    app.view.expanded.add(())
                    app.view.rebuild()
            else:
                app.view.expand_all()
        elif key == ord("c"):
            app.view.collapse_all()
            if isinstance(app.view.value, (dict, list)):
                app.view.expanded.add(())
                app.view.rebuild()
        elif key == ord("["):
            if app.row_index <= 0:
                curses.beep()
            else:
                height, _ = stdscr.getmaxyx()
                load_row(app, app.row_index - 1, height)
        elif key == ord("]"):
            if app.row_index >= app.total_rows - 1:
                curses.beep()
            else:
                height, _ = stdscr.getmaxyx()
                load_row(app, app.row_index + 1, height)
        elif key == ord("{"):
            height, _ = stdscr.getmaxyx()
            load_row(app, 0, height)
        elif key == ord("}"):
            height, _ = stdscr.getmaxyx()
            load_row(app, app.total_rows - 1, height)
        elif key == ord("/"):
            query = prompt(stdscr, "/")
            run_search(app, query.strip())
        elif key == ord("n"):
            step_match(app, 1)
        elif key == ord("N"):
            step_match(app, -1)
        elif key == ord("y"):
            app.status = yank_node(app.view, app.yank_path)
        elif key == ord("r"):
            height, _ = stdscr.getmaxyx()
            reload_source(app, height)
        elif key == ord(":"):
            raw = prompt(stdscr, ":").strip()
            if raw:
                try:
                    target = int(raw)
                    if target >= 0:
                        target -= 1
                    else:
                        target = app.total_rows + target
                    height, _ = stdscr.getmaxyx()
                    load_row(app, target, height)
                except ValueError:
                    app.status = f"goto: invalid row number '{raw}'"
        elif key == ord("?"):
            show_help(stdscr, app)
        elif key in (curses.KEY_NPAGE, ord("f") - 96):
            height, _ = stdscr.getmaxyx()
            app.view.move(max(1, height - 4))
        elif key in (curses.KEY_PPAGE, ord("b") - 96):
            height, _ = stdscr.getmaxyx()
            app.view.move(-max(1, height - 4))
        elif key == curses.KEY_RESIZE:
            continue


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def resolve_initial_index(index: int | None, total_rows: int) -> int:
    resolved = index if index is not None else total_rows - 1
    if resolved < 0:
        resolved = total_rows + resolved
    if not 0 <= resolved < total_rows:
        raise ValueError(f"row index {index} out of range (0..{total_rows - 1})")
    return resolved


def build_common_parser(
    prog: str, description: str, path_help: str
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument("path", nargs="?", type=Path, help=path_help)
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=None,
        help="zero-based row index (negative counts from end; default = last row)",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="print pretty JSON of selected row and exit",
    )
    return parser


def run_source_cli(
    parser: argparse.ArgumentParser,
    source: RowSource,
    index: int | None,
    plain: bool,
    prog: str,
) -> int:
    if source.total_rows <= 0:
        parser.error(f"{source.display_path} has no rows")

    try:
        resolved_index = resolve_initial_index(index, source.total_rows)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        value = source.load_row(resolved_index)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    if plain:
        print_plain(value, sys.stdout)
        return 0

    app = AppState(
        source=source,
        prog=prog,
        yank_path=Path(f"/tmp/{prog}-yank.txt"),
        row_index=resolved_index,
        view=RowView(value=value),
    )
    return curses.wrapper(run_curses, app)


# ---------------------------------------------------------------------------
# Entrypoints
# ---------------------------------------------------------------------------


def main_jsonl(argv: Sequence[str] | None = None) -> int:
    parser = build_common_parser(
        prog="pcat-jsonl",
        description="Pretty interactive viewer for JSONL files or folders of JSON files (vim-style nav).",
        path_help="path to a .jsonl file or a directory of .json files",
    )
    args = parser.parse_args(argv)
    if args.path is None:
        parser.error("missing path. Example: pcat-jsonl data.jsonl")

    path = args.path.expanduser()
    if not path.exists():
        parser.error(f"path not found: {path}")

    if path.is_dir():
        source = JsonDirRowSource.from_path(path)
        if source.total_rows <= 0:
            parser.error(f"{path} has no .json files")
    elif path.name.lower().endswith(".json"):
        source = JsonSingleRowSource.from_path(path)
    else:
        source = JsonlRowSource.from_path(path)
        if source.total_rows <= 0:
            parser.error(f"{path} has no non-empty JSONL rows")
    return run_source_cli(parser, source, args.index, args.plain, prog="pcat-jsonl")


def main_hf_dataset(argv: Sequence[str] | None = None) -> int:
    parser = build_common_parser(
        prog="pcat-hf-dataset",
        description="Pretty interactive viewer for Hugging Face datasets saved with load_from_disk().",
        path_help="path to a dataset saved via datasets.load_from_disk()",
    )
    parser.add_argument(
        "--split",
        help="dataset split to open when load_from_disk() returns a DatasetDict",
    )
    args = parser.parse_args(argv)
    if args.path is None:
        parser.error("missing path. Example: pcat-hf-dataset ./my-dataset")

    path = args.path.expanduser()
    if not path.exists():
        parser.error(f"path not found: {path}")

    try:
        source = HFDatasetRowSource.from_path(path, split=args.split)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2

    if source.total_rows <= 0:
        parser.error(f"{source.display_path} has no rows")
    return run_source_cli(
        parser, source, args.index, args.plain, prog="pcat-hf-dataset"
    )
