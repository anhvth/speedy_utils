"""
pcat --serve: Lightweight web viewer for JSONL rows.

Start a local HTTP server that renders rows as interactive HTML.
Designed to be dependency-free (stdlib only), beautiful, and fast.

Modes:
  generic  – lazy foldable KV-card tree (works for any JSON dict)
  raw      – plain pretty-printed JSON dump
  sdd      – side-by-side messages | teacher_messages chat view
  tokens   – bounded SFT/DPO/KTO decoded text and token metadata
"""

from __future__ import annotations

import html
import json
import sys
import threading
import time
import webbrowser
from contextlib import suppress
from difflib import SequenceMatcher
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from ._shared import JsonlGlobRowSource, RowSource


_FILE_PICKER_CSS = """
.file-picker{max-width:900px;margin:0 auto;padding:24px 16px}
.picker-header{display:flex;align-items:center;gap:12px;margin-bottom:16px;position:sticky;top:0;background:var(--bg);padding:8px 0;z-index:5}
#picker-filter{flex:1;background:var(--elevated);border:1px solid var(--border);color:var(--text);padding:10px 14px;border-radius:8px;font-size:15px;font-family:var(--font);outline:none;transition:border-color .15s}
#picker-filter:focus{border-color:var(--accent)}
.match-count{color:var(--muted);font-size:13px;white-space:nowrap}
.file-list{display:flex;flex-direction:column;gap:2px}
.file-row{display:flex;align-items:center;gap:8px;padding:8px 12px;border-radius:6px;cursor:pointer;transition:background .1s;border:1px solid transparent}
.file-row:hover{background:rgba(255,255,255,0.04)}
.file-row.selected{background:rgba(88,166,255,0.12);border-color:var(--accent)}
.file-row.hidden{display:none}
.file-icon{font-size:16px;flex-shrink:0}
.file-name{font-weight:600;font-size:14px;color:var(--text);white-space:nowrap}
.file-dir{font-size:12px;color:var(--muted);margin-left:auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:50%}
"""


# ---------------------------------------------------------------------------
# CSS  –  inline, zero-dependency, dark theme
# ---------------------------------------------------------------------------

_CSS = (
    _FILE_PICKER_CSS
    + """
:root {
    --bg: #0a0e14; --surface: #121721; --elevated: #1a2030;
    --border: #252d3a; --border-active: #3b4455;
    --text: #cdd6e0; --muted: #6b7685; --accent: #58a6ff;
    --key: #e87980; --string: #a5d6ff; --number: #79c0ff;
    --bool: #ffa657; --null: #ff7b72; --punct: #6b7685;
    --bg-kv: rgba(255,255,255,0.015);
    --font: -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
    --mono: ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}
*,*::before,*::after{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:var(--font);
     font-size:14px;line-height:1.5;margin:0;min-height:100vh}

.header{position:sticky;top:0;z-index:10;background:var(--surface);
        border-bottom:1px solid var(--border);padding:8px 16px;
        display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.header .path{color:var(--muted);font-family:var(--mono);font-size:12px}
.header .row-info{color:var(--text);font-weight:600}
.nav{display:flex;gap:4px;align-items:center}
.nav button,.nav input,.mode-btn{background:var(--elevated);border:1px solid var(--border);
    color:var(--text);padding:4px 10px;border-radius:6px;font-size:12px;
    cursor:pointer;font-family:var(--font);transition:background .15s}
.nav button:hover,.mode-btn:hover{background:var(--border-active)}
.mode-btn.active{background:var(--accent);color:#fff;border-color:var(--accent)}
.nav input{width:64px;text-align:center;padding:4px 6px}
.nav input:focus{outline:none;border-color:var(--accent)}
.nav .sep{color:var(--muted);user-select:none}

.main{padding:12px 16px 80px;max-width:1400px;margin:0 auto}

/* lazy card */
.node-card{border:1px solid var(--border);border-radius:8px;
    margin-bottom:6px;overflow:hidden;background:var(--surface)}
.node-card .node-trigger{padding:8px 12px;cursor:pointer;
    display:flex;align-items:center;gap:8px;user-select:none;
    font-family:var(--mono);font-size:13px;transition:background .12s;
    border-bottom:1px solid transparent}
.node-card .node-trigger:hover{background:rgba(255,255,255,0.03)}
.node-card.open>.node-trigger{border-bottom-color:var(--border)}
.node-card .chevron{font-size:10px;color:var(--muted);flex-shrink:0;
    transition:transform .15s;width:14px;text-align:center;
    display:inline-block}
.node-card.open>.node-trigger .chevron,
.node-card .chevron.rotated{transform:rotate(90deg)}
.node-card .node-label{color:var(--key);font-weight:600;white-space:nowrap}
.node-card .node-summary{color:var(--muted);font-size:11px;margin-left:auto;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:40%}
.node-card .node-body{display:none;padding:6px 0}
.node-card.open>.node-body{display:block}
.node-card .node-body .kv-row{padding-left:24px}
.node-card .node-body .node-card{margin-left:12px}

.kv-row{display:flex;flex-wrap:wrap;align-items:flex-start;gap:4px;
    padding:5px 12px;font-family:var(--mono);font-size:13px;
    min-height:26px;border-left:2px solid transparent;
    transition:border-color .1s}
.kv-row:hover{border-left-color:var(--border-active);background:var(--bg-kv)}
.kv-row .kv-key{color:var(--key);font-weight:600;white-space:nowrap;min-width:80px}
.kv-row .kv-colon{color:var(--punct);margin-right:4px}
.kv-row .kv-value{flex:1;min-width:0;
    white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere}
.kv-row .kv-value.scalar-string{color:var(--string)}
.kv-row .kv-value.scalar-number{color:var(--number)}
.kv-row .kv-value.scalar-bool{color:var(--bool)}
.kv-row .kv-value.scalar-null{color:var(--null)}
.kv-row .kv-value.container{cursor:pointer;color:var(--accent)}
.kv-row .kv-value.container:hover{text-decoration:underline}

.val-string{color:var(--string)} .val-number{color:var(--number)}
.val-bool{color:var(--bool)} .val-null{color:var(--null)}

/* loading spinner */
.loading{display:inline-block;width:14px;height:14px;border:2px solid var(--border);
    border-top-color:var(--accent);border-radius:50%;animation:spin .6s linear infinite;
    margin-left:6px;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}

/* foldable long scalar text */
.scalar-fold{max-height:80px;overflow:hidden;position:relative}
.scalar-fold.expanded{max-height:none}
.scalar-fold .fold-overlay{display:block;position:absolute;bottom:0;left:0;right:0;
    height:30px;background:linear-gradient(transparent,var(--surface));
    cursor:pointer;text-align:center;line-height:30px;
    font-size:10px;color:var(--accent)}
.scalar-fold.expanded .fold-overlay{display:none}
.scalar-fold.cs::after{content:'... click to expand';display:block;
    padding:4px 0;font-size:10px;color:var(--accent);cursor:pointer}

/* sdd chat */
.side-by-side{display:flex;gap:12px}
.side-by-side>div{flex:1;min-width:0;overflow:auto}
.side-by-side h3{font-size:13px;font-weight:600;color:var(--muted);
    padding:4px 0 8px;border-bottom:1px solid var(--border);margin:0 0 8px}
.chat-msg{border:1px solid var(--border);border-radius:8px;margin-bottom:8px;
    overflow:hidden;background:var(--surface)}
.chat-msg.msg-assistant{border-color:#238636;background:#0d2818}
.chat-msg.msg-user{border-color:#6e5cce;background:#1a1535}
.chat-msg.msg-system{border-color:var(--border);background:var(--elevated)}
.msg-role{padding:6px 10px;font-size:11px;font-weight:700;text-transform:uppercase;
    color:var(--muted);border-bottom:1px solid var(--border);
    display:flex;align-items:center;gap:6px}
.msg-role .badge-role{border-radius:999px;padding:1px 8px;font-size:10px;
    background:var(--elevated);color:var(--text)}
.msg-content{white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;
    margin:0;padding:10px 12px;font-family:var(--font);font-size:13px;
    line-height:1.6;color:var(--text);background:transparent;border:none;
    overflow-x:auto;min-width:0}
.side-by-side .msg-content{color:#a5d6ff}
.side-by-side .chat-msg.msg-assistant .msg-content{color:#aff5b4}

/* tokenized sdd */
.tokenized-sdd{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.token-panel{min-width:0;overflow:hidden;border:1px solid var(--border);
    border-radius:8px;background:var(--surface)}
.token-panel h3{font-size:13px;font-weight:700;padding:8px 12px;margin:0;
    border-bottom:1px solid currentColor}
.token-panel .msg-content{font-family:var(--mono)}
.token-panel-student{border-color:#58a6ff;background:#0d2038}
.token-panel-student h3{color:#79c0ff}
.token-panel-student .msg-content,
.token-panel-teacher .msg-content{color:#a5d6ff}
.token-panel-teacher{border-color:#58a6ff;background:#0d2038}
.token-panel-teacher h3{color:#79c0ff}
.token-privileged{color:#e2c5ff;background:rgba(163,113,247,.3);
    border:1px solid rgba(210,168,255,.55);border-radius:3px;padding:0 2px}
.token-panel-response{grid-column:1/-1;border-color:#3fb950;background:#0d2818}
.token-panel-response h3{color:#56d364}
.token-panel-response .msg-content{color:#aff5b4}
.token-window-chunk{padding:0 0 14px;margin:0 0 14px;border-bottom:1px solid var(--border)}
.token-window-sentinel{padding:12px;text-align:center;color:var(--muted);font:11px var(--mono)}
.token-window-sentinel.loading{color:var(--accent)}.token-window-sentinel.done{display:none}
@media(max-width:800px){
    .tokenized-sdd{grid-template-columns:1fr}
    .token-panel-response{grid-column:auto}
}

/* tokenized training datasets */
.token-preview{display:flex;flex-direction:column;gap:12px}
.token-stats{display:flex;flex-wrap:wrap;gap:8px;position:sticky;top:55px;z-index:4;
    padding:8px;border:1px solid var(--border);border-radius:8px;background:rgba(18,23,33,.96)}
.stat-chip{padding:4px 9px;border:1px solid var(--border);border-radius:999px;
    color:var(--muted);font-family:var(--mono);font-size:11px}
.stat-chip strong{color:var(--text)}
.window-controls{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:8px;
    border:1px solid var(--border);border-radius:8px;background:var(--surface)}
.window-controls button,.window-controls input{background:var(--elevated);border:1px solid var(--border);
    color:var(--text);padding:5px 8px;border-radius:6px;font:11px var(--mono)}
.window-controls button{cursor:pointer}.window-controls button:hover{background:var(--border-active)}
.window-controls button.primary{background:var(--accent);color:#fff}.window-controls label{color:var(--muted);
    font:11px var(--mono)}.window-controls input{width:86px;margin-left:3px}
.window-status{margin-left:auto;color:var(--muted);font:11px var(--mono)}
.decoded-window,.token-table-wrap{border:1px solid var(--border);border-radius:8px;
    overflow:auto;background:var(--surface)}.decoded-window h3{margin:0;padding:7px 10px;
    border-bottom:1px solid var(--border);font-size:12px;color:var(--muted)}
.decoded-window pre{margin:0;padding:12px;white-space:pre-wrap;word-break:break-word;
    font:13px/1.6 var(--mono)}.token-decoded{border-radius:2px}
.region-masked{color:#8b949e}.region-trainable,.region-chosen,.region-encourage{color:#aff5b4}
.region-rejected,.region-discourage{color:#ffa198}.region-neutral{color:#aeb6c2}
.token-table{width:100%;border-collapse:collapse;font:11px var(--mono)}
.token-table th{position:sticky;top:0;background:var(--elevated);color:var(--muted);text-align:left}
.token-table th,.token-table td{padding:5px 8px;border-bottom:1px solid var(--border)}
.token-table .token-index,.token-table .token-id,.token-table .token-label{white-space:nowrap;
    color:var(--muted)}.token-table .token-piece{white-space:pre-wrap;word-break:break-all}
.token-table tr.region-trainable,.token-table tr.region-chosen,.token-table tr.region-encourage{
    background:rgba(35,134,54,.12)}.token-table tr.region-rejected,
.token-table tr.region-discourage{background:rgba(218,54,51,.12)}

.plain-dump{white-space:pre-wrap;word-break:break-word;overflow-wrap:anywhere;
    font-family:var(--mono);font-size:13px;color:var(--text);min-width:0}

.footer{position:fixed;bottom:0;left:0;right:0;background:var(--surface);
        border-top:1px solid var(--border);padding:6px 16px;
        font-size:12px;color:var(--muted);display:flex;gap:16px}
.keyboard-hint kbd{background:var(--elevated);border:1px solid var(--border);
    border-radius:3px;padding:1px 5px;font-family:var(--mono);
    font-size:11px;color:var(--text)}
"""
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_picker_page(
    files: list[Path], selected_idx: int = 0, glob_mode: bool = False
) -> str:
    rows = []
    for i, f in enumerate(files):
        parts = str(f).replace("\\\\", "/").split("/")
        rows.append(
            '<div class="file-row{selected}" data-idx="{idx}" data-path="{esc_path}"'
            ' onclick="selectFile({idx})">'
            '<span class="file-icon">{icon}</span>'
            '<span class="file-name">{name}</span>'
            '<span class="file-dir">{dir_part}</span>'
            "</div>".format(
                selected=" selected" if i == selected_idx else "",
                idx=i,
                esc_path=_esc_attr(str(f)),
                icon=chr(128196),
                name=_esc(f.name),
                dir_part=_esc(str(f.parent)) if len(parts) > 1 else "",
            )
        )
    glob_script = "<script>var GLOB_MODE = 1;</script>" if glob_mode else ""
    return (
        '<div class="file-picker">' + glob_script + '<div class="picker-header">'
        '<input id="picker-filter" type="text" placeholder="type to filter files..."'
        ' autofocus oninput="filterFiles()"'
        ' onkeydown="pickerKeyDown(event)">'
        '<span class="match-count" id="match-count">{total} files</span>'
        "</div>"
        '<div class="file-list" id="file-list">'
        "{rows}"
        "</div>"
        "</div>"
    ).format(total=len(files), rows="\n".join(rows))


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _esc_attr(text: str) -> str:
    return html.escape(text, quote=True)


def _esc_js(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


def _scalar_class(value: Any) -> str:
    if value is None:
        return "scalar-null"
    if isinstance(value, bool):
        return "scalar-bool"
    if isinstance(value, (int, float)):
        return "scalar-number"
    if isinstance(value, str):
        return "scalar-string"
    return ""


def _scalar_html(value: Any) -> str:
    """Render a scalar value as HTML. No truncation — full value shown."""
    if value is None:
        return '<span class="val-null">null</span>'
    if isinstance(value, bool):
        return '<span class="val-bool">{}</span>'.format(str(value).lower())
    if isinstance(value, (int, float)):
        return '<span class="val-number">{}</span>'.format(json.dumps(value))
    if isinstance(value, str):
        return '<span class="val-string">{}</span>'.format(html.escape(value))
    return '<span class="val-string">{}</span>'.format(
        html.escape(json.dumps(value, ensure_ascii=False))
    )


def _scalar_value_html(value: Any) -> str:
    """Render scalar into a kv-value span, with fold-on-overflow for long strings."""
    if value is None:
        return '<span class="kv-value scalar-null">null</span>'
    if isinstance(value, (int, float)):
        return '<span class="kv-value scalar-number">{}</span>'.format(
            json.dumps(value)
        )
    if isinstance(value, bool):
        return '<span class="kv-value scalar-bool">{}</span>'.format(str(value).lower())
    if isinstance(value, str):
        s = html.escape(value)
        if len(s) > 300:
            return (
                '<span class="kv-value scalar-string scalar-fold">'
                '<span class="fold-overlay" onclick="event.stopPropagation();'
                "this.parentElement.classList.add('expanded')\"></span>"
                "{}</span>"
            ).format(s)
        return '<span class="kv-value scalar-string">{}</span>'.format(s)
    s = html.escape(json.dumps(value, ensure_ascii=False))
    return '<span class="kv-value scalar-string">{}</span>'.format(s)


def _dict_summary(d: dict) -> str:
    n = len(d)
    keys = list(d.keys())
    preview = ", ".join(str(k) for k in keys[:4])
    if n > 4:
        preview += ", …"
    return "({}) {{{}}}".format(n, _esc(preview))


def _list_summary(lst: list) -> str:
    return "({} items)".format(len(lst))


# ---------------------------------------------------------------------------
# JSON path resolver
# ---------------------------------------------------------------------------


def _resolve_json_path(root: Any, path: str) -> Any:
    """Walk a dotted JSON path like '/messages/0/content'."""
    if path in ("$", "/", ""):
        return root, ""
    p = path.lstrip("$").lstrip("/")
    if not p:
        return root, ""
    cur: Any = root
    parts: list[str] = []
    for seg in p.split("/"):
        seg = seg.strip()
        if seg == "":
            continue
        try:
            if isinstance(cur, dict) and seg in cur:
                parts.append(seg)
                cur = cur[seg]
            elif isinstance(cur, list):
                idx = int(seg)
                parts.append("[{}]".format(idx))
                cur = cur[idx]
            elif isinstance(cur, dict):
                idx = int(seg)
                if idx in cur:
                    parts.append(str(idx))
                    cur = cur[idx]
                else:
                    parts.append(seg)
                    cur = None
                    break
            else:
                cur = None
                break
        except (KeyError, IndexError, ValueError, TypeError):
            cur = None
            break
    breadcrumb = "/".join(parts)
    return cur, breadcrumb


# ---------------------------------------------------------------------------
# Lazy generic mode: shallow skeleton + AJAX load-on-expand
# ---------------------------------------------------------------------------


def _render_node_shallow(value: Any, row_idx: int, path: str) -> str:
    if isinstance(value, dict):
        return _dict_rows_shallow(value, row_idx, path)
    if isinstance(value, list):
        return _list_rows_shallow(value, row_idx, path)
    return _scalar_html(value)


def _dict_rows_shallow(d: dict, row_idx: int, parent_path: str) -> str:
    rows: list[str] = []
    for k, v in d.items():
        seg = _esc_attr(str(k))
        child_path = "{}/{}".format(parent_path, seg)
        if isinstance(v, (dict, list)):
            summary = _dict_summary(v) if isinstance(v, dict) else _list_summary(v)
            rows.append(
                '<div class="kv-row">'
                '<span class="kv-key">{}</span><span class="kv-colon">:</span>'
                "<span class=\"kv-value container\" onclick='expandNode(event,{},{})' "
                'title="click to expand">{}</span>'
                "</div>".format(
                    _esc(str(k)),
                    row_idx,
                    _esc_attr(json.dumps(child_path)),
                    summary,
                )
            )
        else:
            rows.append(
                '<div class="kv-row">'
                '<span class="kv-key">{}</span><span class="kv-colon">:</span>'
                "{}</div>".format(_esc(str(k)), _scalar_value_html(v))
            )
    return "\n".join(rows)


def _list_rows_shallow(lst: list, row_idx: int, parent_path: str) -> str:
    rows: list[str] = []
    for i, item in enumerate(lst):
        child_path = "{}/{}".format(parent_path, i)
        if isinstance(item, (dict, list)):
            summary = (
                _dict_summary(item) if isinstance(item, dict) else _list_summary(item)
            )
            rows.append(
                '<div class="kv-row">'
                '<span class="kv-key">[{}]</span><span class="kv-colon">:</span>'
                "<span class=\"kv-value container\" onclick='expandNode(event,{},{})' "
                'title="click to expand">{}</span>'
                "</div>".format(i, row_idx, _esc_attr(json.dumps(child_path)), summary)
            )
        else:
            rows.append(
                '<div class="kv-row">'
                '<span class="kv-key">[{}]</span><span class="kv-colon">:</span>'
                "{}</div>".format(i, _scalar_value_html(item))
            )
    return "\n".join(rows)


def _render_node_children(root: Any, path: str) -> str:
    """Server-side: resolve *path* in *root*, return children HTML fragment."""
    value, _breadcrumb = _resolve_json_path(root, path)
    if value is None:
        return '<div class="kv-row"><span class="val-null">(not found)</span></div>'
    if isinstance(value, dict):
        return _dict_rows_flat(value)
    if isinstance(value, list):
        return _list_rows_flat(value)
    return _scalar_value_html(value)


def _dict_rows_flat(d: dict) -> str:
    if not d:
        return '<span class="val-null">{}</span>'.format("{}")
    rows = []
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            summary = _dict_summary(v) if isinstance(v, dict) else _list_summary(v)
            rows.append(
                '<div class="kv-row">'
                '<span class="kv-key">{}</span><span class="kv-colon">:</span>'
                '<span class="kv-value container">{}</span>'
                "</div>".format(_esc(str(k)), summary)
            )
        else:
            rows.append(
                '<div class="kv-row">'
                '<span class="kv-key">{}</span><span class="kv-colon">:</span>'
                "{}</div>".format(_esc(str(k)), _scalar_value_html(v))
            )
    return "\n".join(rows)


def _list_rows_flat(lst: list) -> str:
    if not lst:
        return '<span class="val-null">{}</span>'.format("[]")
    rows = []
    for i, item in enumerate(lst):
        if isinstance(item, (dict, list)):
            summary = (
                _dict_summary(item) if isinstance(item, dict) else _list_summary(item)
            )
            rows.append(
                '<div class="kv-row">'
                '<span class="kv-key">[{}]</span><span class="kv-colon">:</span>'
                '<span class="kv-value container">{}</span>'
                "</div>".format(i, summary)
            )
        else:
            rows.append(
                '<div class="kv-row">'
                '<span class="kv-key">[{}]</span><span class="kv-colon">:</span>'
                "{}</div>".format(i, _scalar_value_html(item))
            )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Mode registry
# ---------------------------------------------------------------------------

_MODE_REGISTRY: dict[str, Any] = {}


def register_mode(name: str):
    def decorator(fn):
        _MODE_REGISTRY[name] = fn
        return fn

    return decorator


_DEFAULT_SDD_TOKENIZER = "Qwen/Qwen3.5-27B"
_DEFAULT_TOKEN_WINDOW = 256
_MAX_TOKEN_WINDOW = 1024
_TOKENIZED_SCHEMAS = {
    "sft": {"input_ids", "labels"},
    "dpo": {"prompt_ids", "chosen_ids", "rejected_ids"},
    "kto": {"prompt_ids", "completion_ids", "encourage_label"},
}
_tokenizer_cache: dict[str, Any] = {}


def _get_tokenizer(tokenizer_name: str) -> Any:
    tokenizer = _tokenizer_cache.get(tokenizer_name)
    if tokenizer is not None:
        return tokenizer
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required to decode tokenized PCAT rows; install "
            "transformers or select generic/raw mode"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    _tokenizer_cache[tokenizer_name] = tokenizer
    return tokenizer


def _is_tokenized_sdd(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("format") == "sdd_prompt_response_v1"
        and all(
            isinstance(value.get(key), list)
            for key in ("student_ids", "teacher_ids", "response_ids")
        )
    )


def _training_schema(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    keys = set(value)
    for name in ("kto", "dpo", "sft"):
        if _TOKENIZED_SCHEMAS[name] <= keys:
            return name
    return None


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode(token_ids))


def _training_token_data(
    value: dict[str, Any], schema: str
) -> tuple[list[int], list[int | None], list[str]]:
    if schema == "sft":
        ids = [int(token_id) for token_id in value["input_ids"]]
        labels: list[int | None] = [int(label) for label in value["labels"]]
        if len(ids) != len(labels):
            raise ValueError("SFT input_ids and labels must have the same length")
        regions = ["masked" if label == -100 else "trainable" for label in labels]
        return ids, labels, regions
    if schema == "dpo":
        prompt = [int(token_id) for token_id in value["prompt_ids"]]
        chosen = [int(token_id) for token_id in value["chosen_ids"]]
        rejected = [int(token_id) for token_id in value["rejected_ids"]]
        ids = prompt + chosen + rejected
        regions = (
            ["masked"] * len(prompt)
            + ["chosen"] * len(chosen)
            + ["rejected"] * len(rejected)
        )
        return ids, [None] * len(ids), regions
    label = int(value["encourage_label"])
    completion_region = {1: "encourage", 0: "neutral", -1: "discourage"}.get(label)
    if completion_region is None:
        raise ValueError(
            "KTO encourage_label must be -1, 0, or 1; got {}".format(label)
        )
    prompt = [int(token_id) for token_id in value["prompt_ids"]]
    completion = [int(token_id) for token_id in value["completion_ids"]]
    ids = prompt + completion
    regions = ["masked"] * len(prompt) + [completion_region] * len(completion)
    return ids, [None] * len(ids), regions


def _region_boundaries(regions: list[str]) -> list[dict[str, Any]]:
    boundaries = []
    previous = None
    for offset, region in enumerate(regions):
        if region == previous:
            continue
        boundaries.append({"offset": offset, "region": region})
        previous = region
    return boundaries


def _training_stats(
    value: dict[str, Any], row_idx: int, schema: str | None = None
) -> dict[str, Any]:
    schema = schema or _training_schema(value)
    if schema is None:
        raise ValueError("row is not a recognized tokenized training schema")
    token_ids, _labels, regions = _training_token_data(value, schema)
    counts: dict[str, int] = {}
    for region in regions:
        counts[region] = counts.get(region, 0) + 1
    return {
        "schema": schema,
        "row_index": row_idx,
        "id": value.get("id"),
        "total_tokens": len(token_ids),
        "counts": counts,
        "boundaries": _region_boundaries(regions),
    }


def _normalize_token_window(total: int, offset: int, limit: int) -> tuple[int, int]:
    if total <= 0:
        return 0, 0
    offset = max(0, min(int(offset), total - 1))
    limit = max(1, min(int(limit), _MAX_TOKEN_WINDOW))
    return offset, min(total, offset + limit)


def _parse_render_query(query: str, default_mode: str) -> tuple[str, int, int]:
    params = parse_qs(query)
    requested_mode = params.get("mode", [default_mode])[0]
    allowed_modes = {"auto", "generic", "raw", "sdd", "tokens"}
    mode = requested_mode if requested_mode in allowed_modes else "generic"
    try:
        offset = int(params.get("offset", ["0"])[0])
        limit = int(params.get("limit", [str(_DEFAULT_TOKEN_WINDOW)])[0])
    except ValueError as exc:
        raise ValueError("offset and limit must be integers") from exc
    return mode, offset, limit


def _training_window(
    value: dict[str, Any],
    tokenizer: Any,
    *,
    offset: int = 0,
    limit: int = _DEFAULT_TOKEN_WINDOW,
) -> dict[str, Any]:
    schema = _training_schema(value)
    if schema is None:
        raise ValueError("row is not a recognized tokenized training schema")
    token_ids, labels, regions = _training_token_data(value, schema)
    start, end = _normalize_token_window(len(token_ids), offset, limit)
    segments = []
    if start < end:
        segment_start = start
        current_region = regions[start]
        for index in range(start + 1, end):
            if regions[index] == current_region:
                continue
            segments.append(
                {
                    "start": segment_start,
                    "end": index,
                    "region": current_region,
                    "text": _decode_tokens(tokenizer, token_ids[segment_start:index]),
                }
            )
            segment_start = index
            current_region = regions[index]
        segments.append(
            {
                "start": segment_start,
                "end": end,
                "region": current_region,
                "text": _decode_tokens(tokenizer, token_ids[segment_start:end]),
            }
        )
    items = [
        {
            "index": index,
            "token_id": token_ids[index],
            "label": labels[index],
            "region": regions[index],
            "text": _decode_tokens(tokenizer, [token_ids[index]]).replace("\n", "\\n"),
        }
        for index in range(start, end)
    ]
    return {
        "schema": schema,
        "offset": start,
        "end": end,
        "total_tokens": len(token_ids),
        "segments": segments,
        "items": items,
    }


def _render_window_controls(
    total: int,
    offset: int,
    end: int,
    limit: int,
    boundaries: list[dict[str, Any]],
) -> str:
    buttons = ['<button onclick="jumpToken(0)">start</button>']
    for boundary in boundaries[1:]:
        boundary_offset = int(boundary["offset"])
        buttons.append(
            '<button onclick="jumpToken({})">{} start · {:,}</button>'.format(
                boundary_offset,
                _esc(str(boundary["region"])),
                boundary_offset,
            )
        )
    end_offset = max(0, total - max(1, min(limit, _MAX_TOKEN_WINDOW)))
    buttons.append('<button onclick="jumpToken({})">end</button>'.format(end_offset))
    return (
        '<div class="window-controls">{}'
        '<label>offset <input id="token-offset" type="number" min="0" max="{}" '
        'value="{}"></label>'
        '<label>window <input id="token-limit" type="number" min="1" max="{}" '
        'value="{}"></label>'
        '<button class="primary" onclick="jumpTokenFromInput()">jump</button>'
        '<span class="window-status">showing {:,}–{:,} of {:,}</span></div>'
    ).format(
        "".join(buttons),
        max(0, total - 1),
        offset,
        _MAX_TOKEN_WINDOW,
        min(max(1, limit), _MAX_TOKEN_WINDOW),
        offset,
        max(offset, end - 1),
        total,
    )


def _render_training(
    value: dict[str, Any],
    tokenizer: Any,
    *,
    row_idx: int,
    offset: int,
    limit: int,
) -> str:
    schema = _training_schema(value)
    if schema is None:
        return _render_generic(value, 0)
    stats = _training_stats(value, row_idx, schema)

    chips = [
        '<span class="stat-chip">type <strong>{}</strong></span>'.format(
            schema.upper()
        ),
        '<span class="stat-chip">row <strong>{:,}</strong></span>'.format(row_idx + 1),
        '<span class="stat-chip">total <strong>{:,}</strong></span>'.format(
            stats["total_tokens"]
        ),
    ]
    for region, count in stats["counts"].items():
        chips.append(
            '<span class="stat-chip">{} <strong>{:,}</strong></span>'.format(
                region, count
            )
        )
    if value.get("id") is not None:
        chips.append(
            '<span class="stat-chip">id <strong>{}</strong></span>'.format(
                _esc(str(value["id"]))
            )
        )
    chunk, start, end, total = _render_training_chunk(
        value, tokenizer, offset=offset, limit=limit
    )
    controls = _render_window_controls(
        total,
        start,
        end,
        limit,
        stats["boundaries"],
    )
    return (
        '<div class="token-preview"><div class="token-stats">{}</div>{}'
        '<div id="token-window-stream" data-row="{}" data-next-offset="{}" '
        'data-total="{}" data-limit="{}">{}</div>'
        '<div id="token-window-sentinel" class="token-window-sentinel">'
        "scroll to load more</div></div>"
    ).format(
        "".join(chips),
        controls,
        row_idx + 1,
        end,
        total,
        min(max(1, limit), _MAX_TOKEN_WINDOW),
        chunk,
    )


def _render_training_chunk(
    value: dict[str, Any],
    tokenizer: Any,
    *,
    offset: int,
    limit: int,
) -> tuple[str, int, int, int]:
    window = _training_window(value, tokenizer, offset=offset, limit=limit)
    decoded = "".join(
        '<span class="token-decoded region-{}" data-start="{}" data-end="{}">{}</span>'.format(
            _esc_attr(segment["region"]),
            segment["start"],
            segment["end"],
            _esc(segment["text"]),
        )
        for segment in window["segments"]
    )
    rows = "".join(
        '<tr class="region-{}"><td class="token-index">{:,}</td>'
        '<td class="token-id">{}</td><td class="token-label">{}</td>'
        '<td class="token-region-name">{}</td><td class="token-piece">{}</td></tr>'.format(
            _esc_attr(item["region"]),
            item["index"],
            item["token_id"],
            "—" if item["label"] is None else item["label"],
            _esc(item["region"]),
            _esc(item["text"]),
        )
        for item in window["items"]
    )
    chunk = (
        '<section class="token-window-chunk" data-offset="{}" '
        'data-next-offset="{}" data-total="{}">'
        '<section class="decoded-window"><h3>decoded text</h3><pre>{}</pre></section>'
        '<div class="token-table-wrap"><table class="token-table"><thead><tr>'
        "<th>index</th><th>token ID</th><th>label</th><th>region</th><th>piece</th>"
        "</tr></thead><tbody>{}</tbody></table></div></section>".format(
            window["offset"],
            window["end"],
            window["total_tokens"],
            decoded,
            rows,
        )
    )
    return chunk, window["offset"], window["end"], window["total_tokens"]


def _highlight_teacher_privilege(student_text: str, teacher_text: str) -> str:
    """Escape teacher text and highlight spans absent or changed from the student."""
    parts: list[str] = []
    matcher = SequenceMatcher(None, student_text, teacher_text, autojunk=False)
    for (
        tag,
        _student_start,
        _student_end,
        teacher_start,
        teacher_end,
    ) in matcher.get_opcodes():
        if teacher_start == teacher_end:
            continue
        escaped = html.escape(teacher_text[teacher_start:teacher_end])
        if tag == "equal":
            parts.append(escaped)
        else:
            parts.append('<mark class="token-privileged">{}</mark>'.format(escaped))
    return "".join(parts)


def render_row(
    value: Any,
    row_idx: int,
    mode: str = "auto",
    *,
    tokenizer: Any = None,
    token_offset: int = 0,
    token_limit: int = _DEFAULT_TOKEN_WINDOW,
) -> tuple[str, str]:
    """Render a row value as HTML. Returns (html, resolved_mode)."""
    if mode == "auto":
        mode = _detect_mode(value)
    if mode in _MODE_REGISTRY:
        if mode == "sdd":
            return (
                _render_sdd(
                    value,
                    row_idx,
                    tokenizer=tokenizer,
                    offset=token_offset,
                    limit=token_limit,
                ),
                mode,
            )
        if mode == "tokens":
            if _training_schema(value) is None:
                return _render_generic(value, row_idx), "generic"
            if tokenizer is None:
                tokenizer = _get_tokenizer(_DEFAULT_SDD_TOKENIZER)
            return (
                _render_training(
                    value,
                    tokenizer,
                    row_idx=row_idx,
                    offset=token_offset,
                    limit=token_limit,
                ),
                mode,
            )
        return _MODE_REGISTRY[mode](value, row_idx), mode
    return _render_generic(value, row_idx), "generic"


def _detect_mode(value: Any) -> str:
    if _is_tokenized_sdd(value):
        return "sdd"
    if isinstance(value, dict) and (
        (
            isinstance(value.get("messages"), list)
            and isinstance(value.get("teacher_messages"), list)
        )
        or isinstance(value.get("messages_with_ref"), list)
    ):
        return "sdd"
    if _training_schema(value) is not None:
        return "tokens"
    return "generic"


@register_mode("generic")
def _render_generic(value: Any, row_idx: int) -> str:
    if isinstance(value, dict):
        if not value:
            return '<span class="val-null">{}</span>'.format("{}")
        body = _render_node_shallow(value, row_idx, "$")
        return '<div class="node-card open" id="card-$"><div class="node-body" style="display:block">{}</div></div>'.format(
            body
        )
    return '<div class="node-body">{}</div>'.format(
        _render_node_shallow(value, row_idx, "$")
    )


@register_mode("raw")
def _render_raw(value: Any, _row_idx: int = 0) -> str:
    s = json.dumps(value, ensure_ascii=False, indent=2)
    return '<div class="plain-dump">{}</div>'.format(html.escape(s))


@register_mode("tokens")
def _render_tokens(value: Any, _row_idx: int = 0) -> str:
    # Tokenizer injection is handled by render_row, mirroring the SDD renderer.
    return _render_generic(value, _row_idx)


@register_mode("sdd")
def _render_sdd(
    value: Any,
    _row_idx: int = 0,
    *,
    tokenizer: Any = None,
    offset: int = 0,
    limit: int = _DEFAULT_TOKEN_WINDOW,
) -> str:
    if _is_tokenized_sdd(value):
        if tokenizer is None:
            tokenizer = _get_tokenizer(_DEFAULT_SDD_TOKENIZER)
        chunk, start, end, total = _render_token_window_chunk(
            value, tokenizer, offset=offset, limit=limit
        )
        controls = _render_window_controls(total, start, end, limit, [])
        return (
            '{}<div id="token-window-stream" data-row="{}" data-next-offset="{}" '
            'data-total="{}" data-limit="{}">{}</div>'
            '<div id="token-window-sentinel" class="token-window-sentinel">'
            "scroll to load more</div>"
        ).format(
            controls,
            _row_idx + 1,
            end,
            total,
            min(max(1, limit), _MAX_TOKEN_WINDOW),
            chunk,
        )

    messages = value.get("messages", [])
    teacher = value.get("teacher_messages", [])
    messages_with_ref = value.get("messages_with_ref")
    if isinstance(messages_with_ref, list):
        messages = []
        teacher = []
        for message in messages_with_ref:
            if not isinstance(message, dict):
                continue
            student_message = {
                key: item for key, item in message.items() if key != "env_feedback"
            }
            teacher_message = dict(student_message)
            reference = message.get("env_feedback")
            if isinstance(reference, str) and reference.strip():
                original = teacher_message.get("content") or ""
                if not isinstance(original, str):
                    original = json.dumps(original, ensure_ascii=False)
                injection = "<reference_answer>\n{}\n</reference_answer>".format(
                    reference.strip()
                )
                teacher_message["content"] = (
                    original + "\n\n" + injection if original else injection
                )
            messages.append(student_message)
            teacher.append(teacher_message)

    def _chat_bubble(msg: dict, i: int, compare_msg: Any = None) -> str:
        role = str(msg.get("role", "?"))
        content = msg.get("content", "")
        if isinstance(content, str):
            compare_content = (
                compare_msg.get("content") if isinstance(compare_msg, dict) else None
            )
            rendered_content = (
                _highlight_teacher_privilege(compare_content, content)
                if isinstance(compare_content, str)
                else html.escape(content)
            )
            content_html = '<pre class="msg-content">{}</pre>'.format(rendered_content)
        else:
            content_html = _scalar_html(content)
        extras = {
            key: item for key, item in msg.items() if key not in {"role", "content"}
        }
        extras_html = ""
        if extras:
            extras_html = '<pre class="msg-content">{}</pre>'.format(
                html.escape(json.dumps(extras, ensure_ascii=False, indent=2))
            )
        return (
            '<div class="chat-msg msg-{}">'
            '<div class="msg-role">[{}] <span class="badge-role">{}</span></div>'
            "{}{} </div>"
        ).format(role, i, _esc(role), content_html, extras_html)

    def _render_msgs(msgs: list, title: str, compare_msgs: list | None = None) -> str:
        parts = ["<h3>{} ({})</h3>".format(_esc(title), len(msgs))]
        for i, msg in enumerate(msgs):
            if isinstance(msg, dict):
                compare_msg = (
                    compare_msgs[i]
                    if compare_msgs is not None and i < len(compare_msgs)
                    else None
                )
                parts.append(_chat_bubble(msg, i, compare_msg))
        return "".join(parts)

    return '<div class="side-by-side"><div>{}</div><div>{}</div></div>'.format(
        _render_msgs(messages, "messages"),
        _render_msgs(
            teacher,
            "teacher_messages",
            messages if len(messages) == len(teacher) else None,
        ),
    )


def _render_tokenized_sdd_chunk(
    value: dict[str, Any],
    tokenizer: Any,
    *,
    offset: int,
    limit: int,
) -> tuple[str, int, int, int]:
    keys = ("student_ids", "teacher_ids", "response_ids")
    token_data = {key: [int(token_id) for token_id in value[key]] for key in keys}
    total = max((len(token_data[key]) for key in keys), default=0)
    chunk_start, chunk_end = _normalize_token_window(total, offset, limit)
    windows = {}
    decoded_text = {}
    for key in keys:
        token_ids = token_data[key]
        start = min(chunk_start, len(token_ids))
        end = min(chunk_end, len(token_ids))
        windows[key] = (start, end, len(token_ids))
        decoded_text[key] = _decode_tokens(tokenizer, token_ids[start:end])
    panels = []
    for key, panel_name in (
        ("student_ids", "student"),
        ("teacher_ids", "teacher"),
        ("response_ids", "response"),
    ):
        if key == "teacher_ids":
            content = _highlight_teacher_privilege(
                decoded_text["student_ids"], decoded_text[key]
            )
        else:
            content = html.escape(decoded_text[key])
        panels.append(
            '<div class="token-panel token-panel-{}"><h3>{} '
            "(tokens {:,}–{:,} of {:,})</h3>"
            '<pre class="msg-content">{}</pre></div>'.format(
                panel_name,
                key,
                windows[key][0],
                max(windows[key][0], windows[key][1] - 1),
                windows[key][2],
                content,
            )
        )
    chunk = (
        '<section class="token-window-chunk" data-offset="{}" '
        'data-next-offset="{}" data-total="{}">'
        '<div class="tokenized-sdd">{}</div></section>'
    ).format(chunk_start, chunk_end, total, "".join(panels))
    return chunk, chunk_start, chunk_end, total


def _render_token_window_chunk(
    value: dict[str, Any],
    tokenizer: Any,
    *,
    offset: int,
    limit: int,
) -> tuple[str, int, int, int]:
    """Render the shared infinite-scroll chunk contract for any tokenized row."""
    if _is_tokenized_sdd(value):
        return _render_tokenized_sdd_chunk(value, tokenizer, offset=offset, limit=limit)
    if _training_schema(value) is not None:
        return _render_training_chunk(value, tokenizer, offset=offset, limit=limit)
    raise ValueError("row is not a recognized tokenized schema")


# ---------------------------------------------------------------------------
# JavaScript  –  inline, zero-dependency
# ---------------------------------------------------------------------------

_JS = r"""
var currentMode = document.body.dataset.mode || 'generic';
var selectedFileIdx = -1;
var tokenWindowObserver = null;
var tokenWindowLoading = false;
var MAX_TOKEN_WINDOW_CHUNKS = 8;

function selectFile(idx) {
    var rows = document.querySelectorAll('.file-row');
    var path = rows[idx] ? rows[idx].dataset.path : '';
    if (path) {
        window.location.href = '/file/' + idx;
    }
}

function filterFiles() {
    var q = (document.getElementById('picker-filter') || {}).value || '';
    var lower = q.toLowerCase();
    var rows = document.querySelectorAll('.file-row');
    var count = 0;
    var firstVisible = -1;
    selectedFileIdx = -1;
    rows.forEach(function(r) {
        var name = (r.querySelector('.file-name') || {}).textContent || '';
        var dir = (r.querySelector('.file-dir') || {}).textContent || '';
        var match = !lower || name.toLowerCase().indexOf(lower) >= 0 || dir.toLowerCase().indexOf(lower) >= 0;
        r.classList.toggle('hidden', !match);
        r.classList.remove('selected');
        if (match) {
            count++;
            if (firstVisible < 0) firstVisible = parseInt(r.dataset.idx);
        }
    });
    var mc = document.getElementById('match-count');
    if (mc) mc.textContent = count + ' / ' + rows.length + ' files';
    if (firstVisible >= 0) {
        rows[firstVisible] && rows[firstVisible].classList.add('selected');
        selectedFileIdx = firstVisible;
    }
}

function pickerKeyDown(e) {
    var rows = Array.from(document.querySelectorAll('.file-row')).filter(function(r) { return !r.classList.contains('hidden'); });
    if (rows.length === 0) return;
    var cur = rows.findIndex(function(r) { return r.classList.contains('selected'); });
    var next = cur;
    if (e.key === 'ArrowDown' || (e.key === 'j' && e.ctrlKey)) {
        e.preventDefault();
        next = Math.min(cur + 1, rows.length - 1);
    } else if (e.key === 'ArrowUp' || (e.key === 'k' && e.ctrlKey)) {
        e.preventDefault();
        next = Math.max(cur - 1, 0);
    } else if (e.key === 'Enter') {
        e.preventDefault();
        if (cur >= 0 && rows[cur]) {
            selectFile(parseInt(rows[cur].dataset.idx));
        }
        return;
    } else {
        return;
    }
    if (next !== cur && next >= 0) {
        rows.forEach(function(r) { r.classList.remove('selected'); });
        rows[next].classList.add('selected');
        rows[next].scrollIntoView({block: 'nearest'});
        selectedFileIdx = parseInt(rows[next].dataset.idx);
    }
}

function expandNode(event, rowIdx, path) {
    event.stopPropagation();
    var kvValue = event.currentTarget;
    var existingCard = kvValue.closest('.node-card');
    var card = null;
    var body = null;

    if (existingCard && existingCard.parentNode && existingCard.parentNode.classList.contains('kv-value')) {
        card = existingCard;
        body = card.querySelector('.node-body');
    } else {
        var kvRow = kvValue.closest('.kv-row');
        card = document.createElement('div');
        card.className = 'node-card';
        body = document.createElement('div');
        body.className = 'node-body';
        card.appendChild(body);
        kvRow.parentNode.insertBefore(card, kvRow.nextSibling);
        body.appendChild(kvRow);
    }

    if (card.classList.contains('open')) {
        card.classList.remove('open');
        body.style.display = 'none';
        return;
    }

    if (body.dataset.loaded === '1') {
        card.classList.add('open');
        body.style.display = 'block';
        return;
    }

    body.innerHTML = body.innerHTML + '<div class="loading"></div>';
    card.classList.add('open');
    body.style.display = 'block';

    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/row/' + rowIdx + '/node?path=' + encodeURIComponent(path));
    xhr.onload = function() {
        if (xhr.status === 200) {
            body.innerHTML = xhr.responseText;
            body.dataset.loaded = '1';
        } else {
            body.innerHTML = '<div class="kv-row"><span class="val-null">Error loading</span></div>';
        }
    };
    xhr.onerror = function() {
        body.innerHTML = '<div class="kv-row"><span class="val-null">Network error</span></div>';
    };
    xhr.send();
}

function switchMode(mode) {
    currentMode = mode;
    // Update button states
    var btns = document.querySelectorAll('.mode-btn');
    btns.forEach(function(b) {
        b.classList.toggle('active', b.dataset.mode === mode);
    });
    // Reload current row with new mode
    reloadContent(mode);
}

function reloadContent(mode) {
    var p = window.location.pathname.split('/');
    var rowNum = p[p.length-1] || '1';
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/row/' + rowNum + '?mode=' + (mode || currentMode));
    xhr.onload = function() {
        if (xhr.status === 200) {
            var parser = new DOMParser();
            var doc = parser.parseFromString(xhr.responseText, 'text/html');
            var newMain = doc.querySelector('.main');
            var oldMain = document.querySelector('.main');
            if (newMain && oldMain) {
                oldMain.innerHTML = newMain.innerHTML;
            }
            // Update row info in header
            var newInfo = doc.querySelector('.row-info');
            var oldInfo = document.querySelector('.row-info');
            if (newInfo && oldInfo) oldInfo.textContent = newInfo.textContent;
            // Update active mode button
            var btns = document.querySelectorAll('.mode-btn');
            btns.forEach(function(b) {
                b.classList.toggle('active', b.dataset.mode === (mode || currentMode));
            });
            initTokenWindowScroll();
        }
    };
    xhr.send();
}

function gotoRow(){
    var n=document.getElementById('goto-input').value;
    if(n==='')return;
    navigateTo(parseInt(n));
}
function navigate(d){
    var p=window.location.pathname.split('/');
    var c=parseInt(p[p.length-1])||1;
    var n=c+d;if(n<1)return;
    navigateTo(n);
}
function sampleRow(total){
    if(total<=1){navigateTo(1);return;}
    var p=window.location.pathname.split('/');
    var current=parseInt(p[p.length-1])||1;
    var sampled=current;
    while(sampled===current){sampled=Math.floor(Math.random()*total)+1;}
    navigateTo(sampled);
}
function navigateTo(n) {
    window.location.href = '/row/' + n + '?mode=' + currentMode;
}
function jumpToken(offset) {
    var url = new URL(window.location.href);
    url.searchParams.set('mode', currentMode);
    url.searchParams.set('offset', String(Math.max(0, Number(offset) || 0)));
    var limit = document.getElementById('token-limit');
    if (limit) url.searchParams.set('limit', limit.value);
    window.location.href = url.pathname + '?' + url.searchParams.toString();
}
function jumpTokenFromInput() {
    var input = document.getElementById('token-offset');
    if (input) jumpToken(input.value);
}
function initTokenWindowScroll() {
    if (tokenWindowObserver) tokenWindowObserver.disconnect();
    tokenWindowObserver = null;
    tokenWindowLoading = false;
    var stream = document.getElementById('token-window-stream');
    var sentinel = document.getElementById('token-window-sentinel');
    if (!stream || !sentinel) return;
    function updateDone() {
        var done = Number(stream.dataset.nextOffset) >= Number(stream.dataset.total);
        sentinel.classList.toggle('done', done);
        return done;
    }
    async function loadNextTokenWindow() {
        if (tokenWindowLoading || updateDone()) return;
        tokenWindowLoading = true;
        sentinel.classList.add('loading');
        sentinel.textContent = 'loading next token window…';
        var params = new URLSearchParams({
            offset: stream.dataset.nextOffset,
            limit: stream.dataset.limit
        });
        try {
            var response = await fetch('/row/' + stream.dataset.row + '/token-window?' + params.toString());
            if (!response.ok) throw new Error('HTTP ' + response.status);
            var holder = document.createElement('div');
            holder.innerHTML = await response.text();
            var chunk = holder.firstElementChild;
            if (!chunk || !chunk.classList.contains('token-window-chunk')) {
                throw new Error('invalid token window response');
            }
            stream.appendChild(chunk);
            stream.dataset.nextOffset = chunk.dataset.nextOffset;
            var chunks = stream.querySelectorAll('.token-window-chunk');
            while (chunks.length > MAX_TOKEN_WINDOW_CHUNKS) {
                var first = chunks[0];
                var removedHeight = first.getBoundingClientRect().height;
                first.remove();
                window.scrollBy(0, -removedHeight);
                chunks = stream.querySelectorAll('.token-window-chunk');
            }
            sentinel.textContent = 'scroll to load more';
            updateDone();
        } catch (error) {
            sentinel.textContent = 'load failed; scroll or click to retry';
        } finally {
            sentinel.classList.remove('loading');
            tokenWindowLoading = false;
        }
    }
    sentinel.onclick = loadNextTokenWindow;
    tokenWindowObserver = new IntersectionObserver(function(entries) {
        if (entries.some(function(entry) { return entry.isIntersecting; })) {
            loadNextTokenWindow();
        }
    }, {rootMargin: '600px 0px'});
    tokenWindowObserver.observe(sentinel);
    updateDone();
}
document.addEventListener('keydown',function(e){
    if(e.target.tagName==='INPUT')return;
    if(e.key==='ArrowLeft'||e.key==='[')navigate(-1);
    else if(e.key==='ArrowRight'||e.key===']')navigate(1);
    else if(e.key==='g')navigateTo(1);
    else if(e.key==='G')navigateTo(9999999);
    else if(e.key==='r')location.reload();
    else if(e.key==='s')sampleRow(parseInt(document.body.dataset.totalRows)||1);
    else if(e.key==='1')switchMode('generic');
    else if(e.key==='2')switchMode('sdd');
    else if(e.key==='3')switchMode('raw');
    else if(e.key==='4')switchMode('tokens');
});

// Auto-refresh file list for glob sources (poll /files every 5s)
var _fileRefreshTimer = null;
var _lastFileCount = 0;

function startFileRefresh() {
    if (typeof GLOB_MODE !== 'undefined' && GLOB_MODE === 1) {
        _lastFileCount = document.querySelectorAll('.file-row').length;
        _fileRefreshTimer = setInterval(checkFileUpdates, 5000);
    }
}

function checkFileUpdates() {
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/files', true);
    xhr.onload = function() {
        if (xhr.status !== 200) return;
        try {
            var data = JSON.parse(xhr.responseText);
            if (data.count !== _lastFileCount || data.count === 0) {
                _lastFileCount = data.count;
                refreshFilePicker(data.files || []);
            }
        } catch(e) {}
    };
    xhr.send();
}

function refreshFilePicker(files) {
    var rows = [];
    for (var i = 0; i < files.length; i++) {
        var f = files[i];
        var path = f.path || f;
        var parts = path.replace(/\\/g, '/').split('/');
        var name = f.name || parts[parts.length - 1];
        var dir = parts.length > 1 ? parts.slice(0, -1).join('/') : '';
        rows.push(
            '<div class="file-row" data-idx="' + i + '" data-path="' + _escAttr(path) + '"' +
            ' onclick="selectFile(' + i + ')">' +
            '<span class="file-icon">' + String.fromCharCode(0x1F4C4) + '</span>' +
            '<span class="file-name">' + _escHtml(name) + '</span>' +
            '<span class="file-dir">' + _escHtml(dir) + '</span>' +
            '</div>'
        );
    }
    var fileList = document.getElementById('file-list');
    if (fileList) {
        fileList.innerHTML = rows.join('\n');
    }
    var mc = document.getElementById('match-count');
    if (mc) {
        mc.textContent = files.length + ' files';
    }
    var inp = document.getElementById('picker-filter');
    if (inp) {
        inp.value = '';
    }
}

function _escHtml(text) {
    var d = document.createElement('div');
    d.appendChild(document.createTextNode(text));
    return d.innerHTML;
}

function _escAttr(text) {
    return text.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// Initialize mode from URL
window.addEventListener('DOMContentLoaded', function() {
    var qs = window.location.search;
    if (qs.indexOf('mode=') >= 0) {
        var m = qs.split('mode=')[1].split('&')[0];
        if (m) { currentMode = m; }
    }
    var btns = document.querySelectorAll('.mode-btn');
    btns.forEach(function(b) {
        b.classList.toggle('active', b.dataset.mode === currentMode);
    });
    initTokenWindowScroll();
    startFileRefresh();
});
"""


# ---------------------------------------------------------------------------
# Page templates
# ---------------------------------------------------------------------------


def _page(head_extra: str, body_content: str, body_attrs: str = "") -> str:
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            head_extra,
            "<style>",
            _CSS,
            "</style>",
            "</head>",
            "<body" + body_attrs + ">",
            body_content,
            "<script>",
            _JS,
            "</script>",
            "</body>",
            "</html>",
        ]
    )


def _build_header(source: RowSource, row_idx: int, mode: str) -> str:
    total = source.total_rows
    generic_active = " active" if mode in ("generic", "auto") else ""
    sdd_active = " active" if mode == "sdd" else ""
    tokens_active = " active" if mode == "tokens" else ""
    raw_active = " active" if mode == "raw" else ""
    return (
        '<div class="header">'
        '<span class="path">{}</span>'
        '<span class="row-info">row {} / {}</span>'
        '<div class="nav">'
        '<button onclick="navigate(-1)" title="Previous row (&larr; or [)">&larr; prev</button>'
        '<button onclick="navigate(1)" title="Next row (&rarr; or ])">next &rarr;</button>'
        '<button onclick="sampleRow({})" title="Show a random row (s)">sample</button>'
        '<span class="sep">|</span>'
        '<button onclick="gotoRow()" title="Go to row (g = first, G = last)">go to</button>'
        '<input id="goto-input" type="text" placeholder="{}"'
        " onkeydown=\"if(event.key==='Enter')gotoRow()\">"
        '<span class="sep">/ {}</span>'
        '<span class="sep" style="margin-left:8px">mode</span>'
        '<button class="mode-btn{}" data-mode="generic" onclick="switchMode(\'generic\')" '
        'title="Interactive tree (key 1)">tree</button>'
        '<button class="mode-btn{}" data-mode="sdd" onclick="switchMode(\'sdd\')" '
        'title="Side-by-side chat (key 2)">sdd</button>'
        '<button class="mode-btn{}" data-mode="tokens" onclick="switchMode(\'tokens\')" '
        'title="Bounded token preview (key 4)">tokens</button>'
        '<button class="mode-btn{}" data-mode="raw" onclick="switchMode(\'raw\')" '
        'title="Plain JSON (key 3)">raw</button>'
        "</div>"
        "</div>"
    ).format(
        _esc(source.display_path),
        row_idx + 1,
        total,
        total,
        row_idx + 1,
        total,
        generic_active,
        sdd_active,
        tokens_active,
        raw_active,
    )


def _build_footer() -> str:
    return (
        '<div class="footer">'
        '<span class="keyboard-hint">'
        "<kbd>&larr;</kbd><kbd>&rarr;</kbd> prev/next &nbsp;"
        "<kbd>g</kbd> first &nbsp;"
        "<kbd>G</kbd> last &nbsp;"
        "<kbd>s</kbd> sample &nbsp;"
        "<kbd>r</kbd> reload &nbsp;"
        "<kbd>1</kbd> tree &nbsp;"
        "<kbd>2</kbd> sdd &nbsp;"
        "<kbd>3</kbd> raw &nbsp;"
        "<kbd>4</kbd> tokens"
        "</span>"
        "</div>"
    )


def _error_page(title: str, message: str) -> str:
    body = (
        '<div style="max-width:600px;margin:80px auto;text-align:center;">'
        '<h2 style="color:var(--null);margin-bottom:12px;">{}</h2>'
        '<p style="color:var(--muted);">{}</p>'
        '<p style="margin-top:16px;">'
        '<a href="/row/1" style="color:var(--accent);">&larr; back to first row</a>'
        "</p>"
        "</div>"
    ).format(_esc(title), _esc(message))
    return _page("", body)


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


class _GlobRefreshThread(threading.Thread):
    """Background thread that periodically re-globs the source folder.

    Updates glob_source.files in place. The file picker page polls /files
    to pick up changes.
    """

    def __init__(self, glob_source: JsonlGlobRowSource, interval: float = 5.0) -> None:
        super().__init__(daemon=True)
        self.glob_source = glob_source
        self.interval = interval
        self._stopped = threading.Event()
        self._lock = threading.Lock()

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        while not self._stopped.wait(self.interval):
            try:
                with self._lock:
                    self.glob_source.refresh()
            except Exception:
                pass  # silently retry next cycle

    def get_files(self) -> list[dict[str, str]]:
        """Thread-safe current file list for the /files API."""
        with self._lock:
            return [{"path": str(p), "name": p.name} for p in self.glob_source.files]


class PcatHandler(BaseHTTPRequestHandler):
    source: RowSource = None  # type: ignore[assignment]
    mode: str = "auto"
    tokenizer_name: str = _DEFAULT_SDD_TOKENIZER
    glob_source: Any = None  # JsonlGlobRowSource | None
    refresh_thread: _GlobRefreshThread | None = None
    server_version = "pcat-serve"

    def log_message(self, format, *args):  # noqa: A002
        pass

    def do_GET(self):
        parsed = urlparse(self.path)  # type: ignore[attr-defined]
        route = parsed.path.rstrip("/")

        if route.startswith("/file/"):
            self._serve_file_pick(route)
            return

        if route == "/files":
            self._serve_files_json()
            return

        if route.startswith("/row/") and route.endswith("/token-window"):
            self._serve_token_window(route, parsed)
            return

        if route.startswith("/row/") and "/node" in route:
            self._serve_node(route, parsed)
            return

        if route.startswith("/row/"):
            self._serve_row(route, parsed)
            return

        if route in ("", "/"):
            if self.glob_source is not None:
                self._serve_picker_root()
                return
            self.send_response(302)
            self.send_header("Location", "/row/1")
            self.end_headers()
            return

        self.send_response(404)
        self.end_headers()
        body = _error_page("404 Not Found", "No route for: " + route)
        self.wfile.write(body.encode())

    def _serve_files_json(self):
        """Return current file list as JSON for periodic polling."""
        if self.refresh_thread is not None:
            files = self.refresh_thread.get_files()
        elif self.glob_source is not None:
            files = [{"path": str(p), "name": p.name} for p in self.glob_source.files]
        else:
            files = []
        data = json.dumps({"count": len(files), "files": files})
        payload = data.encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _serve_picker_root(self):
        """Landing page: file picker for glob sources."""
        files = self.glob_source.file_list
        body = _file_picker_page(files, 0, glob_mode=True)
        html = _page("", body)
        self._send_html(html)

    def _serve_file_pick(self, route: str):
        """Select a file from the glob source and redirect to its row view."""
        if self.glob_source is None:
            self._send_html(
                _error_page(
                    "Not a glob source",
                    "This endpoint is only for folder-based sources.",
                ),
                400,
            )
            return
        slug = route[len("/file/") :]
        try:
            idx = int(slug)
        except ValueError:
            self._send_html(
                _error_page("Bad file index", "Invalid file index: " + slug), 400
            )
            return
        try:
            file_source = self.glob_source.select_file(idx)
        except ValueError as exc:
            self._send_html(_error_page("File not found", str(exc)), 404)
            return
        # Swap to the selected file source
        PcatHandler.source = file_source
        # Redirect to first row of this file
        self.send_response(302)
        self.send_header("Location", "/row/1")
        self.end_headers()

    def _serve_node(self, path: str, parsed):
        remainder = path[len("/row/") :]
        idx_str, _, tail = remainder.partition("/node")
        try:
            idx = int(idx_str) - 1
        except ValueError:
            self.send_response(400)
            self.end_headers()
            return

        if idx < 0 or idx >= self.source.total_rows:
            self.send_response(404)
            self.end_headers()
            return

        json_path = parsed.query
        if json_path.startswith("path="):
            json_path = json_path[5:]
        from urllib.parse import unquote_plus

        json_path = unquote_plus(json_path)

        try:
            value = self.source.load_row(idx)
        except Exception:
            self.send_response(500)
            self.end_headers()
            return

        fragment = _render_node_children(value, json_path)
        data = fragment.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_token_window(self, path: str, parsed) -> None:
        slug = path[len("/row/") : -len("/token-window")]
        try:
            idx = int(slug) - 1
            _mode, offset, limit = _parse_render_query(parsed.query, "sdd")
        except ValueError:
            self._send_html(
                _error_page(
                    "Bad token window", "row, offset, and limit must be integers"
                ),
                400,
            )
            return
        if idx < 0 or idx >= self.source.total_rows:
            self._send_html(_error_page("Out of range", "Row is out of range"), 404)
            return
        try:
            value = self.source.load_row(idx)
        except Exception as exc:
            self._send_html(_error_page("Load error", str(exc)), 500)
            return
        if not (_is_tokenized_sdd(value) or _training_schema(value) is not None):
            self._send_html(
                _error_page(
                    "Unsupported row",
                    "Infinite token scrolling requires a recognized tokenized row",
                ),
                400,
            )
            return
        try:
            tokenizer = _get_tokenizer(self.tokenizer_name)
            chunk, _start, _end, _total = _render_token_window_chunk(
                value, tokenizer, offset=offset, limit=limit
            )
        except Exception as exc:
            self._send_html(_error_page("Tokenizer error", str(exc)), 500)
            return
        self._send_html(chunk)

    def _serve_row(self, path: str, parsed):
        total = self.source.total_rows
        if total <= 0:
            self._send_html(_error_page("Empty", "Source has no rows"), 500)
            return

        slug = path[len("/row/") :]
        if slug == "last":
            idx = total - 1
        else:
            try:
                idx = int(slug) - 1
            except ValueError:
                self._send_html(_error_page("Bad row", "Invalid row: " + slug), 400)
                return

        if idx < 0 or idx >= total:
            self._send_html(
                _error_page(
                    "Out of range",
                    "Row {} is out of range (1–{}).".format(idx + 1, total),
                ),
                404,
            )
            return

        # Allow render mode and bounded token window to be changed on the fly.
        try:
            mode, token_offset, token_limit = _parse_render_query(
                parsed.query, self.mode
            )
        except ValueError:
            self._send_html(
                _error_page("Bad token window", "offset and limit must be integers"),
                400,
            )
            return

        try:
            value = self.source.load_row(idx)
        except Exception as exc:
            self._send_html(_error_page("Load error", str(exc)), 500)
            return

        header = _build_header(self.source, idx, mode)
        tokenizer = None
        resolved_hint = _detect_mode(value) if mode == "auto" else mode
        needs_tokenizer = (resolved_hint == "sdd" and _is_tokenized_sdd(value)) or (
            resolved_hint == "tokens" and _training_schema(value) is not None
        )
        if needs_tokenizer:
            try:
                tokenizer = _get_tokenizer(self.tokenizer_name)
            except Exception as exc:
                self._send_html(_error_page("Tokenizer error", str(exc)), 500)
                return
        try:
            rendered, resolved_mode = render_row(
                value,
                idx,
                mode,
                tokenizer=tokenizer,
                token_offset=token_offset,
                token_limit=token_limit,
            )
        except (TypeError, ValueError) as exc:
            self._send_html(_error_page("Preview error", str(exc)), 400)
            return
        if resolved_mode != mode:
            # render_row auto-detected a different mode — rebuild header
            header = _build_header(self.source, idx, resolved_mode)
        body = "{}\n<div class='main'>{}</div>\n{}".format(
            header, rendered, _build_footer()
        )
        self._send_html(
            _page(
                "",
                body,
                ' data-total-rows="{}" data-mode="{}"'.format(
                    total, _esc_attr(resolved_mode)
                ),
            )
        )

    def _send_html(self, html_str: str, code: int = 200):
        data = html_str.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Server entrypoint
# ---------------------------------------------------------------------------


def serve(
    source: RowSource,
    *,
    host: str = "127.0.0.1",
    port: int = 8888,
    mode: str = "auto",
    tokenizer_name: str = _DEFAULT_SDD_TOKENIZER,
    open_browser: bool = True,
    glob_source: Any = None,
    refresh_interval: float = 5.0,
) -> int:
    PcatHandler.source = source
    PcatHandler.mode = mode
    PcatHandler.tokenizer_name = tokenizer_name
    PcatHandler.glob_source = glob_source

    # Start background file-refresh thread for glob sources
    refresh_thread = None
    if glob_source is not None:
        refresh_thread = _GlobRefreshThread(glob_source, interval=refresh_interval)
        refresh_thread.start()
        PcatHandler.refresh_thread = refresh_thread
    else:
        PcatHandler.refresh_thread = None

    url = "http://{}:{}".format(host, port)

    # Peek first row to resolve auto mode for startup display
    display_mode = mode
    if mode == "auto" and source.total_rows > 0:
        try:
            first = source.load_row(0)
            display_mode = _detect_mode(first)
        except Exception:
            pass

    mode_str = mode if mode != "auto" else "auto ({})".format(display_mode)
    print("  pcat serve " + "\u2500" * 38, file=sys.stderr)
    print("  source : " + source.display_path, file=sys.stderr)
    print("  rows   : " + str(source.total_rows), file=sys.stderr)
    print("  mode   : " + mode_str, file=sys.stderr)
    print("  url    : " + url, file=sys.stderr)
    if glob_source is not None:
        print("  refresh: every {:.0f}s".format(refresh_interval), file=sys.stderr)
    print("  " + "\u2500" * 49, file=sys.stderr)

    server = HTTPServer((host, port), PcatHandler)
    if open_browser:
        # Remote-editor browser helpers can wait indefinitely for their child
        # process.  Browser launch is best-effort and must never gate serving.
        def open_browser_in_background() -> None:
            with suppress(Exception):
                webbrowser.open(url)

        threading.Thread(
            target=open_browser_in_background,
            name="pcat-browser-open",
            daemon=True,
        ).start()

    try:
        print("  Serving at {} ...  (Ctrl+C to stop)".format(url), file=sys.stderr)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.", file=sys.stderr)
    finally:
        if refresh_thread is not None:
            refresh_thread.stop()
        server.server_close()
    return 0
