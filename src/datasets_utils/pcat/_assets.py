# -*- coding: utf-8 -*-
"""Static CSS/JS assets served by the pcat web viewer."""

from __future__ import annotations


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
            if (data.count !== _lastFileCount) { location.reload(); }
        } catch(e) {}
    };
    xhr.send();
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
