#!/usr/bin/env python3
"""Simple local web UI for imgsearch.py.

Usage (index the folder first with imgsearch.py):
    python webui.py fotos
    python webui.py fotos --port 8000 --no-browser

Needs only the packages imgsearch.py already uses (Pillow comes with transformers).
Listens on 127.0.0.1 only, and only serves files inside the chosen folder.
"""
import argparse
import hashlib
import io
import json
import mimetypes
import tempfile
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageOps

import imgsearch

THUMB_SIZE = 256
MAX_UPLOAD = 50 * 1024 * 1024

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Image search</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#1b1b1b; --muted:#777; --card:#f2f2f2; --line:#ccc; --accent:#3b6fd4; }
  @media (prefers-color-scheme: dark) { :root { --bg:#17181a; --fg:#e8e8e8; --muted:#999; --card:#222427; --line:#3a3d41; --accent:#6c97ee; } }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  header { position:sticky; top:0; z-index:5; background:var(--bg); border-bottom:1px solid var(--line); padding:12px 16px; }
  .row { display:flex; gap:10px; flex-wrap:wrap; align-items:stretch; }
  input[type=text] { flex:1 1 280px; padding:10px 12px; font-size:16px; border:1px solid var(--line); border-radius:8px; background:var(--card); color:var(--fg); }
  button, select { padding:10px 16px; font-size:15px; border:1px solid var(--line); border-radius:8px; background:var(--card); color:var(--fg); cursor:pointer; }
  button.primary { background:var(--accent); border-color:var(--accent); color:#fff; }
  #drop { flex:0 0 190px; border:2px dashed var(--line); border-radius:8px; display:flex; align-items:center; justify-content:center; text-align:center; font-size:13px; color:var(--muted); padding:6px; min-height:44px; cursor:pointer; }
  #drop.over { border-color:var(--accent); color:var(--accent); }
  #drop img { max-height:44px; max-width:60px; border-radius:4px; margin-right:8px; }
  #status { padding:10px 16px; color:var(--muted); font-size:14px; min-height:20px; }
  #grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(170px, 1fr)); gap:12px; padding:0 16px 24px; }
  .card { background:var(--card); border-radius:8px; overflow:hidden; cursor:pointer; display:flex; flex-direction:column; }
  .card img { width:100%; aspect-ratio:1; object-fit:cover; display:block; background:#0002; }
  .meta { padding:6px 8px; font-size:12px; line-height:1.3; }
  .meta .p { word-break:break-all; }
  .meta .s { color:var(--muted); margin-top:2px; }
  #lb { position:fixed; inset:0; background:#000d; display:none; align-items:center; justify-content:center; flex-direction:column; z-index:10; padding:16px; }
  #lb.show { display:flex; }
  #lb img { max-width:96vw; max-height:84vh; object-fit:contain; }
  #lbcap { color:#ddd; font-size:13px; margin-top:10px; word-break:break-all; text-align:center; }
  #lbbtns { margin-top:8px; display:flex; gap:8px; }
</style></head>
<body>
<header>
  <div class="row">
    <input id="q" type="text" placeholder="Describe what you're looking for, e.g. dog playing in snow" autofocus>
    <select id="k"><option>12</option><option selected>24</option><option>48</option><option>96</option></select>
    <button class="primary" id="go">Search</button>
    <div id="drop" title="Drop, click to choose, or paste an image">Drop an image here<br>(or click / paste)</div>
    <input id="file" type="file" accept="image/*" hidden>
  </div>
</header>
<div id="status"></div>
<div id="grid"></div>
<div id="lb"><img id="lbimg" alt=""><div id="lbcap"></div>
  <div id="lbbtns"><button id="similar">Find similar</button><button id="close">Close</button></div></div>
<script>
const $ = id => document.getElementById(id);
let lbRel = null;

function setStatus(t) { $('status').textContent = t; }

function render(data) {
  const grid = $('grid'); grid.innerHTML = '';
  data.results.forEach(r => {
    const c = document.createElement('div'); c.className = 'card';
    const img = document.createElement('img');
    img.loading = 'lazy'; img.src = '/thumb?p=' + encodeURIComponent(r.rel);
    const m = document.createElement('div'); m.className = 'meta';
    const p = document.createElement('div'); p.className = 'p'; p.textContent = r.path;
    const s = document.createElement('div'); s.className = 's'; s.textContent = 'score ' + r.score.toFixed(3);
    m.append(p, s); c.append(img, m);
    c.onclick = () => openLb(r);
    grid.append(c);
  });
  setStatus(data.results.length + ' results in ' + data.seconds.toFixed(1) + ' s');
}

async function run(promise) {
  setStatus('Searching…');
  try {
    const resp = await promise;
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || resp.statusText);
    render(data);
  } catch (e) { setStatus('Error: ' + e.message); }
}

function textSearch() {
  const q = $('q').value.trim(); if (!q) return;
  clearPreview();
  run(fetch('/api/search?k=' + $('k').value + '&q=' + encodeURIComponent(q)));
}

function imageSearch(file) {
  if (!file || !file.type.startsWith('image/')) { setStatus('That is not an image file.'); return; }
  const d = $('drop'); d.innerHTML = '';
  const im = document.createElement('img'); im.src = URL.createObjectURL(file);
  d.append(im, document.createTextNode(file.name || 'image'));
  run(fetch('/api/search_image?k=' + $('k').value, { method:'POST', body:file }));
}

function clearPreview() { $('drop').innerHTML = 'Drop an image here<br>(or click / paste)'; }

function openLb(r) {
  lbRel = r.rel;
  $('lbimg').src = '/img?p=' + encodeURIComponent(r.rel);
  $('lbcap').textContent = r.path;
  $('lb').classList.add('show');
}
function closeLb() { $('lb').classList.remove('show'); $('lbimg').src = ''; }

$('go').onclick = textSearch;
$('q').addEventListener('keydown', e => { if (e.key === 'Enter') textSearch(); });
$('close').onclick = closeLb;
$('lb').addEventListener('click', e => { if (e.target.id === 'lb') closeLb(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLb(); });
$('similar').onclick = () => {
  if (!lbRel) return; const rel = lbRel; closeLb();
  clearPreview(); $('q').value = '';
  run(fetch('/api/search_similar?k=' + $('k').value + '&p=' + encodeURIComponent(rel)));
};

const drop = $('drop');
drop.onclick = () => $('file').click();
$('file').onchange = e => imageSearch(e.target.files[0]);
['dragenter','dragover'].forEach(ev => document.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave','drop'].forEach(ev => document.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
document.addEventListener('drop', e => { const f = e.dataTransfer.files[0]; if (f) imageSearch(f); });
document.addEventListener('paste', e => {
  const item = [...(e.clipboardData?.items || [])].find(i => i.type.startsWith('image/'));
  if (item) imageSearch(item.getAsFile());
});
</script></body></html>
"""


class State:
    def __init__(self, folder):
        self.folder = folder
        self.model = imgsearch.load_model()
        self.lock = threading.Lock()  # one model call at a time
        self._meta_mtime = None
        self.rel_by_row, self.vecs = {}, np.zeros((0, imgsearch.DIM), np.float32)
        self.row_by_rel = {}
        self.thumb_dir = imgsearch.index_dir(folder) / "thumbs"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)

    def refresh_index(self):
        meta_f = imgsearch.index_dir(self.folder) / "meta.json"
        if not meta_f.exists():
            raise RuntimeError("No index found. Run: python imgsearch.py index <folder>")
        mtime = meta_f.stat().st_mtime
        if mtime != self._meta_mtime:
            files, self.vecs = imgsearch.load_index(self.folder)
            self.row_by_rel = {rel: m["row"] for rel, m in files.items()}
            self.rel_by_row = {row: rel for rel, row in self.row_by_rel.items()}
            self._meta_mtime = mtime
        if not len(self.vecs):
            raise RuntimeError("The index is empty.")

    def rank(self, qvec, k):
        scores = self.vecs @ qvec
        order = np.argsort(-scores)[:k]
        return [{"rel": self.rel_by_row[int(i)],
                 "path": str(self.folder / self.rel_by_row[int(i)]),
                 "score": float(scores[i])} for i in order]

    def safe_path(self, rel):
        p = (self.folder / rel).resolve()
        if not p.is_relative_to(self.folder) or not p.is_file():
            return None
        return p

    def thumb_bytes(self, p):
        key = hashlib.sha1(f"{p}|{p.stat().st_mtime}".encode()).hexdigest()
        cache = self.thumb_dir / f"{key}.jpg"
        if cache.exists():
            return cache.read_bytes()
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            im.thumbnail((THUMB_SIZE, THUMB_SIZE))
            if im.mode != "RGB":
                im = im.convert("RGB")
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=82)
        cache.write_bytes(buf.getvalue())
        return buf.getvalue()


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def err(self, code, msg):
            self.send(code, {"error": msg})

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            first = lambda name, d="": qs.get(name, [d])[0]
            k = max(1, min(int(first("k", "24") or 24), 500))
            try:
                if u.path == "/":
                    return self.send(200, PAGE, "text/html; charset=utf-8")
                if u.path in ("/thumb", "/img"):
                    p = state.safe_path(first("p"))
                    if not p:
                        return self.err(404, "not found")
                    if u.path == "/thumb":
                        return self.send(200, state.thumb_bytes(p), "image/jpeg")
                    ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                    return self.send(200, p.read_bytes(), ctype)
                if u.path == "/api/search":
                    q = first("q").strip()
                    if not q:
                        return self.err(400, "empty query")
                    return self.do_search(lambda: imgsearch.embed_text_query(state.model, q), k)
                if u.path == "/api/search_similar":
                    state.refresh_index()
                    row = state.row_by_rel.get(first("p"))
                    if row is None:
                        return self.err(404, "image not in index")
                    qv = state.vecs[row]
                    return self.do_search(lambda: qv, k)
                self.err(404, "not found")
            except Exception as e:
                self.err(500, str(e))

        def do_POST(self):
            u = urlparse(self.path)
            if u.path != "/api/search_image":
                return self.err(404, "not found")
            k = max(1, min(int(parse_qs(u.query).get("k", ["24"])[0]), 500))
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_UPLOAD:
                return self.err(400, "bad upload size")
            data = self.rfile.read(n)
            tmp = None
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    tmp = Path(f.name)
                    im.save(f, "PNG")
                    f.close()
                return self.do_search(lambda: imgsearch.embed_image(state.model, tmp), k)
            except Exception as e:
                return self.err(500, str(e))
            finally:
                if tmp:
                    tmp.unlink(missing_ok=True)

        def do_search(self, get_qvec, k):
            import time
            t0 = time.time()
            try:
                state.refresh_index()
                with state.lock:
                    qvec = get_qvec()
                results = state.rank(qvec, k)
            except RuntimeError as e:
                return self.err(400, str(e))
            self.send(200, {"results": results, "seconds": time.time() - t0})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a folder: {folder}")
    print("Loading model…")
    state = State(folder)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(state))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving {folder}\nOpen {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()