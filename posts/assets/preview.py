#!/usr/bin/env -S uv run --with markdown --no-project python
"""Live-preview server for a post + its assets — rebuilds HTML on change.

Run:  ./assets/preview.py [path/to/post.md]     (uv pulls `markdown` into a throwaway env)
      defaults to the newest dated `NNNN-*.md` in posts/ · serves http://localhost:8000

Serves posts/ as the web root, so the post's relative links (assets/plots/*.svg,
assets/token-stream.css) resolve exactly as they will on the destination site. The
browser holds an SSE connection to /__reload; the server polls mtimes of every
.md/.css/.svg under posts/ and pushes a reload when any changes. No file watcher dep.

NOT the destination pipeline (unknown) nor this repo's kramdown — python-markdown is a
close approximation. The token-stream spans, external CSS, and SVGs are raw HTML/assets,
so they render identically regardless of engine; only markdown-specific constructs differ.
"""
import http.server
import mimetypes
import pathlib
import socketserver
import sys
import time

import markdown

ROOT = pathlib.Path(__file__).resolve().parent.parent  # posts/
PORT = 8000
MD_EXTENSIONS = ["tables", "fenced_code", "sane_lists", "attr_list"]
WATCH_SUFFIXES = {".md", ".css", ".svg"}

mimetypes.add_type("image/svg+xml", ".svg")


def pick_post() -> pathlib.Path:
    if len(sys.argv) > 1:
        return pathlib.Path(sys.argv[1]).resolve()
    dated = sorted(ROOT.glob("[0-9]*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not dated:
        sys.exit(f"no dated NNNN-*.md post found in {ROOT} — pass one as an argument")
    return dated[0]


POST = pick_post()

PAGE = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — preview</title>
<style>
  body {{ max-width: 46rem; margin: 2.5rem auto; padding: 0 1.25rem;
         font: 16px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         color: #24292f; }}
  img, svg {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ border: 1px solid #d0d7de; padding: 0.35rem 0.7rem; }}
  th {{ background: #f6f8fa; }}
  code {{ background: #f6f8fa; padding: 0.15em 0.35em; border-radius: 4px; }}
  blockquote {{ border-left: 3px solid #d0d7de; margin: 1rem 0; padding: 0 1rem; color: #57606a; }}
  .preview-banner {{ position: fixed; top: 0; right: 0; background: #d6f5d6; color: #24292f;
         font: 12px sans-serif; padding: 3px 8px; border-bottom-left-radius: 6px; opacity: 0.85; }}
</style>
</head><body>
<div class="preview-banner">live preview · {name}</div>
{body}
<script>
  new EventSource("/__reload").onmessage = () => location.reload();
</script>
</body></html>"""


def render() -> bytes:
    text = POST.read_text(encoding="utf-8")
    body = markdown.markdown(text, extensions=MD_EXTENSIONS)
    title = POST.stem
    return PAGE.format(title=title, name=POST.name, body=body).encode("utf-8")


def max_mtime() -> float:
    return max((p.stat().st_mtime for p in ROOT.rglob("*")
                if p.suffix in WATCH_SUFFIXES), default=0.0)


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, *a):  # quiet — one line per real request is enough
        if not self.path.startswith("/__reload"):
            super().log_message(*a)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(render(), "text/html; charset=utf-8")
        if self.path == "/__reload":
            return self._sse()
        return super().do_GET()  # static asset from ROOT

    def _send(self, payload: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last = max_mtime()
        try:
            while True:
                time.sleep(0.3)
                now = max_mtime()
                if now > last:
                    last = now
                    self.wfile.write(b"data: reload\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away / reloaded


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


if __name__ == "__main__":
    print(f"==> Previewing {POST.relative_to(ROOT)}")
    print(f"==> Serving {ROOT} on http://localhost:{PORT}/  (Ctrl-C to stop)")
    with Server(("0.0.0.0", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n==> stopped")
