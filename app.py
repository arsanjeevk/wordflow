"""Local browser interface for WordFlow."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from generation import DEFAULT_CHECKPOINT, generate, load_model


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
model, device = load_model(DEFAULT_CHECKPOINT)
generation_lock = Lock()


class Handler(BaseHTTPRequestHandler):
    def send_data(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        files = {"/": ("index.html", "text/html; charset=utf-8"),
                 "/style.css": ("style.css", "text/css; charset=utf-8"),
                 "/app.js": ("app.js", "application/javascript; charset=utf-8")}
        if self.path not in files:
            self.send_error(404)
            return
        name, content_type = files[self.path]
        self.send_data(200, content_type, (STATIC / name).read_bytes())

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 8192:
                raise ValueError("Request is too large or empty.")
            data = json.loads(self.rfile.read(length))
            prompt = data.get("prompt", "")
            if not isinstance(prompt, str) or len(prompt) > 2000:
                raise ValueError("Prompt must be text under 2,000 characters.")
            tokens = int(data.get("tokens", 30))
            temperature = float(data.get("temperature", 0.8))
            top_k = int(data.get("top_k", 25))
            seed = data.get("seed")
            seed = int(seed) if seed not in (None, "") else None
            with generation_lock:
                result = generate(model, prompt, tokens, temperature,
                                  top_k or None, device, seed)
            body = json.dumps(result).encode("utf-8")
            self.send_data(200, "application/json; charset=utf-8", body)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_data(400, "application/json; charset=utf-8",
                           json.dumps({"error": str(exc)}).encode("utf-8"))
        except Exception as exc:
            self.send_data(500, "application/json; charset=utf-8",
                           json.dumps({"error": f"Generation failed: {exc}"}).encode("utf-8"))


if __name__ == "__main__":
    address = ("127.0.0.1", 8000)
    print(f"Loaded {DEFAULT_CHECKPOINT.name} on {device}. Open http://{address[0]}:{address[1]}", flush=True)
    ThreadingHTTPServer(address, Handler).serve_forever()
