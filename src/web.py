"""Small zero-dependency web UI for the cyber RAG pipeline.

Run with ``python -m src.web``.
"""
from __future__ import annotations

import argparse
import json
import logging
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.config import ROOT, settings


WEB_DIR = ROOT / "web"
MAX_REQUEST_BYTES = 64 * 1024
MAX_QUERY_CHARS = 4_000


def _root_cause(exc: BaseException) -> str:
    while exc.__cause__ is not None:
        exc = exc.__cause__
    return str(exc)


def _serialize_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return only the user-facing, JSON-safe portion of a graph state."""
    answer = state.get("answer")
    return {
        "status": state.get("status", "error"),
        "response": state.get("response", ""),
        "rounds": state.get("round", 0),
        "answer": answer.model_dump(mode="json") if answer is not None else None,
        "trace": state.get("trace", []),
    }


def _system_status() -> dict[str, Any]:
    """Cheap readiness information; this never contacts or bills the LLM."""
    from src.ingestion.indexing import VectorStore
    from src.retrieval import get_active_retriever

    retriever = get_active_retriever()
    if retriever is not None:
        # Local Qdrant holds an exclusive directory lock. Reuse the client that
        # the retrieval pipeline already owns after the first request.
        chunks = retriever.store.count()
    else:
        with VectorStore() as store:
            chunks = store.count()
    return {
        "ready": chunks > 0,
        "indexed_chunks": chunks,
        "documents": len(list(settings.raw_dir.glob("*.pdf"))),
        "provider": settings.llm_provider,
        "model": settings.anthropic_model if settings.is_anthropic else settings.openai_model,
    }


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "CyberRAGDemo/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_asset(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{media_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/status":
            try:
                self._send_json(_system_status())
            except Exception as exc:
                self._send_json({"ready": False, "error": _root_cause(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return

        assets = {
            "/": WEB_DIR / "index.html",
            "/index.html": WEB_DIR / "index.html",
            "/assets/styles.css": WEB_DIR / "styles.css",
            "/assets/app.js": WEB_DIR / "app.js",
        }
        asset = assets.get(path)
        if asset is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_asset(asset)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/ask":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length."}, HTTPStatus.BAD_REQUEST)
            return
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._send_json({"error": "Request body is empty or too large."}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return

        try:
            payload = json.loads(self.rfile.read(length))
            query = payload.get("query", "").strip()
        except (json.JSONDecodeError, AttributeError):
            self._send_json({"error": "Request must be a JSON object."}, HTTPStatus.BAD_REQUEST)
            return

        if not query:
            self._send_json({"error": "Please enter a question."}, HTTPStatus.BAD_REQUEST)
            return
        if len(query) > MAX_QUERY_CHARS:
            self._send_json({"error": f"Question must be under {MAX_QUERY_CHARS:,} characters."}, HTTPStatus.BAD_REQUEST)
            return

        try:
            from src.graph.workflow import run

            self._send_json(_serialize_state(run(query)))
        except Exception as exc:
            logging.exception("Pipeline request failed")
            self._send_json({"error": _root_cause(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Cyber RAG web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=7860, type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s: %(message)s")

    server = HTTPServer((args.host, args.port), DemoHandler)
    print(f"Cyber RAG demo is running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
