"""Local HTTP adapter: accepts OpenAI Chat Completions, calls an upstream Messages API.

The client's ``Authorization`` header is forwarded upstream unchanged, so this process holds
no credential. Request bodies, prompts, images and credentials are never logged.
"""
import argparse
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Iterator, List, Optional, Type

from .translate import StreamTranslator, UnsupportedRequest, aggregate, to_messages_request

logger = logging.getLogger(__name__)

Json = Dict[str, Any]


@dataclass(frozen=True)
class AdapterConfig:
    """Adapter settings.

    Attributes:
        upstream_base_url: Upstream base ending in the API version, e.g. ``https://host/api/v1``.
        host: Listen address; loopback by default.
        port: Listen port; 0 picks a free one.
        default_max_tokens: Token limit used when a request sets none.
        upstream_timeout_seconds: Connect and idle-read timeout for the upstream call.
        anthropic_version: Value of the ``anthropic-version`` header sent upstream.
    """

    upstream_base_url: str
    host: str = "127.0.0.1"
    port: int = 18200
    default_max_tokens: int = 8192
    upstream_timeout_seconds: float = 600.0
    anthropic_version: str = "2023-06-01"


def _sse_events(response: Any) -> Iterator[Json]:
    for line in response:
        text = line.decode("utf-8", errors="replace").strip()
        if not text.startswith("data:"):
            continue
        try:
            event = json.loads(text[5:].strip())
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def _handler(config: AdapterConfig) -> Type[BaseHTTPRequestHandler]:
    upstream_url = config.upstream_base_url.rstrip("/") + "/messages"
    opener = urllib.request.build_opener()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"  # close-delimited bodies; no chunked encoding needed

        def log_message(self, *_: Any) -> None:
            pass

        def _json(self, status: int, body: Json, headers: Optional[Dict[str, str]] = None) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str, headers: Optional[Dict[str, str]] = None) -> None:
            self._json(status, {"error": {"message": message, "type": "adapter_error", "code": status}}, headers)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"status": "ok", "upstream": upstream_url})
            else:
                self._error(404, "not found")

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._error(404, "not found")
                return
            authorization = self.headers.get("Authorization")
            if not authorization:
                self._error(401, "missing Authorization header")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if not isinstance(body, dict):
                    raise UnsupportedRequest("request body must be a JSON object")
                payload = to_messages_request(body, config.default_max_tokens)
            except (json.JSONDecodeError, ValueError) as exc:
                self._error(400, str(exc) if isinstance(exc, UnsupportedRequest) else "invalid JSON body")
                return
            streaming = payload["stream"]
            payload["stream"] = True
            request = urllib.request.Request(upstream_url, json.dumps(payload).encode(), {
                "Content-Type": "application/json", "Authorization": authorization,
                "anthropic-version": config.anthropic_version, "Accept": "text/event-stream"})
            try:
                with opener.open(request, timeout=config.upstream_timeout_seconds) as response:
                    if "text/event-stream" not in response.headers.get("Content-Type", ""):
                        logger.warning("upstream returned a non-SSE success response")
                        self._error(502, "upstream did not return an event stream")
                        return
                    self._relay(response, streaming)
            except urllib.error.HTTPError as exc:
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                exc.close()
                logger.warning("upstream HTTP %s for model %s", exc.code, payload.get("model"))
                # The upstream body may echo credentials or prompts; never forward it.
                self._error(exc.code, f"upstream HTTP {exc.code}",
                            {"Retry-After": retry_after} if retry_after else None)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                logger.warning("upstream network failure: %s", type(exc).__name__)
                self._error(502, "upstream network failure: " + type(exc).__name__)

        def _relay(self, response: Any, streaming: bool) -> None:
            translator = StreamTranslator()
            if not streaming:
                chunks: List[Json] = [c for event in _sse_events(response) for c in translator.feed(event)]
                if translator.finished:
                    self._json(200, aggregate(chunks))
                else:
                    self._error(502, "upstream stream ended before completion")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                for event in _sse_events(response):
                    for chunk in translator.feed(event):
                        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                    self.wfile.flush()
                # [DONE] only after a verified complete upstream stream; a client must be
                # able to tell a truncated answer from a finished one.
                if translator.finished:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                else:
                    logger.warning("upstream stream ended before completion")
            except (BrokenPipeError, ConnectionResetError):
                logger.info("client disconnected mid-stream")
            except (TimeoutError, OSError) as exc:
                logger.warning("upstream stream failed mid-response: %s", type(exc).__name__)

    return Handler


def make_server(config: AdapterConfig) -> ThreadingHTTPServer:
    """Build the adapter server without starting it.

    Args:
        config: Adapter settings.

    Returns:
        A server ready for ``serve_forever``.
    """
    server = ThreadingHTTPServer((config.host, config.port), _handler(config))
    server.daemon_threads = True
    return server


def main() -> None:
    """Run the adapter from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-base-url", required=True, help="e.g. https://host/api/v1")
    parser.add_argument("--host", default=AdapterConfig.host)
    parser.add_argument("--port", type=int, default=AdapterConfig.port)
    parser.add_argument("--default-max-tokens", type=int, default=AdapterConfig.default_max_tokens)
    parser.add_argument("--upstream-timeout-seconds", type=float, default=AdapterConfig.upstream_timeout_seconds)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = AdapterConfig(upstream_base_url=args.upstream_base_url, host=args.host, port=args.port,
                           default_max_tokens=args.default_max_tokens,
                           upstream_timeout_seconds=args.upstream_timeout_seconds)
    server = make_server(config)
    logger.info("listening on http://%s:%s -> %s/messages", config.host, server.server_port,
                config.upstream_base_url.rstrip("/"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
