"""Local HTTP adapter: serves the JSON ``images/edits`` protocol from an image-output chat model.

The pipeline's image editor posts unauthenticated JSON to ``/v1/images/edits``. This process
turns each call into an upstream ``/chat/completions`` request with image output, adding the
credential read from a key file. Prompts, images and the credential are never logged; one
usage line per call (token counts only) is appended to the usage log for cost accounting.
"""
import argparse
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type

from .translate import NoImageReturned, UnsupportedRequest, to_chat_request, to_edits_response

logger = logging.getLogger(__name__)

Json = Dict[str, Any]
_USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


@dataclass(frozen=True)
class AdapterConfig:
    """Adapter settings.

    Attributes:
        upstream_base_url: Upstream base ending in the API version, e.g. ``https://host/api/v1``.
        model: Upstream image model used for every edit.
        api_key_file: File holding the upstream credential.
        host: Listen address; loopback by default.
        port: Listen port; 0 picks a free one.
        usage_log: JSONL file receiving one token-usage line per call; empty disables it.
        upstream_timeout_seconds: Timeout of one upstream attempt.
        attempts: Upstream attempts per call for transient failures.
    """

    upstream_base_url: str
    model: str
    api_key_file: str
    host: str = "127.0.0.1"
    port: int = 18300
    usage_log: str = ""
    upstream_timeout_seconds: float = 80.0
    attempts: int = 2


def _add_usage(total: Dict[str, int], usage: Any) -> None:
    if not isinstance(usage, dict):
        return
    for key in _USAGE_KEYS:
        value = usage.get(key)
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


def _handler(config: AdapterConfig) -> Type[BaseHTTPRequestHandler]:
    upstream_url = config.upstream_base_url.rstrip("/") + "/chat/completions"
    credential = Path(config.api_key_file).read_text().strip()
    opener = urllib.request.build_opener()
    log_lock = threading.Lock()

    def record(status: int, attempts: int, started: float, usage: Dict[str, int]) -> None:
        if not config.usage_log:
            return
        line = json.dumps({"at": time.time(), "model": config.model, "status": status, "attempts": attempts,
                           "elapsed_seconds": round(time.monotonic() - started, 3), "usage": usage})
        with log_lock, open(config.usage_log, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def call(payload: bytes, usage: Dict[str, int]) -> Tuple[int, Json, int]:
        """Return (status, body, attempts) after retrying transient failures."""
        status, message = 502, "upstream call was not attempted"
        for attempt in range(1, config.attempts + 1):
            request = urllib.request.Request(upstream_url, payload, {
                "Content-Type": "application/json", "Authorization": "Bearer " + credential})
            try:
                with opener.open(request, timeout=config.upstream_timeout_seconds) as response:
                    reply = json.loads(response.read())
                _add_usage(usage, reply.get("usage") if isinstance(reply, dict) else None)
                return 200, to_edits_response(reply if isinstance(reply, dict) else {}), attempt
            except urllib.error.HTTPError as exc:
                exc.close()
                # The upstream body may echo the credential or prompt; never forward it.
                status, message = exc.code, f"upstream HTTP {exc.code}"
                if exc.code < 500 and exc.code != 429:
                    return status, {"message": message}, attempt
            except NoImageReturned:
                status, message = 502, "upstream reply holds no image"
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
                status, message = 502, "upstream failure: " + type(exc).__name__
            logger.warning("attempt %s/%s failed: %s", attempt, config.attempts, message)
        return status, {"message": message}, config.attempts

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_: Any) -> None:
            pass

        def _json(self, status: int, body: Json) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": {"message": message, "type": "adapter_error", "code": status}})

        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"status": "ok", "model": config.model, "upstream": upstream_url})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": config.model, "object": "model"}]})
            else:
                self._error(404, "not found")

        def do_POST(self) -> None:
            if self.path != "/v1/images/edits":
                self._error(404, "not found")
                return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if not isinstance(body, dict):
                    raise UnsupportedRequest("request body must be a JSON object")
                payload = json.dumps(to_chat_request(body, config.model)).encode()
            except (json.JSONDecodeError, ValueError) as exc:
                self._error(400, str(exc) if isinstance(exc, UnsupportedRequest) else "invalid JSON body")
                return
            started = time.monotonic()
            usage: Dict[str, int] = {}
            status, result, attempts = call(payload, usage)
            record(status, attempts, started, usage)
            if status == 200:
                self._json(200, result)
            else:
                self._error(status, result["message"])

    return Handler


def make_server(config: AdapterConfig) -> ThreadingHTTPServer:
    """Build the adapter server without starting it.

    Args:
        config: Adapter settings.

    Returns:
        A server ready for ``serve_forever``.

    Raises:
        OSError: When the key file cannot be read.
    """
    server = ThreadingHTTPServer((config.host, config.port), _handler(config))
    server.daemon_threads = True
    return server


def main(argv: Optional[list] = None) -> None:
    """Run the adapter from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-base-url", required=True, help="e.g. https://host/api/v1")
    parser.add_argument("--model", required=True, help="upstream image model")
    parser.add_argument("--api-key-file", required=True)
    parser.add_argument("--host", default=AdapterConfig.host)
    parser.add_argument("--port", type=int, default=AdapterConfig.port)
    parser.add_argument("--usage-log", default=AdapterConfig.usage_log)
    parser.add_argument("--upstream-timeout-seconds", type=float, default=AdapterConfig.upstream_timeout_seconds)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = AdapterConfig(upstream_base_url=args.upstream_base_url, model=args.model,
                           api_key_file=args.api_key_file, host=args.host, port=args.port,
                           usage_log=args.usage_log, upstream_timeout_seconds=args.upstream_timeout_seconds)
    server = make_server(config)
    logger.info("listening on http://%s:%s -> %s (%s)", config.host, server.server_port,
                config.upstream_base_url.rstrip("/"), config.model)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
