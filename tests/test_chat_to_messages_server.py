"""End-to-end behaviour of the adapter HTTP server against a local stand-in upstream."""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from chat_to_messages_adapter.server import AdapterConfig, make_server

SSE_OK = (
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_9","model":"m","usage":{}}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"thinking_delta","thinking":"t"}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,'
    '"delta":{"type":"text_delta","text":"ok"}}\n\n'
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"input_tokens":3,"output_tokens":2}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


class Upstream:
    """Records what the adapter sends and replies with a canned response."""

    def __init__(self):
        self.requests = []
        self.reply = (200, {"Content-Type": "text/event-stream"}, SSE_OK)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"),
                                       "version": self.headers.get("anthropic-version"), "body": body})
                status, headers, text = outer.reply
                data = text.encode()
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/v1"


@pytest.fixture()
def upstream():
    stand_in = Upstream()
    yield stand_in
    stand_in.server.shutdown()
    stand_in.server.server_close()


@pytest.fixture()
def adapter_url(upstream):
    server = make_server(AdapterConfig(upstream_base_url=upstream.url, port=0))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def post(url, body, token="secret-token"):
    request = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json", "Authorization": "Bearer " + token})
    return urllib.request.urlopen(request, timeout=10)


def chat(**extra):
    return {"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}], **extra}


def sse_data(response):
    return [line[5:].strip() for line in response.read().decode().splitlines() if line.startswith("data:")]


def test_streaming_request_is_relayed_as_openai_sse(upstream, adapter_url):
    with post(adapter_url, chat(stream=True)) as response:
        assert "text/event-stream" in response.headers["Content-Type"]
        data = sse_data(response)
    assert data[-1] == "[DONE]"
    chunks = [json.loads(item) for item in data[:-1]]
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert "".join(d.get("content", "") for d in deltas) == "ok"
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "t"
    assert chunks[-1]["usage"]["total_tokens"] == 5
    sent = upstream.requests[0]
    assert sent["path"] == "/api/v1/messages"
    assert sent["auth"] == "Bearer secret-token"
    assert sent["version"] == "2023-06-01"
    assert sent["body"]["stream"] is True and sent["body"]["max_tokens"] == 50


def test_non_streaming_request_returns_one_chat_completion(upstream, adapter_url):
    with post(adapter_url, chat()) as response:
        result = json.loads(response.read())
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"]["content"] == "ok"
    assert upstream.requests[0]["body"]["stream"] is True


def test_upstream_http_error_status_and_retry_after_are_passed_through(upstream, adapter_url):
    upstream.reply = (429, {"Content-Type": "application/json", "Retry-After": "7"},
                      '{"error":"slow down secret-token"}')
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(adapter_url, chat(stream=True))
    assert caught.value.code == 429
    assert caught.value.headers["Retry-After"] == "7"
    assert "secret-token" not in caught.value.read().decode()


def test_truncated_upstream_stream_never_emits_done(upstream, adapter_url):
    truncated = SSE_OK.split("event: message_delta")[0]
    upstream.reply = (200, {"Content-Type": "text/event-stream"}, truncated)
    with post(adapter_url, chat(stream=True)) as response:
        assert "[DONE]" not in sse_data(response)


def test_upstream_non_sse_success_is_a_bad_gateway(upstream, adapter_url):
    upstream.reply = (200, {"Content-Type": "application/json"}, '{"success":false}')
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(adapter_url, chat(stream=True))
    assert caught.value.code == 502


def test_unsupported_request_is_rejected_without_calling_upstream(upstream, adapter_url):
    with pytest.raises(urllib.error.HTTPError) as caught:
        post(adapter_url, chat(tools=[{"type": "function", "function": {"name": "f"}}]))
    assert caught.value.code == 400
    assert upstream.requests == []


def test_missing_authorization_is_rejected(upstream, adapter_url):
    request = urllib.request.Request(adapter_url + "/v1/chat/completions", json.dumps(chat()).encode(),
                                     {"Content-Type": "application/json"})
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=10)
    assert caught.value.code == 401
    assert upstream.requests == []


def test_health_endpoint_reports_ready(adapter_url):
    with urllib.request.urlopen(adapter_url + "/health", timeout=10) as response:
        assert json.loads(response.read())["status"] == "ok"
