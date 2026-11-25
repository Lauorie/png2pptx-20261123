"""Tests for the images/edits -> chat completions adapter."""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import pytest

from images_edits_adapter import (AdapterConfig, NoImageReturned, UnsupportedRequest, make_server,
                                  to_chat_request, to_edits_response)

DATA_URL = "data:image/png;base64,QUJD"
EDIT_BODY: Dict[str, Any] = {
    "model": "SenseNova-U1.5-8B-MoT", "prompt": "remove the label", "images": [{"image_url": DATA_URL}],
    "size": "1024x1024", "output_format": "png", "response_format": "b64_json",
    "img_config": {"width": 1024, "height": 1024, "seed": 0}}
CHAT_REPLY: Dict[str, Any] = {
    "created": 7, "model": "vendor/image-model",
    "choices": [{"message": {"role": "assistant", "content": None,
                             "images": [{"type": "image_url", "image_url": {"url": DATA_URL}}]}}],
    "usage": {"prompt_tokens": 1100, "completion_tokens": 1120, "total_tokens": 2220}}


def test_request_uses_configured_model_and_asks_for_an_image() -> None:
    chat = to_chat_request(EDIT_BODY, "vendor/image-model")
    assert chat["model"] == "vendor/image-model"
    assert chat["modalities"] == ["image", "text"]
    assert chat["messages"] == [{"role": "user", "content": [
        {"type": "text", "text": "remove the label"},
        {"type": "image_url", "image_url": {"url": DATA_URL}}]}]


def test_square_size_requests_square_output_and_other_sizes_do_not() -> None:
    assert to_chat_request(EDIT_BODY, "m")["image_config"] == {"aspect_ratio": "1:1"}
    assert "image_config" not in to_chat_request({**EDIT_BODY, "size": "1024x512"}, "m")


def test_image_url_may_be_given_as_object() -> None:
    body = {**EDIT_BODY, "images": [{"image_url": {"url": DATA_URL}}]}
    assert to_chat_request(body, "m")["messages"][0]["content"][1]["image_url"]["url"] == DATA_URL


@pytest.mark.parametrize("patch", [
    {"prompt": ""}, {"prompt": 3}, {"images": []}, {"images": [{"image_url": "https://example.com/a.png"}]},
    {"response_format": "url"}])
def test_requests_the_upstream_cannot_honour_are_rejected(patch: Dict[str, Any]) -> None:
    with pytest.raises(UnsupportedRequest):
        to_chat_request({**EDIT_BODY, **patch}, "m")


def test_response_carries_inline_image_and_usage() -> None:
    edits = to_edits_response(CHAT_REPLY)
    assert edits["data"] == [{"b64_json": "QUJD"}]
    assert edits["usage"] == CHAT_REPLY["usage"]
    assert edits["created"] == 7


def test_response_without_an_image_is_an_error() -> None:
    reply = {"choices": [{"message": {"role": "assistant", "content": "I cannot do that."}}]}
    with pytest.raises(NoImageReturned):
        to_edits_response(reply)


class _Upstream:
    """Scripted chat completions upstream recording what it receives."""

    def __init__(self, replies: List[Tuple[int, Dict[str, Any]]]) -> None:
        self.replies = list(replies)
        self.seen: List[Dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_: Any) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append({"path": self.path, "authorization": self.headers.get("Authorization"),
                                   "body": body})
                status, reply = outer.replies.pop(0)
                data = json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/api/v1"


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[Any]:
    started: List[Any] = []

    def start(replies: List[Tuple[int, Dict[str, Any]]]) -> Tuple[_Upstream, str, Path]:
        upstream = _Upstream(replies)
        key_file = tmp_path / "key"
        key_file.write_text("test-credential\n")
        usage_log = tmp_path / "usage.jsonl"
        server = make_server(AdapterConfig(upstream_base_url=upstream.base_url, model="vendor/image-model",
                                           api_key_file=str(key_file), port=0, usage_log=str(usage_log)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.extend([upstream.server, server])
        return upstream, f"http://127.0.0.1:{server.server_port}", usage_log

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


def _post(url: str, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    request = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_edit_is_served_from_chat_completions_with_the_file_credential(stack: Any) -> None:
    upstream, url, usage_log = stack([(200, CHAT_REPLY)])
    status, body = _post(url + "/v1/images/edits", EDIT_BODY)
    assert status == 200
    assert body["data"] == [{"b64_json": "QUJD"}]
    assert upstream.seen[0]["path"] == "/api/v1/chat/completions"
    assert upstream.seen[0]["authorization"] == "Bearer test-credential"
    assert upstream.seen[0]["body"]["model"] == "vendor/image-model"
    record = json.loads(usage_log.read_text().strip())
    assert record["status"] == 200 and record["model"] == "vendor/image-model"
    assert record["usage"]["completion_tokens"] == 1120 and record["attempts"] == 1


def test_usage_log_never_contains_prompt_image_or_credential(stack: Any) -> None:
    _, url, usage_log = stack([(200, CHAT_REPLY)])
    _post(url + "/v1/images/edits", EDIT_BODY)
    text = usage_log.read_text()
    assert "remove the label" not in text and "QUJD" not in text and "test-credential" not in text


def test_transient_upstream_failure_is_retried_once(stack: Any) -> None:
    upstream, url, usage_log = stack([(503, {"error": "busy"}), (200, CHAT_REPLY)])
    status, _ = _post(url + "/v1/images/edits", EDIT_BODY)
    assert status == 200 and len(upstream.seen) == 2
    assert json.loads(usage_log.read_text().strip())["attempts"] == 2


def test_reply_without_image_is_retried_then_reported(stack: Any) -> None:
    refusal = {"choices": [{"message": {"content": "no"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}
    upstream, url, usage_log = stack([(200, refusal), (200, refusal)])
    status, body = _post(url + "/v1/images/edits", EDIT_BODY)
    assert status == 502 and "image" in body["error"]["message"]
    assert len(upstream.seen) == 2
    record = json.loads(usage_log.read_text().strip())
    assert record["status"] == 502 and record["usage"]["prompt_tokens"] == 10


def test_client_error_is_not_retried_and_upstream_body_is_not_forwarded(stack: Any) -> None:
    upstream, url, _ = stack([(401, {"error": "bad key test-credential"})])
    status, body = _post(url + "/v1/images/edits", EDIT_BODY)
    assert status == 401 and len(upstream.seen) == 1
    assert "test-credential" not in json.dumps(body)


def test_invalid_request_is_rejected_without_calling_upstream(stack: Any) -> None:
    upstream, url, _ = stack([])
    status, _ = _post(url + "/v1/images/edits", {**EDIT_BODY, "images": []})
    assert status == 400 and upstream.seen == []


def test_health_and_models_report_the_configured_model(stack: Any) -> None:
    _, url, _ = stack([])
    with urllib.request.urlopen(url + "/health", timeout=10) as response:
        assert json.loads(response.read())["model"] == "vendor/image-model"
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
        assert json.loads(response.read())["data"][0]["id"] == "vendor/image-model"
