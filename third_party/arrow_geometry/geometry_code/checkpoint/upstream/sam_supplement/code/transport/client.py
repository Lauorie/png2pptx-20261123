from request_budget import counted
"""One auditable Boyue request, with an end-to-end streaming deadline."""
import asyncio
import base64
import json
from pathlib import Path
import time

import httpx

from .common import sha256, write_json
from .credentials import load_credentials
from .coordinate_precision import normalize_prompt


class TransportError(RuntimeError):
    def __init__(self, message, status_code=None, retryable=False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


def image_list(images):
    return [("CLEAN", Path(images))] if isinstance(images, (str, Path)) else [(label, Path(path)) for label, path in images]


def build_payload(config, system, prompt, image):
    prompt = normalize_prompt(prompt)
    content = [{"type": "text", "text": prompt if isinstance(prompt, str) else json.dumps(prompt, ensure_ascii=False)}]
    for label, path in image_list(image):
        content += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")}}]
    payload = {"model": config["model"], "temperature": config["temperature"],
            "max_tokens": config["max_tokens"], "thinking": config["thinking"],
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}
    if config.get("reasoning_effort") is not None:
        payload["reasoning_effort"] = config["reasoning_effort"]
    return payload


def retryable_http(status,body,config):
    if status in (408,429,500,502,503,504):return True
    # Exact saved requests rejected as 1210 succeeded unchanged on replay.
    # Retry only this observed provider code, under the existing retry budget;
    # other parameter/auth errors remain non-retryable.
    if status==400 and config.get('retry_provider_1210',False):
        try:return str(json.loads(body).get('error',{}).get('code'))=='1210'
        except (ValueError,TypeError,AttributeError):return False
    return False


@counted
def request(config, system, prompt, image, output_dir):
    base, key, auth = load_credentials(config["base_url"])
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    payload = build_payload(config, system, prompt, image)
    write_json(folder / "wire_request.json", payload)
    write_json(folder / "request_metadata.json", {
        "base_url": base, "auth": auth,
        "images": [{"label": label, "path": str(path.resolve()), "sha256": sha256(path)} for label, path in image_list(image)],
        "timeout_seconds": config["request_timeout_seconds"], "idle_timeout_seconds":config.get('stream_idle_timeout_seconds',120),
        "transport": "SSE with total and idle deadlines", "automatic_retries": 0})
    started = time.monotonic()
    parts, reasoning = [], []
    state = {"state": "requesting", "sse_done": False, "finish_reason": None, "usage": None,
             "first_event_seconds":None,"last_event_seconds":None,"reasoning_chars":0,"content_chars":0}
    write_json(folder / "status.json", state)

    async def send():
        timeout = httpx.Timeout(config["request_timeout_seconds"], connect=15, write=60,
                                read=config.get('stream_idle_timeout_seconds',120))
        async with httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False) as client:
            async with client.stream("POST", base + "/chat/completions", json=payload,
                                     headers={"Authorization": "Bearer " + key}) as response:
                state["http_status"] = response.status_code
                if response.status_code != 200:
                    await response.aread()
                    body = response.text.replace(key, "<REDACTED>")
                    (folder / "http_response.txt").write_text(body)
                    raise TransportError(f"HTTP {response.status_code}: {body[:500]}", response.status_code,
                                         retryable=retryable_http(response.status_code,body,config))
                last = time.monotonic()
                with (folder / "events.jsonl").open("w") as events, (folder / "http_response.txt").open("w") as raw:
                    async for line in response.aiter_lines():
                        if time.monotonic() - started >= config["request_timeout_seconds"]:
                            raise TimeoutError("Total request deadline exceeded")
                        line = line.replace(key, "<REDACTED>")
                        raw.write(line + "\n")
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            state["sse_done"] = True
                            break
                        event = json.loads(data)
                        elapsed=time.monotonic()-started
                        state['last_event_seconds']=elapsed
                        if state['first_event_seconds'] is None:state['first_event_seconds']=elapsed
                        events.write(json.dumps(dict(elapsed_seconds=elapsed,event=event), ensure_ascii=False) + "\n")
                        if event.get("error"):
                            raise TransportError(json.dumps(event["error"], ensure_ascii=False))
                        state["response_model"] = event.get("model", state.get("response_model"))
                        state["usage"] = event.get("usage") or state["usage"]
                        for choice in event.get("choices", []):
                            state["finish_reason"] = choice.get("finish_reason") or state["finish_reason"]
                            delta = choice.get("delta") or {}
                            text = delta.get("content") or ""
                            think = delta.get("reasoning_content") or delta.get("reasoning") or ""
                            if not isinstance(text, str) or not isinstance(think, str):
                                raise TransportError("Non-text stream delta")
                            parts.append(text)
                            reasoning.append(think)
                            state['content_chars']+=len(text);state['reasoning_chars']+=len(think)
                        if time.monotonic() - last > 5:
                            raw.flush()
                            events.flush()
                            write_json(folder / "status.json", {**state, "seconds": time.monotonic() - started})
                            last = time.monotonic()

    async def bounded():
        await asyncio.wait_for(send(), timeout=config["request_timeout_seconds"])

    try:
        asyncio.run(bounded())
        if not state["sse_done"] or state["finish_reason"] != "stop":
            raise TransportError("Incomplete stream; partial text is not a valid result",retryable=state['finish_reason'] in (None,'network_error'))
        state["state"] = "transport_complete"
        return "".join(parts)
    except Exception as exc:
        state["state"] = "failed"
        state['failure_kind']=('idle_timeout' if isinstance(exc,httpx.ReadTimeout) else
            'total_deadline' if isinstance(exc,(TimeoutError,asyncio.TimeoutError)) else
            'network_error' if isinstance(exc,httpx.TransportError) else 'protocol_or_http_error')
        write_json(folder / "error.json", {"error_type": type(exc).__name__, "error": str(exc).replace(key, "<REDACTED>")})
        if isinstance(exc, TransportError):
            raise
        raise TransportError(f"{type(exc).__name__}: {str(exc).replace(key, '<REDACTED>')}",
                             retryable=isinstance(exc,httpx.TransportError)) from None
    finally:
        (folder / "content.txt").write_text("".join(parts), encoding="utf-8")
        (folder / "reasoning.txt").write_text("".join(reasoning), encoding="utf-8")
        write_json(folder / "status.json", {**state, "seconds": time.monotonic() - started})
