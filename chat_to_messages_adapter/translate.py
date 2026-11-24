"""Pure translation between OpenAI Chat Completions and the Anthropic-style Messages API.

Reasoning controls follow what the paperbypass gateway accepts for GLM on ``/messages``
(probed 2026-11-23): ``output_config.effort`` in low/medium/high works, while
``thinking: {type: enabled}`` without a budget and ``thinking: {type: disabled}`` are rejected.
Thinking is on by default, so "enabled" needs no field; "disabled" has no exact equivalent
and is approximated with low effort.
"""
from typing import Any, Dict, Iterable, List, Optional

Json = Dict[str, Any]

FINISH_REASONS: Dict[str, str] = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "content_filter",
}
UNSUPPORTED_FIELDS = ("tools", "tool_choice", "functions", "function_call", "response_format", "logprobs")
EFFORTS = ("low", "medium", "high")


class UnsupportedRequest(ValueError):
    """The chat request uses a feature this adapter does not translate."""


def _image_block(image_url: Any) -> Json:
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url:
        raise UnsupportedRequest("image_url part has no url")
    if not url.startswith("data:"):
        return {"type": "image", "source": {"type": "url", "url": url}}
    header, _, data = url.partition(",")
    media_type, _, encoding = header[len("data:"):].partition(";")
    if encoding != "base64" or not media_type or not data:
        raise UnsupportedRequest("only base64 data URLs are supported for images")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _content_blocks(content: Any) -> List[Json]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        raise UnsupportedRequest("message content must be a string or a list of parts")
    blocks: List[Json] = []
    for part in content:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind == "text":
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif kind == "image_url":
            blocks.append(_image_block(part.get("image_url")))
        else:
            raise UnsupportedRequest(f"unsupported content part type: {kind!r}")
    return blocks


def _system_text(content: Any) -> str:
    blocks = _content_blocks(content)
    if any(block["type"] != "text" for block in blocks):
        raise UnsupportedRequest("system messages must be text")
    return "".join(block["text"] for block in blocks)


def _effort(body: Json) -> Optional[str]:
    effort = body.get("reasoning_effort")
    if effort is not None:
        if effort not in EFFORTS:
            raise UnsupportedRequest(f"reasoning_effort must be one of {EFFORTS}")
        return effort
    thinking = body.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        return "low"
    return None


def to_messages_request(body: Json, default_max_tokens: int) -> Json:
    """Translate a Chat Completions request body into a Messages request body.

    Args:
        body: Parsed Chat Completions request.
        default_max_tokens: Used when the request sets neither token limit.

    Returns:
        The Messages request body, with ``stream`` copied from the request.

    Raises:
        UnsupportedRequest: When the request cannot be represented faithfully.
    """
    for name in UNSUPPORTED_FIELDS:
        if body.get(name):
            raise UnsupportedRequest(f"{name} is not supported by this adapter")
    if body.get("n", 1) != 1:
        raise UnsupportedRequest("only n=1 is supported")
    source = body.get("messages")
    if not isinstance(source, list) or not source:
        raise UnsupportedRequest("messages must be a non-empty list")

    system: List[str] = []
    messages: List[Json] = []
    for message in source:
        role = message.get("role") if isinstance(message, dict) else None
        if role in ("system", "developer"):
            system.append(_system_text(message.get("content")))
        elif role in ("user", "assistant"):
            messages.append({"role": role, "content": _content_blocks(message.get("content"))})
        else:
            raise UnsupportedRequest(f"unsupported message role: {role!r}")
    if not messages:
        raise UnsupportedRequest("at least one user or assistant message is required")

    result: Json = {
        "model": body.get("model"),
        "messages": messages,
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or default_max_tokens,
        "stream": bool(body.get("stream")),
    }
    if system:
        result["system"] = "\n\n".join(system)
    for name in ("temperature", "top_p"):
        if body.get(name) is not None:
            result[name] = body[name]
    stop = body.get("stop")
    if stop:
        result["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
    effort = _effort(body)
    if effort is not None:
        result["output_config"] = {"effort": effort}
    return result


def _usage(usage: Json) -> Json:
    prompt = sum(usage.get(name) or 0 for name in
                 ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
    completion = usage.get("output_tokens") or 0
    result: Json = {"prompt_tokens": prompt, "completion_tokens": completion,
                    "total_tokens": prompt + completion}
    thinking = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
    if thinking is not None:
        result["completion_tokens_details"] = {"reasoning_tokens": thinking}
    return result


class StreamTranslator:
    """Turns Messages stream events into Chat Completions chunks, one event at a time."""

    def __init__(self) -> None:
        self.finished = False
        self._meta: Json = {"id": None, "model": None}
        self._usage: Json = {}
        self._finish: Optional[str] = None

    def _chunk(self, **fields: Any) -> Json:
        return {"object": "chat.completion.chunk", **self._meta, **fields}

    def _delta(self, delta: Json, finish: Optional[str] = None) -> Json:
        return self._chunk(choices=[{"index": 0, "delta": delta, "finish_reason": finish}])

    def feed(self, event: Json) -> List[Json]:
        """Consume one upstream event.

        Args:
            event: Parsed ``data:`` payload of a Messages stream event.

        Returns:
            Zero or more Chat Completions chunks to forward.
        """
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message") or {}
            self._meta = {"id": message.get("id"), "model": message.get("model")}
            self._usage.update(message.get("usage") or {})
            return []
        if kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta" and delta.get("text"):
                return [self._delta({"content": delta["text"]})]
            if delta.get("type") == "thinking_delta" and delta.get("thinking"):
                return [self._delta({"reasoning_content": delta["thinking"]})]
            return []
        if kind == "message_delta":
            self._usage.update({k: v for k, v in (event.get("usage") or {}).items() if v is not None})
            reason = (event.get("delta") or {}).get("stop_reason")
            if reason:
                self._finish = FINISH_REASONS.get(reason, reason)
            return []
        if kind == "message_stop":
            if self._finish is None:
                return []
            self.finished = True
            return [self._delta({}, self._finish), self._chunk(choices=[], usage=_usage(self._usage))]
        if kind == "error":
            return [self._chunk(choices=[], error=event.get("error") or {"type": "upstream_error"})]
        return []


def aggregate(chunks: Iterable[Json]) -> Json:
    """Fold translated chunks into one non-streaming Chat Completions response.

    Args:
        chunks: Chunks produced by ``StreamTranslator`` for a finished stream.

    Returns:
        A ``chat.completion`` object.
    """
    content: List[str] = []
    reasoning: List[str] = []
    result: Json = {"object": "chat.completion", "id": None, "model": None, "usage": None}
    finish: Optional[str] = None
    for chunk in chunks:
        result["id"], result["model"] = chunk.get("id"), chunk.get("model")
        result["usage"] = chunk.get("usage") or result["usage"]
        for choice in chunk.get("choices", []):
            finish = choice.get("finish_reason") or finish
            content.append(choice["delta"].get("content", ""))
            reasoning.append(choice["delta"].get("reasoning_content", ""))
    result["choices"] = [{"index": 0, "finish_reason": finish, "message": {
        "role": "assistant", "content": "".join(content), "reasoning_content": "".join(reasoning)}}]
    return result
