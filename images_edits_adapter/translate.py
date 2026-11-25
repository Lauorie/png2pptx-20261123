"""Pure translation between the JSON ``images/edits`` protocol and chat completions."""
from typing import Any, Dict, List

Json = Dict[str, Any]

_DATA_PREFIX = "data:image/"


class UnsupportedRequest(ValueError):
    """The edit request cannot be expressed as an image-output chat completion."""


class NoImageReturned(ValueError):
    """The upstream reply holds no inline image."""


def _image_url(entry: Any) -> str:
    value = entry.get("image_url") if isinstance(entry, dict) else None
    if isinstance(value, dict):
        value = value.get("url")
    if not isinstance(value, str) or not value.startswith(_DATA_PREFIX):
        raise UnsupportedRequest("each image must be an inline data:image/ URL")
    return value


def to_chat_request(body: Json, model: str) -> Json:
    """Build the chat completions request for one edit.

    Args:
        body: JSON body of the ``images/edits`` call.
        model: Upstream image model; the model named by the client is ignored.

    Returns:
        A chat completions request asking for image output.

    Raises:
        UnsupportedRequest: When the prompt or images are missing or not inline.
    """
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise UnsupportedRequest("prompt must be a non-empty string")
    images = body.get("images")
    if not isinstance(images, list) or not images:
        raise UnsupportedRequest("images must be a non-empty list")
    if body.get("response_format", "b64_json") != "b64_json":
        raise UnsupportedRequest("only response_format b64_json is supported")
    content: List[Json] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": _image_url(entry)}} for entry in images)
    request: Json = {"model": model, "messages": [{"role": "user", "content": content}],
                     "modalities": ["image", "text"]}
    width, _, height = str(body.get("size", "")).partition("x")
    if width and width == height:
        request["image_config"] = {"aspect_ratio": "1:1"}
    return request


def to_edits_response(reply: Json) -> Json:
    """Convert a chat completions reply into an ``images/edits`` response.

    Args:
        reply: Upstream chat completions JSON.

    Returns:
        ``{"created", "data": [{"b64_json"}], "usage"}``.

    Raises:
        NoImageReturned: When the reply has no inline image.
    """
    choices = reply.get("choices") or [{}]
    message = choices[0].get("message") or {}
    for entry in message.get("images") or []:
        value = entry.get("image_url") if isinstance(entry, dict) else None
        if isinstance(value, dict):
            value = value.get("url")
        if isinstance(value, str) and value.startswith(_DATA_PREFIX) and "," in value:
            return {"created": reply.get("created"), "data": [{"b64_json": value.split(",", 1)[1]}],
                    "usage": reply.get("usage")}
    raise NoImageReturned("upstream reply holds no inline image")
