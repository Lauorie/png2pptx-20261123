"""Local adapter exposing an OpenAI Chat Completions endpoint on top of a Messages API."""
from .server import AdapterConfig, make_server
from .translate import StreamTranslator, UnsupportedRequest, aggregate, to_messages_request

__all__ = [
    "AdapterConfig",
    "make_server",
    "StreamTranslator",
    "UnsupportedRequest",
    "aggregate",
    "to_messages_request",
]
