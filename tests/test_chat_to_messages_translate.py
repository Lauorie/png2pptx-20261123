"""Behaviour of the pure chat-completions <-> messages translation layer."""
import pytest

from chat_to_messages_adapter.translate import (
    StreamTranslator,
    UnsupportedRequest,
    aggregate,
    to_messages_request,
)


def chat_body(**extra):
    body = {"model": "z-ai/glm-5.3-flash", "max_tokens": 1000,
            "messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return body


def test_system_messages_become_top_level_system():
    body = chat_body(messages=[{"role": "system", "content": "Be terse."},
                               {"role": "user", "content": "hi"}])
    out = to_messages_request(body, default_max_tokens=8192)
    assert out["system"] == "Be terse."
    assert out["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


def test_data_url_image_becomes_base64_source():
    parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD", "detail": "high"}},
             {"type": "text", "text": "describe"}]
    out = to_messages_request(chat_body(messages=[{"role": "user", "content": parts}]), 8192)
    assert out["messages"][0]["content"] == [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"}},
        {"type": "text", "text": "describe"}]


def test_remote_image_url_becomes_url_source():
    parts = [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]
    out = to_messages_request(chat_body(messages=[{"role": "user", "content": parts}]), 8192)
    assert out["messages"][0]["content"] == [
        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}]


def test_reasoning_effort_maps_to_output_config_and_thinking_enabled_is_omitted():
    out = to_messages_request(chat_body(thinking={"type": "enabled"}, reasoning_effort="high"), 8192)
    assert out["output_config"] == {"effort": "high"}
    assert "thinking" not in out and "reasoning_effort" not in out


def test_thinking_enabled_without_effort_sends_no_reasoning_controls():
    out = to_messages_request(chat_body(thinking={"type": "enabled"}), 8192)
    assert "thinking" not in out and "output_config" not in out


def test_thinking_disabled_maps_to_low_effort():
    out = to_messages_request(chat_body(thinking={"type": "disabled"}), 8192)
    assert out["output_config"] == {"effort": "low"}
    assert "thinking" not in out


def test_sampling_and_limits_are_carried_and_openai_only_fields_dropped():
    out = to_messages_request(chat_body(temperature=0.2, top_p=0.9, seed=7, stop=["END"], stream=True,
                                        stream_options={"include_usage": True}, service_tier="auto"), 8192)
    assert out["max_tokens"] == 1000 and out["temperature"] == 0.2 and out["top_p"] == 0.9
    assert out["stop_sequences"] == ["END"] and out["stream"] is True
    assert not {"seed", "stream_options", "service_tier", "stop"} & out.keys()


def test_max_tokens_falls_back_to_max_completion_tokens_then_default():
    body = chat_body(max_completion_tokens=77)
    del body["max_tokens"]
    assert to_messages_request(body, 8192)["max_tokens"] == 77
    del body["max_completion_tokens"]
    assert to_messages_request(body, 8192)["max_tokens"] == 8192


@pytest.mark.parametrize("body", [
    chat_body(tools=[{"type": "function", "function": {"name": "f"}}]),
    chat_body(n=2),
    chat_body(messages=[{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}]),
    chat_body(messages=[]),
])
def test_unsupported_requests_are_rejected(body):
    with pytest.raises(UnsupportedRequest):
        to_messages_request(body, 8192)


MESSAGES_STREAM = [
    {"type": "message_start", "message": {"id": "msg_1", "model": "z-ai/glm-5.3-flash",
                                          "usage": {"input_tokens": 0, "output_tokens": 0}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hm"}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "x"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "ping"},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "o"}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "k"}},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
     "usage": {"input_tokens": 19, "output_tokens": 75, "output_tokens_details": {"thinking_tokens": 72}}},
    {"type": "message_stop"},
]


def run_stream(events):
    translator = StreamTranslator()
    chunks = [chunk for event in events for chunk in translator.feed(event)]
    return translator, chunks


def test_stream_is_translated_to_chat_completion_chunks():
    translator, chunks = run_stream(MESSAGES_STREAM)
    assert translator.finished
    assert all(c["object"] == "chat.completion.chunk" and c["id"] == "msg_1"
               and c["model"] == "z-ai/glm-5.3-flash" for c in chunks)
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "hm"
    assert "".join(d.get("content", "") for d in deltas) == "ok"
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "stop"
    assert chunks[-1]["choices"] == []
    assert chunks[-1]["usage"] == {"prompt_tokens": 19, "completion_tokens": 75, "total_tokens": 94,
                                   "completion_tokens_details": {"reasoning_tokens": 72}}


@pytest.mark.parametrize("stop_reason,finish", [
    ("end_turn", "stop"), ("stop_sequence", "stop"), ("max_tokens", "length"), ("refusal", "content_filter")])
def test_stop_reasons_map_to_finish_reasons(stop_reason, finish):
    events = [MESSAGES_STREAM[0], {"type": "message_delta", "delta": {"stop_reason": stop_reason},
                                   "usage": {"input_tokens": 1, "output_tokens": 1}}, {"type": "message_stop"}]
    _, chunks = run_stream(events)
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]] == [finish]


def test_upstream_error_event_becomes_error_chunk_and_stream_is_not_finished():
    translator, chunks = run_stream([MESSAGES_STREAM[0],
                                     {"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}])
    assert chunks[-1]["error"]["type"] == "overloaded_error"
    assert not translator.finished


def test_truncated_stream_is_not_finished():
    translator, _ = run_stream(MESSAGES_STREAM[:8])
    assert not translator.finished


def test_aggregate_builds_a_non_streaming_chat_completion():
    _, chunks = run_stream(MESSAGES_STREAM)
    result = aggregate(chunks)
    assert result["object"] == "chat.completion" and result["id"] == "msg_1"
    assert result["choices"] == [{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": "ok", "reasoning_content": "hm"}}]
    assert result["usage"]["total_tokens"] == 94
