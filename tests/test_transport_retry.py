"""A short upstream outage is ridden out instead of failing a whole figure."""
import io
import queue
import threading
import urllib.error
from pathlib import Path
from typing import Any, Iterator, List
from unittest.mock import patch

import pytest

from full_recovery.api_gateway import ApiRelay
from full_recovery.transport_retry import TRANSIENT_ATTEMPTS, gives_up, wait_seconds


class _Response:
    headers = {'Content-Type': 'text/event-stream'}

    def __enter__(self) -> '_Response':
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def __iter__(self) -> Iterator[bytes]:
        yield b'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":"stop"}]}\n'
        yield b'data: [DONE]\n'


def _error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError('https://example.invalid', status, 'upstream', {}, io.BytesIO(b'{}'))


def _run(tmp_path: Path, replies: List[Any]) -> ApiRelay:
    key = tmp_path / 'key'
    key.write_text('test-key')
    key.chmod(0o600)
    relay = ApiRelay({'api_key_file': str(key), 'api_global_concurrency': 2, 'api_model': 'test',
                      'api_base_url': 'https://example.invalid/v1',
                      'api_request_start_interval_seconds': .001}, tmp_path)
    cancelled = threading.Event()
    relay.output = queue.Queue()
    with patch('full_recovery.api_gateway.urllib.request.build_opener') as build, \
            patch.object(cancelled, 'wait', return_value=False) as waits:
        build.return_value.open.side_effect = replies
        relay.produce({'messages': []}, relay.output, cancelled=cancelled)
    # The request pacer shares this event for sub-second spacing waits.
    relay.waits = [call.args[0] for call in waits.call_args_list if call.args[0] >= 1]
    return relay


def test_four_bad_gateway_replies_are_retried_until_the_service_answers(tmp_path):
    relay = _run(tmp_path, [_error(502) for _ in range(4)] + [_Response()])
    assert relay.output.queue[-1][0] == 'complete'
    assert relay.stats['failed'] == 0 and relay.stats['completed'] == 1


def test_four_network_failures_are_retried_until_the_service_answers(tmp_path):
    relay = _run(tmp_path, [ConnectionError('reset') for _ in range(4)] + [_Response()])
    assert relay.output.queue[-1][0] == 'complete'


def test_a_persistent_outage_still_fails_after_the_retry_budget(tmp_path):
    relay = _run(tmp_path, [_error(502) for _ in range(10)])
    assert relay.stats['failed'] == 1 and relay.stats['completed'] == 0
    assert relay.stats['upstream_attempts'] == TRANSIENT_ATTEMPTS


def test_waits_between_transient_retries_grow_to_cover_a_half_minute_outage(tmp_path):
    relay = _run(tmp_path, [_error(503) for _ in range(4)] + [_Response()])
    assert relay.waits == sorted(relay.waits) and sum(relay.waits) >= 30


def test_a_request_error_is_not_retried(tmp_path):
    relay = _run(tmp_path, [_error(400), _Response()])
    assert relay.stats['failed'] == 1 and relay.stats['upstream_attempts'] == 1


@pytest.mark.parametrize('status', [500, 502, 503, 504, 408])
def test_transient_statuses_use_the_transient_budget(status):
    assert not gives_up(status, TRANSIENT_ATTEMPTS - 2, rate_limit_attempts=3)
    assert gives_up(status, TRANSIENT_ATTEMPTS - 1, rate_limit_attempts=3)


def test_rate_limits_keep_their_own_budget():
    assert not gives_up(429, 5, rate_limit_attempts=8)
    assert gives_up(429, 7, rate_limit_attempts=8)


def test_wait_is_bounded():
    assert wait_seconds(0) == 2 and wait_seconds(20) == 16


def _refusal(provider_status: int) -> urllib.error.HTTPError:
    body = b'{"success":false,"error":{"message":"upstream error %d","reason":"upstream_error"}}' % provider_status
    return urllib.error.HTTPError('https://example.invalid', 502, 'upstream', {}, io.BytesIO(body))


def test_a_request_the_provider_refuses_is_not_sent_again(tmp_path):
    from full_recovery.transport_retry import REFUSED
    relay = _run(tmp_path, [_refusal(403), _Response()])
    assert relay.output.queue[-1] == ('failed', REFUSED)
    assert relay.stats['upstream_attempts'] == 1 and relay.waits == []


@pytest.mark.parametrize('detail, expected', [
    ('{"error":{"message":"upstream error 403"}}', True),
    ('{"error":{"message":"upstream error 400"}}', True),
    ('{"error":{"message":"upstream error 429"}}', False),
    ('{"error":{"message":"upstream error 408"}}', False),
    ('{"error":{"message":"upstream error 503"}}', False),
    ('', False),
])
def test_only_a_provider_client_error_counts_as_a_refusal(detail, expected):
    from full_recovery.transport_retry import refused
    assert refused(502, detail) is expected


def test_a_refusal_reaches_the_caller_as_unprocessable_instead_of_bad_gateway():
    from full_recovery.transport_retry import REFUSED, reply_status
    assert reply_status(REFUSED) == 422 and reply_status('API HTTP 502') == 502
