"""Retry budget for transient upstream failures.

Three tries one and two seconds apart gave up within about ten seconds, so a
brief gateway outage failed a request, its shape and with it the whole figure.
Transient failures now get five tries with waits of 2, 4, 8 and 16 seconds;
rate limits keep their own configured budget and request errors are not retried.
A request the provider itself rejects is reported at once as HTTP 422.
"""
import re
from typing import Optional

TRANSIENT_STATUSES = (408, 500, 502, 503, 504)
TRANSIENT_ATTEMPTS = 5
MAX_WAIT_SECONDS = 16


def attempts(rate_limit_attempts: int) -> int:
    """Return how many tries the request loop must allow."""
    return max(TRANSIENT_ATTEMPTS, rate_limit_attempts)


def gives_up(status: Optional[int], attempt: int, rate_limit_attempts: int) -> bool:
    """Tell whether a failed try is the last one.

    Args:
        status: HTTP status of the failure, or ``None`` for a network failure.
        attempt: Zero-based index of the try that just failed.
        rate_limit_attempts: Configured number of tries for HTTP 429.

    Returns:
        ``True`` when the failure must be reported instead of retried.
    """
    if status == 429:
        return attempt >= rate_limit_attempts - 1
    if status is None or status in TRANSIENT_STATUSES:
        return attempt >= TRANSIENT_ATTEMPTS - 1
    return True


def wait_seconds(attempt: int) -> int:
    """Return the pause after the zero-based ``attempt`` failed."""
    return min(MAX_WAIT_SECONDS, 2 ** (attempt + 1))


REFUSED = 'API upstream refused the request'
REFUSED_STATUS = 422
_PROVIDER_STATUS = re.compile(r'upstream error (\d{3})')


def refused(status: Optional[int], detail: str) -> bool:
    """Tell whether the provider rejected this very request.

    The gateway wraps a provider client error as HTTP 502 with the text
    ``upstream error 4xx``. Sending the same request again gets the same
    answer, so it must not be retried like an outage.

    Args:
        status: HTTP status the gateway returned.
        detail: Start of the gateway's error body.

    Returns:
        ``True`` for a wrapped provider 4xx other than 408 and 429.
    """
    found = _PROVIDER_STATUS.search(detail) if status == 502 else None
    return bool(found) and found.group(1)[0] == '4' and found.group(1) not in ('408', '429')


def reply_status(message: str) -> int:
    """Return the HTTP status a failed request is answered with."""
    return REFUSED_STATUS if message == REFUSED else 502
