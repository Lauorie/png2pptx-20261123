"""Optional run-wide bounds on a single model request.

All are off unless configured. The output cap stops a request that reasons
without end from running to the backend's 100,000-token budget; the idle
timeout replaces the gateway's 5-minute wait for a stream that has gone silent.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

MAX_TOKENS_KEY = 'api_max_tokens'
IDLE_TIMEOUT_KEY = 'api_idle_timeout_seconds'
HEDGE_AFTER_KEY = 'api_hedge_after_seconds'


def _positive(value: Any, name: str, kinds: tuple) -> None:
    if isinstance(value, bool) or not isinstance(value, kinds) or value <= 0:
        raise ValueError(f'{name} must be a positive number')


def to_settings(settings: Mapping[str, Any], max_tokens: Optional[int],
                idle_timeout_seconds: Optional[float],
                hedge_after_seconds: Optional[float] = None) -> Dict[str, Any]:
    """Return run settings with the request limits recorded.

    Args:
        settings: Existing run settings; not modified.
        max_tokens: Output cap for every request, or None for the backends' own caps.
        idle_timeout_seconds: Seconds a stream may stay silent, or None for the gateway default.
        hedge_after_seconds: Seconds before an unanswered request is sent a second time
            (see request_hedge), or None to never do so.

    Returns:
        A new settings dictionary.

    Raises:
        ValueError: When a limit is not a positive number.
    """
    result = dict(settings)
    if max_tokens is not None:
        _positive(max_tokens, 'max_tokens', (int,))
        result[MAX_TOKENS_KEY] = max_tokens
    if idle_timeout_seconds is not None:
        _positive(idle_timeout_seconds, 'idle_timeout_seconds', (int, float))
        result[IDLE_TIMEOUT_KEY] = idle_timeout_seconds
    if hedge_after_seconds is not None:
        _positive(hedge_after_seconds, 'hedge_after_seconds', (int, float))
        result[HEDGE_AFTER_KEY] = hedge_after_seconds
    return result


def capped(body: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the run's output cap to one request; the cap never raises a smaller one.

    Args:
        body: Request body; not modified.
        settings: Run settings.

    Returns:
        The request body to send.
    """
    result = dict(body)
    cap = settings.get(MAX_TOKENS_KEY)
    if cap is None:
        return result
    own = result.get('max_tokens')
    result['max_tokens'] = min(own, cap) if type(own) is int else cap
    return result
