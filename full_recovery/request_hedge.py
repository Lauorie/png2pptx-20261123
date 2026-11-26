"""Ask a model request a second time when the first one falls far behind.

Most requests of a run answer within a minute, but now and then one reasons
on for many minutes or its stream slows to a crawl, and the whole stage waits
for it. With a delay configured, a request still unanswered after that long is
sent again alongside the first; whichever completes first is delivered and the
other is cancelled. An answer cut off at the output cap waits for the other
request and is only used when that one fails.

Events are held back until the winner is known, so a consumer sees one whole
answer, never a mix of two. Off unless the run sets the delay.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, List, Mapping, Optional, Tuple

from .request_limits import HEDGE_AFTER_KEY

logger = logging.getLogger(__name__)

Produce = Callable[[Mapping[str, Any], Any, Optional[float], threading.Event], None]
Message = Tuple[str, Any]


class _Tagged:
    """Queue facade that labels everything one attempt reports with its index."""

    def __init__(self, index: int, merged: 'queue.Queue[Tuple[int, Message]]') -> None:
        self.index, self.merged = index, merged

    def put(self, message: Message) -> None:
        self.merged.put((self.index, message))


def delay(settings: Mapping[str, Any]) -> Optional[float]:
    """Return the run's hedge delay in seconds, or None when hedging is off.

    Args:
        settings: Run settings.

    Returns:
        Seconds to wait before the second request, or None.
    """
    return settings.get(HEDGE_AFTER_KEY)


def _truncated(result: Mapping[str, Any]) -> bool:
    return result['choices'][0].get('finish_reason') == 'length'


def relay(produce: Produce, body: Mapping[str, Any], output: Any, request_timeout: Optional[float],
          cancelled: threading.Event, hedge_after: float) -> None:
    """Run one request, with a second copy if the first is still unanswered after a delay.

    Args:
        produce: Sends one request; reports ``('event', data)`` messages followed by
            ``('complete', result)`` or ``('failed', detail)`` and stops when its event is set.
        body: Request body; passed unchanged to every attempt.
        output: Queue that receives the winning attempt's messages.
        request_timeout: Per-request deadline handed to ``produce``.
        cancelled: Set by the caller when the consumer has gone away.
        hedge_after: Seconds before the second request is sent.

    Raises:
        ValueError: When ``hedge_after`` is not a positive number.
    """
    if isinstance(hedge_after, bool) or not isinstance(hedge_after, (int, float)) or hedge_after <= 0:
        raise ValueError('hedge_after must be a positive number')
    merged: 'queue.Queue[Tuple[int, Message]]' = queue.Queue()
    stops: List[threading.Event] = []
    events: List[List[str]] = []

    def start() -> None:
        stop = threading.Event()
        stops.append(stop)
        events.append([])
        threading.Thread(target=produce, args=(body, _Tagged(len(stops) - 1, merged), request_timeout, stop),
                         daemon=True).start()

    def deliver(index: int, result: Mapping[str, Any]) -> None:
        for data in events[index]:
            output.put(('event', data))
        output.put(('complete', result))

    def stop_all() -> None:
        for stop in stops:
            stop.set()

    start()
    hedge_at = time.monotonic() + hedge_after
    running = 1
    held: Optional[Tuple[int, Mapping[str, Any]]] = None
    failure: Optional[str] = None
    try:
        while True:
            if cancelled.is_set():
                return
            try:
                index, (kind, value) = merged.get(timeout=0.05 if len(stops) < 2 else 0.5)
            except queue.Empty:
                if len(stops) < 2 and time.monotonic() >= hedge_at:
                    logger.info('Request unanswered after %.0f s; asking again', hedge_after)
                    start()
                    running += 1
                continue
            if kind == 'event':
                events[index].append(value)
                continue
            running -= 1
            if kind == 'complete' and not (_truncated(value) and running):
                deliver(index, value)
                return
            if kind == 'complete':
                held = (index, value)
                continue
            failure = failure or value
            if running:
                continue
            if held is not None:
                deliver(*held)
            else:
                output.put(('failed', failure))
            return
    finally:
        stop_all()
