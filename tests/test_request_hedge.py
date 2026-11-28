"""A request that falls far behind is asked a second time and the first answer wins.

Measured on one figure with Sage-S2.1-Flash at low effort: each arrow stage
runs dozens of units side by side and then waits for one straggler (387 s,
519 s and 148 s where the other units took under a minute), 29 of the run's
32 minutes.
"""
import queue
import threading

import pytest

from full_recovery import request_hedge, request_limits


def result(finish='stop', text='answer'):
    return {'choices': [{'index': 0, 'finish_reason': finish, 'message': {'content': text}}]}


def drain(output):
    items = []
    while True:
        try:
            items.append(output.get_nowait())
        except queue.Empty:
            return items


class Upstream:
    """Scripted stand-in for the relay's ``produce``: one behaviour per attempt."""

    def __init__(self, *behaviours):
        self.behaviours = list(behaviours)
        self.started = []
        self.stopped = []
        self.lock = threading.Lock()

    def __call__(self, body, output, request_timeout, cancelled):
        with self.lock:
            index = len(self.started)
            self.started.append(body)
        self.behaviours[index](output, cancelled)
        if cancelled.is_set():
            with self.lock:
                self.stopped.append(index)


def answers(text, finish='stop'):
    def behave(output, cancelled):
        output.put(('event', text))
        output.put(('event', '[DONE]'))
        output.put(('complete', result(finish, text)))
    return behave


def hangs(then=None):
    def behave(output, cancelled):
        output.put(('event', 'partial'))
        cancelled.wait(10)
        if then is not None and not cancelled.is_set():
            then(output, cancelled)
    return behave


def fails(output, cancelled):
    output.put(('failed', 'API HTTP 502'))


def run(upstream, hedge_after=0.05, cancelled=None):
    output = queue.Queue()
    request_hedge.relay(upstream, {'messages': []}, output, None,
                        cancelled or threading.Event(), hedge_after)
    return drain(output)


def test_a_request_that_answers_in_time_is_asked_once():
    upstream = Upstream(answers('fast'))
    assert run(upstream, hedge_after=5) == [
        ('event', 'fast'), ('event', '[DONE]'), ('complete', result(text='fast'))]
    assert len(upstream.started) == 1


def test_a_straggler_is_asked_again_and_the_second_answer_is_delivered_alone():
    upstream = Upstream(hangs(), answers('second'))
    assert run(upstream) == [
        ('event', 'second'), ('event', '[DONE]'), ('complete', result(text='second'))]
    assert len(upstream.started) == 2
    assert upstream.started[0] == upstream.started[1]


def test_the_losing_request_is_cancelled():
    upstream = Upstream(hangs(), answers('second'))
    run(upstream)
    for _ in range(100):
        if upstream.stopped:
            break
        threading.Event().wait(0.01)
    assert upstream.stopped == [0]


def test_a_truncated_answer_waits_for_the_other_request():
    started_second = threading.Event()

    def first(output, cancelled):
        started_second.wait(5)
        answers('ran to the cap', 'length')(output, cancelled)

    def second(output, cancelled):
        started_second.set()
        threading.Event().wait(0.1)
        answers('complete')(output, cancelled)

    items = run(Upstream(first, second))
    assert items[-1] == ('complete', result(text='complete'))
    assert ('event', 'ran to the cap') not in items


def test_a_truncated_answer_is_used_when_the_other_request_fails():
    started_second = threading.Event()

    def first(output, cancelled):
        started_second.wait(5)
        answers('ran to the cap', 'length')(output, cancelled)

    def second(output, cancelled):
        started_second.set()
        threading.Event().wait(0.1)
        fails(output, cancelled)

    items = run(Upstream(first, second))
    assert items[-1] == ('complete', result('length', 'ran to the cap'))


def test_a_failure_after_the_second_request_started_waits_for_it():
    started_second = threading.Event()

    def first(output, cancelled):
        started_second.wait(5)
        fails(output, cancelled)

    def second(output, cancelled):
        started_second.set()
        threading.Event().wait(0.1)
        answers('second')(output, cancelled)

    assert run(Upstream(first, second))[-1] == ('complete', result(text='second'))


def test_a_request_that_fails_before_the_delay_fails_as_before():
    upstream = Upstream(fails)
    assert run(upstream, hedge_after=5) == [('failed', 'API HTTP 502')]
    assert len(upstream.started) == 1


def test_both_requests_failing_reports_one_failure():
    started_second = threading.Event()

    def first(output, cancelled):
        started_second.wait(5)
        fails(output, cancelled)

    def second(output, cancelled):
        started_second.set()
        fails(output, cancelled)

    assert run(Upstream(first, second)) == [('failed', 'API HTTP 502')]


def test_a_consumer_that_leaves_stops_every_request():
    gone = threading.Event()
    upstream = Upstream(hangs(), hangs())
    worker = threading.Thread(target=run, args=(upstream, 0.05, gone))
    worker.start()
    for _ in range(200):
        if len(upstream.started) == 2:
            break
        threading.Event().wait(0.01)
    gone.set()
    worker.join(5)
    assert not worker.is_alive()
    for _ in range(200):
        if len(upstream.stopped) == 2:
            break
        threading.Event().wait(0.01)
    assert sorted(upstream.stopped) == [0, 1]


def test_the_delay_must_be_positive():
    with pytest.raises(ValueError):
        request_hedge.relay(Upstream(), {}, queue.Queue(), None, threading.Event(), 0)


def test_the_delay_is_recorded_in_the_run_settings():
    out = request_limits.to_settings({'api_model': 'm'}, None, None, hedge_after_seconds=90)
    assert out == {'api_model': 'm', 'api_hedge_after_seconds': 90}
    with pytest.raises(ValueError):
        request_limits.to_settings({}, None, None, hedge_after_seconds=-1)


def test_the_gateway_only_hedges_when_the_run_asks_for_it():
    assert request_hedge.delay({'api_model': 'm'}) is None
    assert request_hedge.delay({'api_hedge_after_seconds': 90}) == 90
