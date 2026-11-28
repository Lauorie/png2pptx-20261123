"""The arrow branch can run at its own reasoning effort and output cap.

Arrow geometry asks one small question per candidate region.  Under the
run-wide forced "high" effort a single such request was observed to spend
57,824 reasoning tokens (19 minutes) to answer "this is a box border".
"""
import pytest

from full_recovery import arrow_budget
from full_recovery.api_gateway import payload_for_api


def body(**template):
    return {'max_tokens': 113000, 'messages': [], 'chat_template_kwargs': template}


def test_gateway_keeps_a_pinned_stage_effort_under_a_forced_run_effort():
    out = payload_for_api(body(reasoning_effort='low', pin_reasoning_effort=True), 'm', 'high')
    assert out['reasoning_effort'] == 'low'
    assert out['thinking'] == {'type': 'enabled'}
    assert 'chat_template_kwargs' not in out and 'pin_reasoning_effort' not in out


def test_gateway_still_forces_the_run_effort_on_unpinned_requests():
    assert payload_for_api(body(reasoning_effort='low'), 'm', 'high')['reasoning_effort'] == 'high'


def test_gateway_rejects_an_invalid_pinned_effort():
    with pytest.raises(ValueError):
        payload_for_api(body(reasoning_effort='extreme', pin_reasoning_effort=True), 'm', 'high')


def test_unconfigured_budget_leaves_the_payload_untouched():
    payload = body(clear_thinking=True)
    assert arrow_budget.bounded(payload, arrow_budget.ArrowBudget()) == payload


def test_budget_pins_effort_and_only_ever_lowers_the_output_cap():
    budget = arrow_budget.ArrowBudget(reasoning_effort='low', max_tokens=16000)
    out = arrow_budget.bounded(body(clear_thinking=True), budget)
    assert out['chat_template_kwargs'] == {'clear_thinking': True, 'reasoning_effort': 'low',
                                           'pin_reasoning_effort': True}
    assert out['max_tokens'] == 16000
    small = {**body(), 'max_tokens': 8192}
    assert arrow_budget.bounded(small, budget)['max_tokens'] == 8192
    assert small['chat_template_kwargs'] == {}   # input is not mutated


def test_budget_round_trips_through_settings():
    settings = arrow_budget.to_settings({}, 'low', 16000)
    assert settings == {'arrow_reasoning_effort': 'low', 'arrow_max_tokens': 16000}
    assert arrow_budget.from_settings(settings) == arrow_budget.ArrowBudget('low', 16000)
    assert arrow_budget.to_settings({'a': 1}, None, None) == {'a': 1}
    assert arrow_budget.from_settings({}) == arrow_budget.ArrowBudget()


def test_arrow_branch_gets_its_own_gateway_address():
    assert arrow_budget.arrow_base_url('http://127.0.0.1:18188/v1') == 'http://127.0.0.1:18188/arrow/v1'
    assert arrow_budget.arrow_base_url('http://127.0.0.1:18189/v1/') == 'http://127.0.0.1:18189/arrow/v1'
    with pytest.raises(ValueError):
        arrow_budget.arrow_base_url('http://127.0.0.1:18188')


def test_gateway_bounds_only_requests_that_arrive_on_the_arrow_route():
    settings = arrow_budget.to_settings({}, 'low', 16000)
    request = {'max_tokens': 113000, 'thinking': {'type': 'enabled'}, 'messages': []}
    plain = arrow_budget.routed_body('/v1/chat/completions', request, settings)
    assert plain == request
    assert payload_for_api(plain, 'm', 'high')['reasoning_effort'] == 'high'
    arrow = arrow_budget.routed_body('/arrow/v1/chat/completions', request, settings)
    wire = payload_for_api(arrow, 'm', 'high')
    assert (wire['reasoning_effort'], wire['max_tokens'], wire['thinking']) == ('low', 16000, {'type': 'enabled'})
    assert arrow_budget.routed_body('/arrow/v1/chat/completions', request, {}) == request
    assert arrow_budget.routed_body('/v2/chat/completions', request, settings) is None
    assert 'chat_template_kwargs' not in request


@pytest.mark.parametrize('effort,cap', [('extreme', None), (None, 0), (None, -5)])
def test_invalid_budgets_are_rejected(effort, cap):
    with pytest.raises(ValueError):
        arrow_budget.to_settings({}, effort, cap)
