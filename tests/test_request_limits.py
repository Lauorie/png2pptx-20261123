"""A run can bound how long one bad request holds a branch up.

Observed on one figure with Sage-S2.1-Flash: one shape request reasoned up to
the 100,000-token cap (16 minutes) and two streams went silent and were only
given up after the 5-minute idle wait; together about 28 of the branch's 46 minutes.
"""
import pytest

from full_recovery import request_limits


def test_unconfigured_limits_leave_settings_and_requests_untouched():
    settings = {'api_model': 'm'}
    body = {'max_tokens': 100000, 'messages': []}
    assert request_limits.to_settings(settings, None, None) == settings
    assert request_limits.capped(body, settings) == body


def test_limits_are_recorded_in_the_run_settings_without_touching_the_input():
    settings = {'api_model': 'm'}
    out = request_limits.to_settings(settings, 32000, 60)
    assert out == {'api_model': 'm', 'api_max_tokens': 32000, 'api_idle_timeout_seconds': 60}
    assert settings == {'api_model': 'm'}


def test_the_output_cap_only_ever_lowers_a_request():
    settings = request_limits.to_settings({}, 32000, None)
    body = {'max_tokens': 100000, 'messages': []}
    assert request_limits.capped(body, settings)['max_tokens'] == 32000
    assert body['max_tokens'] == 100000
    assert request_limits.capped({'max_tokens': 256}, settings)['max_tokens'] == 256
    assert request_limits.capped({'messages': []}, settings)['max_tokens'] == 32000


@pytest.mark.parametrize('cap,idle', [(0, None), (-5, None), (True, None), ('32000', None),
                                      (None, 0), (None, -1), (None, 'soon')])
def test_invalid_limits_are_rejected(cap, idle):
    with pytest.raises(ValueError):
        request_limits.to_settings({}, cap, idle)
