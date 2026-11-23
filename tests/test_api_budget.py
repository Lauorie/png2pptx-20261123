import json
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
from unittest.mock import patch

import pytest

from full_recovery.api_budget import BudgetGuard, BudgetPaused, BudgetBusy, balance_snapshot
from full_recovery.api_gateway import ApiRelay, payload_for_api
from full_recovery.inference_profile import apply


def setup_budget(tmp_path, limit=10):
    monitor = tmp_path/'balance.txt'
    monitor.write_text(f'[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S %z}] '
                       'OK Key=1 剩余额度=100 累计消费=0\n')
    path = tmp_path/'config.json'
    path.write_text(json.dumps({'limit_cny': limit, 'balance_unit_cny': 1,
        'balance_file': str(monitor), 'baseline': {'key_id': '1', 'total_spent': '0', 'remaining': '100'},
        'prices': {'gpt-test': {'input_cny_per_million': 1000,
                  'output_cny_per_million': 1000, 'max_input_tokens': 1000}}}))
    return path


def test_atomic_reservations_do_not_oversubscribe_shared_budget(tmp_path):
    config = setup_budget(tmp_path)
    def reserve(_):
        guard = BudgetGuard(config)
        try:
            return guard.reserve({'model': 'gpt-test', 'max_tokens': 1000}, 'worker')
        except BudgetBusy:
            return None
    with ThreadPoolExecutor(max_workers=24) as pool:
        ids = list(pool.map(reserve, range(96)))
    assert len([x for x in ids if x]) == 5
    guard = BudgetGuard(config)
    assert guard.check()['pending_reservations_cny'] == 10
    ident = next(x for x in ids if x)
    guard.settle(ident, {'prompt_tokens': 100, 'completion_tokens': 200})
    assert guard.check()['task_charge_upper_bound_cny'] == .3
    # A separate process/restart sees both settled and in-flight liabilities.
    assert BudgetGuard(config).check()['pending_reservations_cny'] == 8


def test_unknown_charge_is_retained_and_new_requests_pause(tmp_path):
    guard = BudgetGuard(setup_budget(tmp_path, 2))
    ident = guard.reserve({'model': 'gpt-test', 'max_tokens': 1000}, 'test')
    guard.settle(ident)
    with pytest.raises(BudgetPaused):
        guard.reserve({'model': 'gpt-test', 'max_tokens': 1000}, 'test')
    assert guard.pause_path.exists()


def test_stale_monitor_fails_closed(tmp_path):
    config = setup_budget(tmp_path)
    (tmp_path/'balance.txt').write_text('[2020-01-01 00:00:00 +0000] OK Key=1 剩余额度=100 累计消费=0\n')
    with pytest.raises(BudgetPaused, match='stale'):
        BudgetGuard(config).check()


@pytest.mark.parametrize('remaining', ['0', '-2.1904'])
def test_exhausted_quota_is_read_and_paused_before_task_limit(tmp_path, remaining):
    config = setup_budget(tmp_path, 500)
    monitor = tmp_path/'balance.txt'
    monitor.write_text(monitor.read_text() +
        f'[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S %z}] '
        f'OK Key=1 剩余额度={remaining} 累计消费=102.1904\n')
    assert balance_snapshot(monitor)['remaining'] == remaining
    with pytest.raises(BudgetPaused, match='quota exhausted'):
        BudgetGuard(config).check()


def test_payment_required_stops_other_requests_without_waiting_for_balance(tmp_path):
    import io
    import urllib.error
    config = setup_budget(tmp_path, 500)
    key = tmp_path/'key';key.write_text('test-key');key.chmod(0o600)
    settings = {'api_key_file':str(key),'api_global_concurrency':1,
                'api_model':'gpt-test','api_base_url':'https://example.invalid/v1',
                'api_budget_config':str(config)}
    with patch('full_recovery.api_gateway.urllib.request.build_opener') as opener:
        opener.return_value.open.side_effect = urllib.error.HTTPError(
            'https://example.invalid/v1/chat/completions',402,'Payment Required',{},io.BytesIO(b'quota exhausted'))
        first = ApiRelay(settings,tmp_path)
        first.produce({'max_tokens':1000},queue.Queue())
        assert (tmp_path/'PAUSED.json').exists()
        second = ApiRelay(settings,tmp_path)
        second.produce({'max_tokens':1000},queue.Queue())
        assert opener.return_value.open.call_count == 1


def test_shared_account_spending_also_stops_our_requests(tmp_path):
    guard = BudgetGuard(setup_budget(tmp_path))
    monitor = tmp_path/'balance.txt'
    monitor.write_text(monitor.read_text().replace('剩余额度=100 累计消费=0', '剩余额度=89 累计消费=11'))
    with pytest.raises(BudgetPaused):
        guard.check()


def test_paused_budget_never_reaches_upstream(tmp_path):
    config = setup_budget(tmp_path)
    (tmp_path/'PAUSED.json').write_text('{}')
    key = tmp_path/'key'; key.write_text('test-key'); key.chmod(0o600)
    relay = ApiRelay({'api_key_file': str(key), 'api_global_concurrency': 1,
                     'api_model': 'gpt-test', 'api_base_url': 'https://example.invalid/v1',
                     'api_budget_config': str(config)}, tmp_path)
    with patch('full_recovery.api_gateway.urllib.request.build_opener') as opener:
        output = queue.Queue()
        relay.produce({'max_tokens': 1000}, output)
        opener.return_value.open.assert_not_called()
    assert relay.stats['active'] == 0
    assert list(output.queue) == [('failed', 'Task budget is paused')]


def test_high_overrides_all_local_low_and_glm_controls():
    out = payload_for_api({'reasoning_effort': 'low', 'temperature': 0,
                           'chat_template_kwargs': {'reasoning_effort': 'low', 'enable_thinking': False}},
                          'gpt-5.6-terra', 'high')
    assert out['reasoning_effort'] == 'high'
    assert not {'thinking', 'temperature', 'chat_template_kwargs'} & out.keys()


def test_explicit_account_mode_records_usage_and_stops_at_reported_limit(tmp_path):
    config = setup_budget(tmp_path, 500)
    data = json.loads(config.read_text())
    data.update(mode='account_balance',prices={},warning_cny=400)
    config.write_text(json.dumps(data))
    guard = BudgetGuard(config)
    ident = guard.reserve({'model':'gpt-5.6-terra','max_tokens':100000},'test')
    guard.settle(ident,{'prompt_tokens':12000,'completion_tokens':8000})
    status = guard.check()
    assert status['task_charge_upper_bound_cny'] is None
    assert status['requests'] == {'usage_recorded':1}
    assert not status['unreported_provider_charges_bounded']
    monitor = tmp_path/'balance.txt'
    monitor.write_text(monitor.read_text().replace('累计消费=0','累计消费=500'))
    with pytest.raises(BudgetPaused):
        guard.reserve({'model':'gpt-5.6-terra','max_tokens':100000},'test')
    assert guard.pause_path.exists()


def test_lean_keeps_source_guards_and_is_reversible():
    config = {'fitting': {'geometry_guard_enabled': True, 'coverage_rounds': 1},
              'model_conflict_review': True}
    settings, lean, text = apply({}, config, {}, model='gpt-5.6-terra', effort='high', profile='lean')
    assert not lean['fitting']['check_enabled']
    assert lean['fitting']['geometry_guard_enabled']
    assert lean['fitting']['coverage_rounds'] == 1
    assert lean['model_conflict_review']
    assert lean['styling']['critic_enabled']
    assert not lean['styling']['conditional_pattern_critic']
    assert 'check_enabled' not in config['fitting']
    _, full, _ = apply(settings, lean, text, model='gpt-5.6-terra', effort='high', profile='full')
    assert full['fitting']['check_enabled'] and full['styling']['conditional_pattern_critic']
