"""The complete workflow must not retain a hidden 12-request limiter."""
from copy import deepcopy
import threading

from full_recovery.prepare_api import align_shape_limits


def test_shared_gateway_capacity_reaches_the_shape_limiter():
    settings={'api_global_concurrency':96,'glm_concurrency':12,'case_workers':24,
              'glm_max_tokens':100000,'reasoning_effort':'high'}
    config={'api_concurrency':12,'case_workers':2,'fitting':{'max_revisions':1},
            'styling':{'max_tokens':100000}}
    inference=deepcopy((settings['glm_max_tokens'],settings['reasoning_effort'],
                        config['fitting'],config['styling']))
    align_shape_limits(settings,config)
    slots=threading.BoundedSemaphore(settings['glm_concurrency'])
    assert all(slots.acquire(blocking=False) for _ in range(96))
    assert not slots.acquire(blocking=False)
    assert config['api_concurrency']==96 and config['case_workers']==24
    assert inference==(settings['glm_max_tokens'],settings['reasoning_effort'],
                       config['fitting'],config['styling'])


def test_separate_shape_relay_keeps_its_allocated_share():
    settings={'api_global_concurrency':96,'glm_concurrency':12,'case_workers':24,
              'api_relays':[{'role':'shape','api_global_concurrency':48},
                            {'role':'text','api_global_concurrency':48}]}
    config={}
    align_shape_limits(settings,config)
    assert settings['glm_concurrency']==config['api_concurrency']==48


def test_small_requested_limit_is_not_promoted_to_an_old_default():
    settings={'api_global_concurrency':3,'glm_concurrency':12,'case_workers':2}
    config={}
    align_shape_limits(settings,config)
    assert settings['glm_concurrency']==config['api_concurrency']==3
