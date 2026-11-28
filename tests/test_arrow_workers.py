"""Arrow task pools follow the request limit instead of a fixed width of 8."""
from full_recovery.arrow_workers import widened


def test_task_pools_grow_to_half_the_request_limit():
    config = {'task_workers': 8, 'geometry_critic_workers': 8, 'case_workers': 2}
    assert widened(config, 48) == {'task_workers': 24, 'geometry_critic_workers': 24,
                                   'case_workers': 2}


def test_a_small_request_limit_keeps_the_existing_width():
    config = {'task_workers': 8, 'geometry_critic_workers': 8}
    assert widened(config, 8) == config


def test_a_wider_existing_pool_is_not_narrowed():
    assert widened({'task_workers': 32}, 48)['task_workers'] == 32


def test_configs_without_task_pools_gain_no_keys():
    config = {'local_workers': 32, 'api_concurrency': 48}
    assert widened(config, 48) == config


def test_the_input_config_is_not_mutated():
    config = {'task_workers': 8}
    widened(config, 48)
    assert config == {'task_workers': 8}
