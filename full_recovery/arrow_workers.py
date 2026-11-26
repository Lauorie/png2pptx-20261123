"""Width of the arrow branch's per-image task pools.

The frozen arrow template runs eight tasks per image whatever the request limit
is, so an image with many arrow candidates queues behind its own pool while most
request slots stay idle. The pools follow the request limit instead.
"""
from typing import Any, Dict, Tuple

TASK_POOL_KEYS: Tuple[str, ...] = ('task_workers', 'geometry_critic_workers')


def widened(config: Dict[str, Any], api_concurrency: int) -> Dict[str, Any]:
    """Return a config whose task pools use at least half the request limit.

    Args:
        config: One arrow stage configuration.
        api_concurrency: Request limit shared by every arrow stage.

    Returns:
        A copy of ``config``; pools it does not define are not added and a pool
        that is already wider is kept.
    """
    target = int(api_concurrency) // 2
    result = dict(config)
    for key in TASK_POOL_KEYS:
        if key in result:
            result[key] = max(int(result[key]), target)
    return result
