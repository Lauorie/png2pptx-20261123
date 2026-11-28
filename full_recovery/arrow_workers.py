"""Width of the arrow branch's per-image task pools.

The frozen arrow template runs eight tasks per image whatever the request limit
is, so an image with many arrow candidates queues behind its own pool while most
request slots stay idle. The pools follow the request limit instead.
"""
from typing import Any, Dict, Optional, Tuple

TASK_POOL_KEYS: Tuple[str, ...] = ('task_workers', 'geometry_critic_workers')
# Pools counted in images: how many the branch works on, and how many do local computation, at once.
IMAGE_POOL_KEYS: Tuple[str, ...] = ('case_workers', 'cpu_workers')


def widened(config: Dict[str, Any], api_concurrency: int, images: Optional[int] = None) -> Dict[str, Any]:
    """Return a config whose pools follow the request limit and the number of images at once.

    The template works on two images at a time whatever the job is told, so in
    a job of many images the last ones wait most of the run for their turn.

    Args:
        config: One arrow stage configuration.
        api_concurrency: Request limit shared by every arrow stage.
        images: Images the job works on at once; ``None`` keeps the template's.

    Returns:
        A copy of ``config``; pools it does not define are not added and a pool
        that is already wider is kept.
    """
    targets = {key: int(api_concurrency) // 2 for key in TASK_POOL_KEYS}
    if images is not None:
        targets.update({key: int(images) for key in IMAGE_POOL_KEYS})
    result = dict(config)
    for key, target in targets.items():
        if key in result:
            result[key] = max(int(result[key]), target)
    return result
