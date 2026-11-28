"""Keep the service's GPU memory bounded and report what inference really uses.

PyTorch keeps the memory of the largest request it has served as a cache. One
figure with many masks therefore leaves the process holding most of the card
for good, although the model and an ordinary request need a fraction of it.
After each request the cache is returned once it exceeds a limit, and the
memory in use, the request's peak and the amount returned are reported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

MIB = 1024 * 1024
DEFAULT_CACHE_LIMIT_MIB = 2048


@dataclass(frozen=True)
class MemoryPolicy:
    """How much idle cache to keep, and an optional ceiling for the whole process."""

    cache_limit_mib: int = DEFAULT_CACHE_LIMIT_MIB
    hard_limit_mib: Optional[int] = None


def policy_from(config: Mapping[str, Any]) -> MemoryPolicy:
    """Read the policy from the service config.

    Args:
        config: May hold ``gpu_cache_limit_mib`` and ``gpu_memory_limit_mib``.

    Returns:
        The policy; without the keys, a 2 GiB cache and no ceiling.

    Raises:
        ValueError: When a limit is not a non-negative, respectively positive, number.
    """
    cache = config.get('gpu_cache_limit_mib', DEFAULT_CACHE_LIMIT_MIB)
    hard = config.get('gpu_memory_limit_mib')
    if isinstance(cache, bool) or not isinstance(cache, (int, float)) or cache < 0:
        raise ValueError('gpu_cache_limit_mib must be a non-negative number')
    if hard is not None and (isinstance(hard, bool) or not isinstance(hard, (int, float)) or hard <= 0):
        raise ValueError('gpu_memory_limit_mib must be a positive number')
    return MemoryPolicy(int(cache), None if hard is None else int(hard))


def apply_limit(torch: Any, policy: MemoryPolicy) -> Optional[int]:
    """Give the allocator the process ceiling, so exceeding it fails clearly instead of filling the card.

    Returns:
        The ceiling in MiB, or None when the policy sets none.
    """
    if policy.hard_limit_mib is None:
        return None
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, policy.hard_limit_mib * MIB / total))
    return policy.hard_limit_mib


def settle(torch: Any, policy: MemoryPolicy) -> Dict[str, int]:
    """Return an oversized cache to the GPU after a request and report the request's memory.

    Returns:
        Memory in use, memory held by the process, the request's peak and the cache returned, in MiB.
    """
    cuda = torch.cuda
    in_use = cuda.memory_allocated()
    held = cuda.memory_reserved()
    peak = cuda.max_memory_allocated()
    released = 0
    if held - in_use > policy.cache_limit_mib * MIB:
        cuda.empty_cache()
        released = held - cuda.memory_reserved()
        held = cuda.memory_reserved()
    cuda.reset_peak_memory_stats()
    return {'in_use_mib': in_use // MIB, 'reserved_mib': held // MIB, 'request_peak_mib': peak // MIB,
            'cache_released_mib': released // MIB}
