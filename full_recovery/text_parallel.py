"""Send the text stage's independent model requests at the same time.

The reference visual audit asks one region, applies the reply, then builds the
next region's request. No request needs an earlier answer to be sent, so the
audit is run twice around one parallel round of requests:

1. a planning pass in which every region builds its request against the
   inventory as it stands and no request is sent;
2. all planned requests are sent together;
3. an applying pass in which every region receives its stored reply, in the
   reference order, through the reference validation and merge code.

A reply is therefore written against the inventory from before the audit. The
reference merge already folds a re-reported occurrence into the existing one,
and a retirement that an earlier region has already applied is dropped.

The crop review after it already runs in a thread pool; its fixed width of four
and the matching per-image request limit are raised together.
"""
from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger(__name__)

PROTOCOL = 'parallel_text_requests.1'
DEFAULT_AUDIT_WORKERS = 4
DEFAULT_CROP_WORKERS = 12

Ask = Callable[..., Dict[str, Any]]


class _Planned(Exception):
    """Raised in place of a model call while requests are only being collected."""


def _require_positive(workers: int, name: str) -> None:
    if type(workers) is not int or workers < 1:
        raise ValueError(f'{name} must be a positive integer')


def _without_stale_retirements(reply: Dict[str, Any], prompt: Dict[str, Any]) -> Dict[str, Any]:
    """Drop retirements whose occurrences an earlier region already removed.

    Args:
        reply: Validated reply of one region.
        prompt: The region's request as rebuilt against the current inventory.

    Returns:
        A copy of the reply that only retires occurrences still present.
    """
    result = deepcopy(reply)
    if not isinstance(result, dict) or not isinstance(result.get('duplicate_ids'), list):
        return result
    present = {row.get('id') for row in prompt.get('existing_occurrences', [])}
    kept = [item for item in result['duplicate_ids']
            if item.get('id') in present and item.get('duplicate_of') in present]
    if len(kept) != len(result['duplicate_ids']):
        logger.warning('Dropped %d retirement(s) an earlier audit region already applied',
                       len(result['duplicate_ids']) - len(kept))
    result['duplicate_ids'] = kept
    return result


def audit_in_parallel(audit: Callable[[Ask], Any], ask: Ask, workers: int) -> Any:
    """Run a region-by-region audit with all of its requests in flight together.

    Args:
        audit: Runs the sequential audit, using the function it is given for
            every model request ``(runtime, folder, prompt, images, validator, opt)``.
        ask: Sends one request and returns its validated reply.
        workers: Largest number of requests in flight.

    Returns:
        Whatever ``audit`` returns once every stored reply has been applied.

    Raises:
        ValueError: When ``workers`` is not a positive integer.
    """
    _require_positive(workers, 'workers')
    planned: List[Tuple[Any, ...]] = []
    sent_images: Dict[Path, bytes] = {}

    def plan(runtime: Any, folder: Path, prompt: Dict[str, Any], images: List[Tuple[str, Path]],
             validator: Any, opt: Dict[str, Any]) -> Dict[str, Any]:
        planned.append((runtime, folder, deepcopy(prompt), list(images), validator, opt))
        raise _Planned(str(folder))

    drafted = audit(plan)
    if not planned:
        return drafted
    with ThreadPoolExecutor(max_workers=min(workers, len(planned))) as pool:
        replies: Dict[str, Future] = {str(call[1]): pool.submit(ask, *call) for call in planned}
    # The applying pass redraws each region's hint image from the updated
    # inventory; keep the files that were actually sent as the evidence.
    for call in planned:
        for _, path in call[3]:
            sent_images[Path(path)] = Path(path).read_bytes()

    def replay(runtime: Any, folder: Path, prompt: Dict[str, Any], images: List[Tuple[str, Path]],
               validator: Any, opt: Dict[str, Any]) -> Dict[str, Any]:
        return _without_stale_retirements(replies[str(folder)].result(), prompt)

    try:
        return audit(replay)
    finally:
        for path, content in sent_images.items():
            if path.read_bytes() != content:
                path.write_bytes(content)


def crop_executor(workers: int) -> Callable[..., ThreadPoolExecutor]:
    """Build a thread-pool factory that ignores the caller's fixed width.

    Args:
        workers: Pool width to use instead.

    Returns:
        A drop-in replacement for ``ThreadPoolExecutor``.

    Raises:
        ValueError: When ``workers`` is not a positive integer.
    """
    _require_positive(workers, 'workers')

    def executor(max_workers: int | None = None, **kwargs: Any) -> ThreadPoolExecutor:
        return ThreadPoolExecutor(max_workers=workers, **kwargs)

    return executor


def with_request_limit(config: Dict[str, Any], workers: int = DEFAULT_CROP_WORKERS) -> Dict[str, Any]:
    """Let one image's text stage keep as many requests in flight as its crop pool is wide.

    The reference limit of four requests per image would otherwise queue the
    wider crop pool behind the same four slots.

    Args:
        config: Text backend configuration.
        workers: Crop-review pool width.

    Returns:
        A copy of the configuration whose request limit is at least ``workers``.

    Raises:
        ValueError: When ``workers`` is not a positive integer.
    """
    _require_positive(workers, 'workers')
    return {**config, 'api_concurrency': max(config['api_concurrency'], workers)}


def install(audit_workers: int = DEFAULT_AUDIT_WORKERS, crop_workers: int = DEFAULT_CROP_WORKERS) -> None:
    """Make the loaded text backend send its audit and crop requests in parallel.

    Args:
        audit_workers: Visual-audit regions asked at the same time.
        crop_workers: Crop-review groups processed at the same time.

    Raises:
        ValueError: When a worker count is not a positive integer.
    """
    from full_recovery import text_diagnostic_repair
    from shape_integration.texts import inventory_audit, two_stage_crop
    _require_positive(audit_workers, 'audit_workers')
    if getattr(inventory_audit, '_parallel_requests_protocol', None) == PROTOCOL:
        return
    sequential = inventory_audit.audit
    send = text_diagnostic_repair.ask_inventory
    current = threading.local()

    def ask_inventory(*args: Any) -> Dict[str, Any]:
        return getattr(current, 'ask', send)(*args)

    def audit(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        def run(ask: Ask) -> Dict[str, Any]:
            current.ask = ask
            try:
                return sequential(*args, **kwargs)
            finally:
                del current.ask
        return audit_in_parallel(run, send, audit_workers)

    text_diagnostic_repair.ask_inventory = ask_inventory
    inventory_audit.audit = audit
    two_stage_crop.ThreadPoolExecutor = crop_executor(crop_workers)
    inventory_audit._parallel_requests_protocol = PROTOCOL
