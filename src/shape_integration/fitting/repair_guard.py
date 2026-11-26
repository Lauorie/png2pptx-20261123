"""Keep a coverage repair from throwing away geometry it was only asked to extend.

Coverage asks an owner to add a source part it lacks. The repair observes the
object afresh, and the model sometimes answers with the requested piece alone;
the critic, pointed at that piece, then accepts it and the rest of the object
is gone. A repair whose requests are all ``missing`` must still cover the area
the previous fit covered, otherwise the previous fit is kept.
"""
from typing import Any, Dict, List, Optional, Sequence

Box = List[float]
MINIMUM_RETAINED = 0.8


def footprint(fit: Dict[str, Any]) -> Optional[Box]:
    """Return the box around every successfully fitted part, or ``None``."""
    boxes = [part['bbox_2d'] for part in fit.get('parts', [])
             if part.get('status') == 'ok' and part.get('bbox_2d')]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def retained(base: Box, repair: Optional[Box]) -> float:
    """Return the fraction of ``base`` that lies inside ``repair``."""
    area = max(0., base[2] - base[0]) * max(0., base[3] - base[1])
    if repair is None or area <= 0:
        return 0. if repair is None else 1.
    width = min(base[2], repair[2]) - max(base[0], repair[0])
    height = min(base[3], repair[3]) - max(base[1], repair[1])
    return max(0., width) * max(0., height) / area


def discards_base(requests: Sequence[Dict[str, Any]], base_fit: Dict[str, Any],
                  repair_fit: Dict[str, Any], minimum: float = MINIMUM_RETAINED) -> bool:
    """Tell whether an add-only repair lost most of the previous geometry.

    Args:
        requests: Coverage requests of the repair, each with a ``kind``.
        base_fit: ``fit_original.json`` content of the previous fit.
        repair_fit: ``fit_original.json`` content the repair selected.
        minimum: Smallest fraction of the previous footprint to keep.

    Returns:
        ``True`` when every request was ``missing`` and the repair covers less
        than ``minimum`` of the area the previous fit covered.
    """
    if not requests or any(request.get('kind') != 'missing' for request in requests):
        return False
    base = footprint(base_fit)
    if base is None:
        return False
    return retained(base, footprint(repair_fit)) < minimum
