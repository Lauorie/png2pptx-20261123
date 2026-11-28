"""Drop an outline that could not physically fit the part it outlines.

When a part's outline colour is close to its fill colour, the width
measurement can return the width of the part itself. Drawn in PowerPoint, an
8 px outline on an 8 px bar triples the bar. Such an outline is the fill seen
twice, so it is removed and the fill kept. A wide outline in a clearly
different colour is left alone: a light inset framed in black is real.
"""
from typing import Any, Dict, Optional, Sequence, Tuple

MAXIMUM_SHARE = 0.5
SAME_PIGMENT_DISTANCE = 80.


def _distance(first: str, second: str) -> float:
    """Return the RGB distance between two ``#RRGGBB`` colours."""
    a, b = (tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in (first, second))
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** .5


def fit(style: Dict[str, Any], bounds: Sequence[float],
        maximum_share: float = MAXIMUM_SHARE) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Return the style without an outline wider than the part allows.

    Args:
        style: Part style with ``fill``, ``stroke`` and ``width_px``.
        bounds: ``(min_x, min_y, max_x, max_y)`` of the part in source pixels.
        maximum_share: Largest outline width as a share of the smaller side.

    Returns:
        The unchanged style and ``None``, or a copy without the outline and a
        record of what was dropped. Only an outline close to the fill colour
        is dropped.
    """
    if style.get('stroke') is None or style.get('fill') is None:
        return style, None
    side = min(bounds[2] - bounds[0], bounds[3] - bounds[1])
    width = style.get('width_px') or 0.
    if side <= 0 or width <= maximum_share * side:
        return style, None
    if _distance(style['fill'], style['stroke']) > SAME_PIGMENT_DISTANCE:
        return style, None
    return {**style, 'stroke': None}, {
        'before_width_px': width, 'before_stroke': style['stroke'], 'smaller_side_px': side,
        'policy': 'fill_coloured_outline_wider_than_half_the_part.2'}
