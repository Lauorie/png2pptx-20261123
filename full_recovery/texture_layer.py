"""Layer of a texture image that replaces no native part.

Such an image used to be appended as the frontmost object, relying on a list of
named foreground parts being cut out of it. Any icon or image the list missed
was hidden. Objects lying inside the texture are in front of it by
construction, so the image is inserted just below the first of them.
"""
from typing import Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]

CONTAINED_FRACTION = 0.9


def _area(bounds: Box) -> float:
    return max(0.0, bounds[2] - bounds[0]) * max(0.0, bounds[3] - bounds[1])


def _contained(inner: Box, outer: Box) -> bool:
    area = _area(inner)
    if area <= 0:
        return False
    overlap = (max(inner[0], outer[0]), max(inner[1], outer[1]), min(inner[2], outer[2]), min(inner[3], outer[3]))
    return _area(overlap) >= CONTAINED_FRACTION * area


def behind_contained(texture: Box, siblings: Sequence[Optional[Box]], default: int) -> int:
    """Return the sibling index at which to insert the texture image.

    Args:
        texture: Placement of the image in the siblings' coordinate frame.
        siblings: Bounds of each existing child in layer order; ``None`` for
            children without bounds.
        default: Index used when the texture contains no sibling.

    Returns:
        The index of the first sibling lying inside the texture, else ``default``.
    """
    for index, bounds in enumerate(siblings):
        if bounds is not None and _contained(bounds, texture):
            return index
    return default
