"""Layer of a texture image that replaces no native part.

Such an image used to be appended as the frontmost object, relying on a list of
named foreground parts being cut out of it. Any icon or image the list missed
was hidden. Objects lying inside the texture are in front of it by
construction, so the image is inserted just below the first of them.
"""
from typing import Any, List, Optional, Sequence, Tuple

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


P_NS = '{http://schemas.openxmlformats.org/presentationml/2006/main}'
A_NS = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
RECTANGULAR = ('rect', 'roundRect')


def above_covering(picture: Box, siblings: Sequence[Sequence[Box]], default: int) -> int:
    """Return the sibling index at which a backdrop picture stays visible.

    A picture is placed to be seen. If an opaque area of a sibling covers it,
    the picture goes directly above the last such sibling; siblings after
    that one stay in front of it.

    Args:
        picture: Placement of the picture in slide coordinates.
        siblings: For each existing child in layer order, its opaque areas.
        default: Index used when no sibling covers the picture.

    Returns:
        The insertion index.
    """
    index = default
    for position, areas in enumerate(siblings):
        if any(_contained(picture, area) for area in areas):
            index = max(index, position + 1)
    return index


def _transform(group: Any, scale: Tuple[float, float], offset: Tuple[float, float]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return the child-to-slide scale and offset of ``group`` inside its own frame."""
    frame = group.find(P_NS + 'grpSpPr/' + A_NS + 'xfrm')
    if frame is None or any(frame.find(A_NS + name) is None for name in ('off', 'ext', 'chOff', 'chExt')):
        return scale, offset
    off, ext, child_off, child_ext = (frame.find(A_NS + name) for name in ('off', 'ext', 'chOff', 'chExt'))
    ratio = tuple(int(ext.get(key)) / max(1, int(child_ext.get(key))) for key in ('cx', 'cy'))
    moved = tuple(offset[i] + scale[i] * (int(off.get(axis)) - int(child_off.get(axis)) * ratio[i])
                  for i, axis in enumerate(('x', 'y')))
    return (scale[0] * ratio[0], scale[1] * ratio[1]), moved


def _opaque(properties: Any) -> bool:
    """Return whether a shape's properties describe a fully opaque solid fill."""
    solid = properties.find(A_NS + 'solidFill')
    colour = None if solid is None else solid.find(A_NS + 'srgbClr')
    if colour is None:
        return False
    alpha = colour.find(A_NS + 'alpha')
    return alpha is None or int(alpha.get('val')) >= 100000


def opaque_boxes(node: Any, scale: Tuple[float, float] = (1., 1.), offset: Tuple[float, float] = (0., 0.)) -> List[Box]:
    """Return the slide-coordinate boxes that ``node`` paints opaquely.

    Only unrotated rectangles and rounded rectangles with a solid, fully
    opaque fill count: for them the bounding box is the painted area. Groups
    are searched through their coordinate frames.
    """
    kind = node.tag.rsplit('}', 1)[-1]
    if kind == 'grpSp':
        inner_scale, inner_offset = _transform(node, scale, offset)
        return [found for child in node for found in opaque_boxes(child, inner_scale, inner_offset)]
    if kind != 'sp':
        return []
    properties = node.find(P_NS + 'spPr')
    frame = None if properties is None else properties.find(A_NS + 'xfrm')
    preset = None if properties is None else properties.find(A_NS + 'prstGeom')
    if frame is None or preset is None or preset.get('prst') not in RECTANGULAR or not _opaque(properties):
        return []
    if any(frame.get(key, '0') not in ('0', 'false') for key in ('rot', 'flipH', 'flipV')):
        return []
    off, ext = frame.find(A_NS + 'off'), frame.find(A_NS + 'ext')
    left, top = offset[0] + scale[0] * int(off.get('x')), offset[1] + scale[1] * int(off.get('y'))
    return [(left, top, left + scale[0] * int(ext.get('cx')), top + scale[1] * int(ext.get('cy')))]
