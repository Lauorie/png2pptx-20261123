"""Native parts a reviewer may name as lying in front of an image asset.

Only parts offered here can be cut out of a placed image. Simple filled shapes
were always offered; a filled closed outline, such as the page of a document
icon, was not, so its inner marks could be kept in front while the page itself
stayed hidden behind the image.
"""
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from shapely.geometry import box

from .texture_fallback import polygon

SIMPLE_KINDS = ('ellipse', 'rectangle', 'rounded_rectangle')


def _closed_outline(part: Mapping[str, Any]) -> bool:
    return part.get('kind') == 'path' and polygon(part) is not None


def overlapping(parts: Mapping[str, Mapping[str, Any]], styles: Mapping[str, Mapping[str, Any]],
                own_keys: Iterable[str], target_box: Sequence[float]) -> List[Dict[str, Any]]:
    """List the filled foreign parts that intersect the target without covering it.

    Args:
        parts: Fitted scene parts by key.
        styles: Recovered styles by part key.
        own_keys: Parts of the asset's own owner, never offered.
        target_box: Source box of the image asset.

    Returns:
        One ``{'key', 'bbox'}`` row per candidate, in scene order.
    """
    own = set(own_keys)
    target = box(*target_box)
    rows: List[Dict[str, Any]] = []
    for key, part in parts.items():
        if key in own or not isinstance(styles.get(key, {}).get('fill'), str):
            continue
        if part.get('kind') not in SIMPLE_KINDS and not _closed_outline(part):
            continue
        extent = box(*part['bbox_2d'])
        if target.intersection(extent).area > 4 and not extent.covers(target):
            rows.append({'key': key, 'bbox': part['bbox_2d']})
    return rows
