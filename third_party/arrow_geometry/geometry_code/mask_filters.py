"""Reusable, reversible preprocessing for both global and native crop SAM masks.

Small + isolated components are speck candidates. A removal budget prevents this
heuristic from erasing dotted lines or tiny whole instances. No largest-component
selection, dilation, closing, skeleton pruning or instance decisions occur here.
"""
import cv2
import numpy as np
from scipy.spatial import cKDTree


DEFAULT = dict(max_component_area_native=4, min_isolation_native=8., max_removed_fraction=.01)


def clean_mask(mask, settings=None):
    settings = DEFAULT if settings is None else settings
    mask = np.asarray(mask, bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    candidates = []
    removed = np.zeros_like(mask)
    if not mask.any():
        return mask.copy(), removed, dict(input_pixels=0, output_pixels=0, removed_pixels=0,
            components=0, candidate_components=[], status='empty', settings=settings)
    ys, xs = np.nonzero(mask)
    xy = np.column_stack([xs, ys]); owners = labels[ys, xs]
    tiny_ids = np.flatnonzero(areas <= settings['max_component_area_native']) + 1
    if len(tiny_ids):
        tree = cKDTree(xy)
        for ident in tiny_ids:
            points = xy[owners == ident]
            # k > own component size guarantees another component if one exists.
            k = min(len(xy), len(points) + 1)
            distances, indices = tree.query(points, k=k)
            distances = np.asarray(distances).reshape(len(points), k)
            indices = np.asarray(indices).reshape(len(points), k)
            other = owners[indices] != ident
            nearest = float(distances[other].min()) if other.any() else None
            if nearest is not None and nearest > settings['min_isolation_native']:
                removed[labels == ident] = True
                candidates.append(dict(component=int(ident), area=int(areas[ident-1]),
                    bbox=stats[ident,:4].tolist(), nearest_other_foreground_native=nearest))
    fraction = float(removed.sum()) / int(mask.sum())
    apply = fraction <= settings['max_removed_fraction']
    candidate_pixels = int(removed.sum())
    if not apply:
        removed[:] = False
    cleaned = mask & ~removed
    info = dict(input_pixels=int(mask.sum()), output_pixels=int(cleaned.sum()),
        removed_pixels=int(removed.sum()), candidate_pixels=candidate_pixels,
        components=int(count-1), candidate_components=candidates,
        status='filtered' if removed.any() else 'deferred_high_removal_fraction' if not apply else 'unchanged',
        settings=settings, note='Removed pixels are retained separately; thresholds are preprocessing only, not SAM scores.')
    return cleaned, removed, info
