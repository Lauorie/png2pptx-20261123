"""Clamp a crop's original-pixel long edge; never crop away source content."""
import math

def long_edge(extent, bounds=None, fixed=1080):
    edge = max(map(float, extent))
    if not math.isfinite(edge) or edge <= 0:
        raise ValueError('Invalid source crop extent')
    if bounds is None:
        return int(fixed)
    if len(bounds) != 2 or not 32 <= bounds[0] <= bounds[1]:
        raise ValueError('Invalid display long-edge interval')
    return round(max(bounds[0], min(bounds[1], edge)))
