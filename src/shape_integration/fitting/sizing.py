"""Historical 3x/max-edge sizing followed by the min-edge intervention."""
import math


def crop_size(global_extent, *, scale=3., min_long_edge=1080, max_long_edge=2048):
    if (len(global_extent) != 2 or any(not math.isfinite(v) or v <= 0 for v in global_extent)
            or type(scale) not in (int, float) or not math.isfinite(scale) or scale < 1
            or type(min_long_edge) is not int or type(max_long_edge) is not int
            or not 2 <= min_long_edge <= max_long_edge):
        raise ValueError('Invalid crop sizing policy')
    factor = min(scale, max_long_edge / max(global_extent))
    base = [max(2, round(v * factor)) for v in global_extent]
    size = base.copy()
    if max(base) < min_long_edge:
        size = [max(2, round(v * min_long_edge / max(base))) for v in base]
    return tuple(size), {'policy': 'global_crop_scale_then_min_long_edge.1',
                         'requested_crop_scale': scale, 'crop_min_long_edge': min_long_edge,
                         'crop_max_long_edge': max_long_edge, 'base_size_px': base,
                         'crop_size_global_px': list(global_extent),
                         'actual_size_px': size,
                         'effective_scale_xy': [size[i]/global_extent[i] for i in (0, 1)]}
