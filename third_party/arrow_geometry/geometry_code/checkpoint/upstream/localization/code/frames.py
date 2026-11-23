"""All model boxes are GLOBAL pixel centers; conversions are backend-only."""
import math
from pathlib import Path
from PIL import Image
from pipeline_frames import Crop, resized_size


def global_frame(native_size):
    return Crop(tuple(native_size), (0, 0, *native_size), resized_size(native_size, 2048))


def native_box(box, view):
    exact = view.to_global(box[:2]) + view.to_global(box[2:])
    visible = [max(0, min(view.global_size[k % 2]-1, v)) for k, v in enumerate(exact)]
    return {"raw_bbox_global_px": list(box), "mapped_native_unclipped": exact,
            "bbox_native": visible, "boundary_intersection_applied": exact != visible,
            "convention": "pixel centers; explicit intersection with native center bounds; raw/exact values retained"}


def crop_from_global(box, view, min_edge=1080, max_edge=2048, scale=3):
    """Pixel-footprint crop, no semantic enlargement or automatic whole-figure fallback."""
    exact = native_box(box, view)["mapped_native_unclipped"]
    w, h = view.global_size
    bounds = (max(0, math.floor(exact[0])), max(0, math.floor(exact[1])),
              min(w, math.ceil(exact[2])+1), min(h, math.ceil(exact[3])+1))
    cw, ch = bounds[2]-bounds[0], bounds[3]-bounds[1]
    if cw <= 0 or ch <= 0:
        raise ValueError("Empty inspection crop")
    edge = max(min_edge, min(max_edge, max(cw, ch)*scale))
    dims = tuple(max(2, round(d*edge/max(cw, ch))) for d in (cw, ch))
    return Crop(tuple(view.global_size), bounds, dims)


def local_views(native_size, overlap=.10):
    """Reuse 2x2 overlapping local observation windows, not reconstruction partitions."""
    w, h = native_size
    out = []
    for row in range(2):
        for col in range(2):
            x0, y0, x1, y1 = round(col*w/2), round(row*h/2), round((col+1)*w/2), round((row+1)*h/2)
            dx, dy = math.ceil((x1-x0)*overlap), math.ceil((y1-y0)*overlap)
            b = (max(0, x0-dx), max(0, y0-dy), min(w, x1+dx), min(h, y1+dy))
            cw, ch = b[2]-b[0], b[3]-b[1]
            edge = max(1080, min(2048, 3*max(cw, ch)))
            dims = tuple(max(2, round(v*edge/max(cw, ch))) for v in (cw, ch))
            out.append((f"L{row*2+col+1}", Crop(tuple(native_size), b, dims)))
    return out


def visible_global_bounds(local, global_view):
    b = local.box
    # First/last native pixel centers, distinct from PIL half-open crop bounds.
    exact = global_view.to_local(b[:2]) + global_view.to_local([b[2]-1, b[3]-1])
    return [max(0, min(global_view.input_size[k % 2]-1, v)) for k, v in enumerate(exact)]


def intersects(a, b):
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]
