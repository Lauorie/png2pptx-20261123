"""Spatial search bookkeeping and native-source crops; no semantic classification."""
import math

from PIL import Image, ImageDraw

from ..frames import Crop, resized_size


def union_boxes(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def contains(outer, inner):
    return all(outer[i] <= inner[i] for i in (0, 1)) and all(outer[i] >= inner[i] for i in (2, 3))


def pixel_box(bbox, size):
    return [max(0, math.floor(bbox[0])), max(0, math.floor(bbox[1])),
            min(size[0], math.ceil(bbox[2])+1), min(size[1], math.ceil(bbox[3])+1)]


def uncovered_boxes(boxes, size):
    """Exact rectangle subtraction in the global image's pixel-footprint domain."""
    gaps = [[0, 0, *size]]
    for box in boxes:
        x0, y0, x1, y1 = pixel_box(box, size)
        remaining = []
        for a, b, c, d in gaps:
            l, t, r, z = max(a, x0), max(b, y0), min(c, x1), min(d, y1)
            if l >= r or t >= z:
                remaining.append([a, b, c, d]); continue
            remaining.extend(v for v in ([a,b,c,t], [a,z,c,d], [a,t,l,z], [r,t,c,z])
                             if v[0] < v[2] and v[1] < v[3])
        gaps = remaining
    # Merge adjacent pieces without introducing unseen holes.
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(gaps):
            for j in range(i+1, len(gaps)):
                b = gaps[j]; u = union_boxes([a, b])
                area = lambda q: (q[2]-q[0])*(q[3]-q[1])
                if area(u) == area(a)+area(b):
                    gaps[i] = u; gaps.pop(j); changed = True; break
            if changed: break
    return gaps


def gap_regions(boxes, size):
    gaps = uncovered_boxes(boxes, size)
    # Thin borders and fragmented gaps are inspected together, never silently discarded.
    if len(gaps) > 4:
        gaps = [union_boxes(gaps)]
    result = []
    for a, b, c, d in gaps:
        x0, y0 = min(a, size[0]-2), min(b, size[1]-2)
        result.append([x0, y0, max(x0+1, c-1), max(y0+1, d-1)])
    return result


def split_box(box):
    """Uniform fallback only for an explicitly unresolved region lacking zoom proposals."""
    a,b,c,d = box
    if max(c-a, d-b) < 4: return []
    if c-a >= d-b:
        mid = (a+c)/2
        return [[a,b,mid,d], [mid,b,c,d]]
    mid = (b+d)/2
    return [[a,b,c,mid], [a,mid,c,d]]


def native_crop(source, bbox, global_size):
    """Crop original pixels first, then compose the existing half-pixel transforms."""
    requested = Crop.from_model(bbox, global_size)
    sw, sh = source.size; gw, gh = global_size
    a,b,c,d = requested.box
    native = [max(0, math.floor(a*sw/gw)), max(0, math.floor(b*sh/gh)),
              min(sw, math.ceil(c*sw/gw)), min(sh, math.ceil(d*sh/gh))]
    image = source.crop(native)
    frame = Crop(tuple(global_size), (native[0]*gw/sw, native[1]*gh/sh,
                                     native[2]*gw/sw, native[3]*gh/sh), resized_size(image.size))
    metadata = frame.metadata(source.size)
    metadata.update(source_crop_half_open=native, sampled_from='original_resolution',
                    source_crop_size=list(image.size),
                    source_to_local_scale_xy=[frame.input_size[0]/image.width, frame.input_size[1]/image.height])
    return frame, image, metadata


def coverage_fraction(boxes, size):
    mask = Image.new('1', tuple(size)); draw = ImageDraw.Draw(mask)
    for box in boxes:
        a,b,c,d = pixel_box(box, size)
        draw.rectangle((a,b,c-1,d-1), fill=1)
    return mask.histogram()[1]/(size[0]*size[1])
