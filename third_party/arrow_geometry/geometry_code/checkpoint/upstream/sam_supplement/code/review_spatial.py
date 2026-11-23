"""Spatial scheduling/crops copied from v022; these functions do not merge or delete boxes."""
import hashlib
import json
from frames import crop_from_global

def compact(rows):
    return [{k:r[k] for k in ('id','bbox','render_type','grouping','label','extent')} for r in rows]

def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])

def overlap(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))

def signature(row):
    return hashlib.sha256(json.dumps(compact([row])[0], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def pair_key(a, b):
    return tuple(sorted((signature(a), signature(b))))

def unresolved_pairs(rows, checked):
    return [(a['id'], b['id']) for i, a in enumerate(rows) for b in rows[i + 1:]
            if overlap(a['bbox'], b['bbox']) > 0 and pair_key(a, b) not in checked]

def batches(rows, limit, checked=None):
    """Disjoint editable IDs per wave; bbox overlap only chooses who sees whom."""
    if limit < 2:
        raise ValueError('Review batch must permit at least two candidates')
    index = {r['id']: r for r in rows}
    pending = set(index)
    pairs = None if checked is None else unresolved_pairs(rows, checked)
    result = []
    while pending:
        if pairs is None:
            seed = max(pending, key=lambda k: (area(index[k]['bbox']), k))
            neighbors = [k for k in pending - {seed} if overlap(index[k]['bbox'], index[seed]['bbox']) > 0]
        else:
            active = [(a, b) for a, b in pairs if a in pending and b in pending]
            if not active:
                break
            degrees = {k: sum(k in edge for edge in active) for k in pending}
            seed = max(pending, key=lambda k: (degrees[k], area(index[k]['bbox']), k))
            neighbors = list({b if a == seed else a for a, b in active if seed in (a, b)})
        neighbors.sort(key=lambda k: (-overlap(index[seed]['bbox'], index[k]['bbox']) /
                                     min(area(index[seed]['bbox']), area(index[k]['bbox'])), k))
        chosen = [seed] + neighbors[:limit - 1]
        result.append([index[k] for k in chosen]); pending.difference_update(chosen)
    return result

def frame_for(rows, view, context_scale, expansion=0):
    boxes = [r['bbox'] for r in rows]
    box = [min(b[k] for b in boxes) if k < 2 else max(b[k] for b in boxes) for k in range(4)]
    center = [(box[k] + box[k + 2]) / 2 for k in (0, 1)]
    # Min context is expressed in original image pixels, not resized display pixels.
    minimum = [25 * view.input_size[k] / view.global_size[k] for k in (0, 1)]
    extent = [max(box[k + 2] - box[k], minimum[k]) * context_scale * (1.6 ** expansion) for k in (0, 1)]
    window = [max(0., center[k] - extent[k] / 2) for k in (0, 1)] + [
        min(view.input_size[k] - 1., center[k] + extent[k] / 2) for k in (0, 1)]
    return crop_from_global(window, view, min_edge=1080, max_edge=1536)
