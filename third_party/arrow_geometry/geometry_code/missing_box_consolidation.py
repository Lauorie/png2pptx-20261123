"""Consolidate reviewed omission regions before append-only geometry recovery.

Matches use native vector bounds, never localization boxes or stroke width.
Original members remain in provenance; a merged locator carries no one-member
description that could incorrectly narrow geometry recovery to that member.
"""
import copy
import math


def area(box):
    return max(0., box[2] - box[0]) * max(0., box[3] - box[1])


def iou(a, b):
    overlap = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    return overlap / max(1e-12, area(a) + area(b) - overlap)


def _roots(a, b, c):
    if abs(a) < 1e-12:
        return [] if abs(b) < 1e-12 else [-c / b]
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    return [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)]


def path_bounds(row):
    if row.get('beziers'):
        points = []
        for curve in row['beziers']:
            ts = {0., 1.}
            for d in (0, 1):
                p0, p1, p2, p3 = [p[d] for p in curve]
                ts.update(t for t in _roots(3 * (-p0 + 3*p1 - 3*p2 + p3),
                                            6 * (p0 - 2*p1 + p2), 3 * (p1 - p0)) if 0 < t < 1)
            for t in ts:
                u = 1 - t
                points.append([u**3*curve[0][d] + 3*u*u*t*curve[1][d] +
                               3*u*t*t*curve[2][d] + t**3*curve[3][d] for d in (0, 1)])
    else:
        points = row.get('points') or [row['start'], *row.get('via', []), row['end']]
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]


def nonzero_bbox(box):
    result = list(box)
    for d in (0, 1):
        if box[d + 2] - box[d] < 1:
            center = (box[d] + box[d + 2]) / 2
            result[d], result[d + 2] = center - .5, center + .5
    return result


def consolidate(additions, existing_rows, *, threshold=.7, filter_existing=True):
    """Return kept targets and a complete audit for one image; inputs unchanged.

    Every original member pair must meet the threshold, so overlap chains do
    not accumulate into a large box. Existing filtering requires every member
    to match the same existing path. Containment alone does not remove a target.
    """
    if not 0 < threshold <= 1:
        raise ValueError('IoU threshold must be in (0, 1]')
    groups = []
    for item in sorted(additions, key=lambda x: (x['render_type'], -area(x['bbox_native']), x['id'])):
        matches = []
        for index, group in enumerate(groups):
            if group[0]['render_type'] != item['render_type']:
                continue
            score = min(iou(item['bbox_native'], member['bbox_native']) for member in group)
            if score >= threshold:
                matches.append((score, -index))
        if matches:
            groups[-max(matches)[1]].append(item)
        else:
            groups.append([item])
    merged = []
    for members in groups:
        target = copy.deepcopy(members[0])
        boxes = [m['bbox_native'] for m in members]
        target.update(bbox_native=[min(b[0] for b in boxes), min(b[1] for b in boxes),
                                   max(b[2] for b in boxes), max(b[3] for b in boxes)],
                      grouping='group' if any(m.get('grouping') == 'group' for m in members) else 'single',
                      locator='' if len(members) > 1 else target.get('locator', ''),
                      member_ids=[m['id'] for m in members],
                      source_candidates=list(dict.fromkeys(m['source_candidate'] for m in members)),
                      members=copy.deepcopy(members))
        merged.append(target)
    existing = []
    if filter_existing:
        for row in existing_rows:
            box = path_bounds(row)
            existing.append(dict(id=row['id'], bbox=nonzero_bbox(box), raw_bbox=box,
                                 render_type='block_arrow' if row['kind'] == 'block_arrow' else 'linework'))
    kept, filtered, comparisons = [], [], []
    for target in merged:
        matches = [(min(iou(m['bbox_native'], old['bbox']) for m in target['members']), old['id'], old)
                   for old in existing if old['render_type'] == target['render_type']]
        best = max(matches, key=lambda x: (x[0], x[1])) if matches else None
        reject = bool(best and best[0] >= threshold)
        comparison = dict(id=target['id'], filtered=reject, best_iou=best[0] if best else 0,
                          matched_existing=best[1] if best else None)
        if best:
            comparison.update(existing_bbox=best[2]['bbox'], existing_raw_bbox=best[2]['raw_bbox'])
        comparisons.append(comparison)
        (filtered if reject else kept).append(target)
    return kept, dict(iou_threshold=threshold, existing_filter_enabled=filter_existing,
                     input_count=len(additions), merged_count=len(merged),
                     filtered_count=len(filtered), output_count=len(kept),
                     merged_targets=merged, filtered_targets=filtered, existing_comparisons=comparisons)
