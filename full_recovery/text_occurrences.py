"""Ground repeated plain text in observed source pixels, then reconcile recalls.

No source count is inferred from a diagram template. Complete source glyphs,
an equal-count one-to-one assignment and bounded displacement are required.
Ambiguous groups retain their existing output and an explicit diagnostic.
"""
from copy import deepcopy
import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

PROTOCOL = 'source_grounded_text_occurrences.1'


def refit_position(row, fit):
    """Place the existing native glyphs at their source center, retaining size."""
    metric = row['layout']['measurement']
    b = row['bbox_2d']
    rotated = metric.get('rotated_ink_bounds')
    pw, ph = ((rotated[2]-rotated[0], rotated[3]-rotated[1])
              if rotated else (metric['width'], metric['height']))
    base = min((b[2]-b[0]+1)/pw, (b[3]-b[1]+1)/ph)
    fraction = row['font_size_px']/(metric['reference_size_px']*base)
    placed = fit(row, metric, fraction)
    if abs(placed['font_size_px']-row['font_size_px']) > 1e-6:
        raise ValueError('Source occurrence placement changed the supplied font size')
    return placed


def eligible(row):
    return (row.get('kind', 'text') == 'text'
        and row.get('layout_mode', 'single_line') == 'single_line'
        and row.get('visual_line_count', 1) == 1
        and abs(row.get('rotation_deg', 0)) < 1
        and 2 <= len(row.get('text', '').strip()) <= 64
        and all(c.isalnum() or c.isspace() for c in row['text'])
        and all(r.get('script', 'normal') == 'normal' for r in row.get('runs', []))
        and all(r.get('color',row.get('color'))==row.get('color')
            and r.get('bold',row.get('bold',False))==row.get('bold',False)
            and r.get('italic',row.get('italic',False))==row.get('italic',False)
            for r in row.get('runs',[]))
        and len(row.get('bbox_2d', [])) == 4
        and np.all(np.isfinite(row['bbox_2d']))
        and np.all(np.asarray(row['bbox_2d'][2:]) > row['bbox_2d'][:2]))


def ink(rgb, pigment, background):
    delta = background - pigment
    if float(delta @ delta) < 64:
        return None
    return np.clip(((background-rgb)*delta).sum(2)/float(delta @ delta), 0, 1)


def tight(rgb, pigment):
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    mask = ink(rgb.astype(float), pigment, background)
    if mask is None:
        return None
    yy, xx = np.nonzero(mask > .5)
    if len(xx) < 5:
        return None
    return [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)], background


def source_template(row, source):
    from .source_text_direction import observe
    receipt = observe(row, source)
    if not receipt:
        return None
    best = max(receipt['alternatives'], key=lambda r: r['source_ink_dice'])
    if best['source_ink_dice'] < .8 or abs(best['angle']) >= 1:
        return None
    metric = row['layout']['measurement']
    with Image.open(metric['ink_template']) as image:
        alpha = image.convert('RGBA').getchannel('A')
        bounds = alpha.getbbox()
        if bounds is None:
            return None
        alpha = alpha.crop(bounds)
    span = np.asarray(row['bbox_2d'][2:])-row['bbox_2d'][:2]
    base = min(span[0]/alpha.width, span[1]/alpha.height)*best['probe_scale']
    size = np.maximum(1, np.rint(np.array(alpha.size)*base)).astype(int)
    lo = np.array(receipt['source_probe_box'][:2])+best['probe_offset']
    region = [*np.maximum(0, lo-2), *np.minimum(source.size, lo+size+2)]
    crop = np.asarray(source.crop(tuple(region)).convert('RGB'))
    pigment = np.array([int(row['color'][i:i+2], 16) for i in (1, 3, 5)])
    measured = tight(crop, pigment)
    if not measured:
        return None
    bbox, background = measured
    if (bbox[0]+region[0]<=0 or bbox[1]+region[1]<=0 or
            bbox[2]+region[0]>=source.width or bbox[3]+region[1]>=source.height):
        return None
    return {'rgb': crop, 'ink_box': bbox, 'pigment': pigment,
            'background': background, 'anchor_id': row['id'], 'score': best['source_ink_dice']}


def complete_word(rgb, bbox, pigment, background):
    x0, y0, x1, y1 = bbox
    gap = max(1, round((y1-y0)/3))
    for a, b in [(max(0, x0-gap), x0), (x1, min(rgb.shape[1], x1+gap))]:
        if b <= a:
            continue
        mask = ink(rgb[y0:y1, a:b].astype(float), pigment, background)
        if mask is not None and np.mean(mask > .5) > .1:
            return False
    return True


def source_matches(template, source, region):
    rgb = np.asarray(source.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kernel = cv2.cvtColor(template['rgb'], cv2.COLOR_RGB2GRAY)
    h, w = kernel.shape
    if h >= gray.shape[0] or w >= gray.shape[1]:
        return []
    score = cv2.matchTemplate(gray, kernel, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.nonzero((score >= .90) & (score >= cv2.dilate(score, np.ones((3, 3), 'uint8'))))
    matches = []
    for y, x in sorted(zip(ys, xs), key=lambda p: -float(score[p])):
        measured = tight(rgb[y:y+h, x:x+w], template['pigment'])
        if not measured:
            continue
        box = np.array(measured[0])+np.array([x, y, x, y])
        center = (box[:2]+box[2:])/2
        if np.any(center < region[:2]) or np.any(center > region[2:]):
            continue
        if any(np.linalg.norm(center-np.asarray(m['center'])) < .5*min(box[2:]-box[:2]) for m in matches):
            continue
        if not complete_word(rgb, box, template['pigment'], template['background']):
            continue
        matches.append({'box': box.tolist(), 'center': center.tolist(), 'score': float(score[y, x])})
    return matches


def groups(rows):
    groups = {}
    for row in rows:
        if not eligible(row) or row.get('source_coverage'):
            continue
        key = (row['text'], row.get('font'), row.get('bold', False), row.get('italic', False))
        groups.setdefault(key, []).append(row)
    for family in groups.values():
        def size(row):
            return row.get('font_size_px') or (row['bbox_2d'][2]-row['bbox_2d'][0])/max(1,len(row['text']))
        pending = sorted(family, key=size)
        while pending:
            group = [pending.pop(0)]
            reference = size(group[0])
            similar = [r for r in pending if max(size(r)/reference,reference/size(r)) < 1.5]
            group.extend(similar);pending = [r for r in pending if r not in similar]
            if len(group) >= 3:
                yield group


def ground(rows, source):
    """Return copied rows, changed IDs and source receipts; never fit glyphs here."""
    output = deepcopy(rows);table = {r['id']: r for r in output};changed = [];audit = []
    for group in groups(rows):
        boxes = np.array([r['bbox_2d'] for r in group], float)
        typical = np.median(boxes[:, 2:]-boxes[:, :2], axis=0)
        region = np.r_[boxes[:, :2].min(0)-typical, boxes[:, 2:].max(0)+typical]
        found = [];anchors = [];kernels = set()
        for row in group:
            template = source_template(row, source)
            if not template:
                continue
            anchors.append(template['anchor_id'])
            # Examine every distinct source phase before claiming equal count.
            # Identical crops share their observation rather than rerunning it.
            key = (template['rgb'].shape, template['rgb'].tobytes())
            if key in kernels:
                continue
            kernels.add(key)
            matches = source_matches(template, source, region)
            for match in matches:
                center = np.asarray(match['center'])
                same = next((i for i, old in enumerate(found) if
                    np.linalg.norm(center-old['center']) < .5*min(
                    np.array(match['box'][2:])-match['box'][:2])), None)
                if same is None:
                    found.append(match)
                elif match['score'] > found[same]['score']:
                    found[same] = match
            if len(found) > len(group):
                break
        record = {'protocol': PROTOCOL, 'ids': [r['id'] for r in group], 'state': 'abstain'}
        if len(found) != len(group):
            record['reason'] = 'No complete source template with an equal occurrence count'
            record['source_count'] = len(found)
            audit.append(record);continue
        centers = (boxes[:, :2]+boxes[:, 2:])/2
        targets = np.array([m['center'] for m in found])
        distance = np.linalg.norm(centers[:, None]-targets[None], axis=2)
        ii, jj = linear_sum_assignment(distance)
        neighbors = np.linalg.norm(targets[:, None]-targets[None], axis=2)
        np.fill_diagonal(neighbors, np.inf)
        reach = np.maximum(np.max(typical), neighbors.min(1))
        if any(distance[i, j] > reach[j] for i, j in zip(ii, jj)):
            record['reason'] = 'Source assignment requires ambiguous long-distance relocalization'
            audit.append(record);continue
        record.update(state='grounded', anchor_ids=anchors, source_count=len(found), observations=[])
        for i, j in zip(ii, jj):
            old = group[i];row = table[old['id']];b = found[j]['box']
            # Source boxes are half-open; the text backend uses pixel centers.
            b = [float(b[0]), float(b[1]), float(b[2]-1), float(b[3]-1)]
            evidence = {'protocol': PROTOCOL, 'source_bbox_2d': b,
                'source_match_score': found[j]['score'], 'anchor_ids': anchors,
                'one_to_one_equal_source_count': True}
            row['source_occurrence_grounding'] = evidence
            record['observations'].append({'id': old['id'], 'before': old['bbox_2d'], 'after': b, **evidence})
            if max(abs(np.array(b)-old['bbox_2d'])) > .75:
                row['bbox_2d'] = b;changed.append(row['id'])
        audit.append(record)
    return output, changed, audit


def reconcile_recalls(rows, source):
    """Retire a recall only against a grounded owner of the SAME source ink."""
    output = [];audit = []
    owners = [r for r in rows if r.get('source_occurrence_grounding') and not r.get('source_coverage')]
    rgb = np.asarray(source.convert('RGB'))
    for row in rows:
        retired = False
        if eligible(row) and row.get('source_coverage', {}).get('complete_label_reference'):
            b = np.asarray(row['bbox_2d']);lo = np.maximum(0, np.floor(b[:2]-1)).astype(int)
            hi = np.minimum(source.size, np.ceil(b[2:]+2)).astype(int)
            pigment = np.array([int(row['color'][i:i+2], 16) for i in (1, 3, 5)])
            measured = tight(rgb[lo[1]:hi[1], lo[0]:hi[0]], pigment) if np.all(hi > lo) else None
            if measured:
                actual = np.array(measured[0])+np.tile(lo, 2);actual[2:] -= 1
                matches = [r for r in owners if r['text'] == row['text']
                    and np.max(abs(actual-np.array(r['source_occurrence_grounding']['source_bbox_2d']))) <= 1]
                if len(matches) == 1:
                    retired = True
                    audit.append({'id': row['id'], 'owner_id': matches[0]['id'],
                        'action': 'retire_recall_alias', 'source_bbox_2d': actual.tolist(),
                        'reason': 'Complete source glyphs already represented by one grounded occurrence'})
        if not retired:
            output.append(row)
    return output, audit
