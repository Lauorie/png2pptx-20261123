"""Use backend boxes to select textual identity context, never expose their values."""
from ..common import intersect
from .observation import sanitize


def _contains(a, b):
    return a[0] <= b[0] and a[1] <= b[1] and a[2] >= b[2] and a[3] >= b[3]


def identity_context(candidate, candidates, assets, history, padding, *, exclude_shape_relations=()):
    excluded = set(exclude_shape_relations)
    if not excluded <= {'contains_target', 'nearby'}:
        raise ValueError('Unknown shape neighbor exclusion relation')
    target = candidate['bbox_2d']
    pad = padding * max(target[2]-target[0], target[3]-target[1])
    crop = [target[0]-pad, target[1]-pad, target[2]+pad, target[3]+pad]
    reasons = [candidate.get('reason', '')]
    reasons += [r.get('reason', '') for r in history if r.get('id') == candidate['id']][-2:]
    descriptions = list(dict.fromkeys(sanitize(r)[0] for r in reasons if sanitize(r)[0]))
    descriptions = descriptions or [sanitize('Localization selected '+candidate['label']+' as '+candidate['kind'])[0]]
    nearby = []
    for group, objects in [('separate_shape_candidate', candidates), ('image_asset', assets)]:
        for obj in objects:
            if group == 'separate_shape_candidate' and obj['id'] == candidate['id']:
                continue
            box = obj.get('bbox_2d')
            if box is None or not intersect(crop, box):
                continue
            relation = ('inside_target' if _contains(target, box) else 'contains_target' if _contains(box, target)
                        else 'overlaps_target' if intersect(target, box) else 'nearby')
            if group == 'separate_shape_candidate' and relation in excluded:
                continue
            nearby.append({'id': obj['id'], 'role': group, 'relation_hint': relation,
                           'label': sanitize(obj.get('label', obj.get('target', '')))[0],
                           'kind': sanitize(obj.get('kind', obj.get('category', '')))[0]})
    return {'identity_descriptions': descriptions, 'nearby_inventory': nearby,
            'ownership_note': 'Relations are approximate localization hints, not proof of ownership. '
                              'Avoid duplicating separately represented objects; report ambiguous parent/child ownership.'}
