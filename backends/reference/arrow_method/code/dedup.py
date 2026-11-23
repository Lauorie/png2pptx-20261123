"""Whole-path nominations followed by conservative, model-authored ownership edits.

No coordinate or style fitting, mask subtraction, partial-path removal, or API I/O.
"""
import copy
from itertools import combinations

import numpy as np


def polyline(row):
    if row.get('beziers'):
        t = np.linspace(0, 1, 129)[:, None]
        chunks = []
        for c in np.asarray(row['beziers'], float):
            chunks.append((1-t)**3*c[0] + 3*(1-t)**2*t*c[1] +
                          3*(1-t)*t*t*c[2] + t**3*c[3])
        p = np.concatenate(chunks)
    else:
        p = np.asarray(row.get('points') or [row['start'], *row.get('via', []), row['end']], float)
    return p[np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-8]]


def sample(row, count=160):
    p = polyline(row)
    if len(p) < 2:
        return None, 0.0
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    q = np.linspace(0, s[-1], count)
    return np.column_stack([np.interp(q, s, p[:, k]) for k in (0, 1)]), float(s[-1])


def distances(points, line):
    """Exact point-to-polyline distances, avoiding nearest-sample aliasing."""
    a, v = line[:-1], np.diff(line, axis=0)
    delta = points[:, None, :] - a
    t = np.clip(np.sum(delta*v, axis=-1) / np.sum(v*v, axis=-1), 0, 1)
    return np.linalg.norm(delta-t[..., None]*v, axis=-1).min(axis=1)


def nominate(rows, endpoint_px=6, p90_px=3, min_length_px=15, length_ratio=.88):
    items = []
    for r in rows:
        if r.get('closed') or r.get('kind') == 'block_arrow':
            continue
        ps, length = sample(r)
        if ps is not None and length >= min_length_px:
            items.append((r, ps, length, polyline(r)))
    pairs = []
    for (a, ap, al, ag), (b, bp, bl, bg) in combinations(items, 2):
        ratio = min(al, bl) / max(al, bl)
        forward = max(np.linalg.norm(ap[0]-bp[0]), np.linalg.norm(ap[-1]-bp[-1]))
        reverse = max(np.linalg.norm(ap[0]-bp[-1]), np.linalg.norm(ap[-1]-bp[0]))
        endpoint = min(forward, reverse)
        if ratio < length_ratio or endpoint > endpoint_px:
            continue
        d = np.r_[distances(ap, bg), distances(bp, ag)]
        p90 = max(float(np.quantile(distances(ap, bg), .9)),
                  float(np.quantile(distances(bp, ag), .9)))
        if p90 <= p90_px:
            pairs.append(dict(id=f'P{len(pairs)+1:02d}', a=a['id'], b=b['id'],
                endpoint_max_px=float(endpoint), centerline_p90_px=p90,
                centerline_max_px=float(d.max()), length_ratio=ratio,
                reversed=bool(reverse < forward)))
    return pairs


def validate(answer, pairs):
    known = {p['id'] for p in pairs}
    if not isinstance(answer, dict) or not isinstance(answer.get('decisions'), list):
        raise ValueError('Expected {decisions:[...]}')
    ds = answer['decisions']
    if len(ds) != len(known) or {d.get('id') for d in ds} != known:
        raise ValueError('Each supplied pair must appear exactly once')
    for d in ds:
        if d.get('relation') not in ('duplicate', 'distinct', 'shared', 'uncertain'):
            raise ValueError('Unknown physical relation')
        if d.get('keep') not in ('A', 'B', 'both'):
            raise ValueError('keep must be A, B or both')
        if not isinstance(d.get('fully_redundant'), bool):
            raise ValueError('fully_redundant must be boolean')
        if not isinstance(d.get('reason'), str) or not d['reason'].strip():
            raise ValueError('Provide a short visible reason')
        if d['relation'] == 'duplicate':
            if d['keep'] == 'both' or not d['fully_redundant']:
                raise ValueError('A duplicate requires a fully redundant loser and one keeper')
        elif d['keep'] != 'both' or d['fully_redundant']:
            raise ValueError('Nonduplicate and uncertain pairs must keep both')
    return answer


def opposite_heads(a, b, reverse=False):
    def occupied(r):
        s = r.get('style', {})
        return {i for i, k in enumerate(('start', 'end')) if s.get(k)}
    ah, bh = occupied(a), occupied(b)
    if reverse:
        bh = {1-i for i in bh}
    return bool(ah and bh and ah.isdisjoint(bh))


def apply(scene, pairs, answer):
    """Every removal needs a direct duplicate decision to an actual survivor.

    Conflicting decisions or an opposite-head ambiguity retain the entire group.
    Connected components never imply transitive physical identity by themselves.
    """
    validate(answer, pairs)
    rows = {r['id']: r for r in scene['arrows']}
    decision = {d['id']: d for d in answer['decisions']}
    edges, adjacency, guards = {}, {}, []
    for p in pairs:
        d = decision[p['id']]
        if d['relation'] != 'duplicate':
            continue
        if opposite_heads(rows[p['a']], rows[p['b']], p['reversed']):
            guards.append(dict(pair=p['id'], reason='Opposite endpoint heads require separate review; retained both'))
            continue
        edges[frozenset((p['a'], p['b']))] = (p, d)
        for a, b in ((p['a'], p['b']), (p['b'], p['a'])):
            adjacency.setdefault(a, set()).add(b)
    removals, seen, held = [], set(), []
    order = {r['id']: i for i, r in enumerate(scene['arrows'])}
    for seed in sorted(adjacency, key=order.get):
        if seed in seen:
            continue
        component, stack = set(), [seed]
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node]-component)
        seen |= component
        internal = [p for p in pairs if p['a'] in component and p['b'] in component]
        if any(frozenset((p['a'], p['b'])) not in edges for p in internal):
            held.append(dict(ids=sorted(component), reason='Conflicting/uncertain internal relation; kept all'))
            continue
        candidates = [k for k in component if all(frozenset((k, o)) in edges for o in component-{k})]
        if not candidates:
            held.append(dict(ids=sorted(component), reason='No direct duplicate evidence to one common survivor'))
            continue
        votes = {k: 0 for k in candidates}
        for p in internal:
            d = decision[p['id']]
            preferred = p['a'] if d['keep'] == 'A' else p['b']
            if preferred in votes:
                votes[preferred] += 1
        keeper = max(candidates, key=lambda k: (votes[k], -order[k]))
        for loser in sorted(component-{keeper}, key=order.get):
            p, d = edges[frozenset((keeper, loser))]
            removals.append(dict(remove=loser, keep=keeper, pair=p['id'],
                reason=d['reason'], model_preferred=p['a'] if d['keep']=='A' else p['b']))
    deleted = {r['remove'] for r in removals}
    updated = copy.deepcopy(scene)
    updated['arrows'] = [r for r in updated['arrows'] if r['id'] not in deleted]
    # Kept rows remain byte-for-byte JSON equivalent, including geometry/style.
    assert all(r == rows[r['id']] for r in updated['arrows'])
    assert all(r['keep'] not in deleted for r in removals)
    return updated, dict(removed=removals, guards=guards, held_components=held,
                         retained_geometry_style_unchanged=True)
