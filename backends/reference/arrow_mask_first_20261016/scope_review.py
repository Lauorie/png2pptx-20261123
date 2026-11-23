"""After recovery, remove only explicitly identified out-of-scope paths."""
import copy
from pathlib import Path
import numpy as np
import coverage_visual
import evidence as ev
import protocol
from lib import fit
from scene_snapshot import canonical_rows
from transport.common import write_json as write


def prompt(size, inventory):
    return dict(
        task='Review ONLY whether existing reconstructed paths are eligible line/arrow content. Return removals, not new geometry.',
        size=list(size), existing_paths=inventory,
        instructions=[
            protocol.SCOPE,
            'ORIGINAL is source evidence. RECOVERED shows existing paths in blue over a faded source, labelled with unique P IDs. '
            'Remove a path only when it clearly traces an excluded non-arrow shape boundary, composite-grid cell edge, text stroke or image texture. '
            'A valid block arrow remains eligible. A missing arrowhead or a dense connector network is not a reason to remove a path.',
            'Do not add paths, change coordinates, fix styles, repair omissions or deduplicate overlapping arrows. '
            'Poor geometry alone is not a scope error. If one path mixes a valid connector and a shape border, keep it and report uncertain; '
            'this review must not delete its valid portion.',
            'Use only supplied existing P IDs. Unlisted paths are kept unchanged. '
            'Return concise visible reasons for removals and uncertainties; no reasoning about diagram architecture.',
        ],
        output_schema=dict(remove='[{id:existing P ID,reason:brief visible scope error}]',
                           uncertain='[{id:existing P ID,reason:brief uncertainty}]'),
    )


def validate(raw, ids):
    if not isinstance(raw, dict):
        raise ValueError('Scope review must be an object')
    seen = set()
    result = {}
    for field in ('remove', 'uncertain'):
        items = raw.get(field)
        if not isinstance(items, list):
            raise ValueError(field + ' must be a list')
        result[field] = []
        for item in items:
            if not isinstance(item, dict) or item.get('id') not in ids or item['id'] in seen:
                raise ValueError('Review IDs must be known and appear in only one decision')
            if not isinstance(item.get('reason'), str) or not item['reason'].strip():
                raise ValueError('Each review decision needs a visible reason')
            seen.add(item['id'])
            result[field].append(dict(id=item['id'], reason=item['reason']))
    return result


def apply(results, removed_ids):
    known = {r['id'] for g in results for r in g.get('rows', [])}
    if not set(removed_ids) <= known:
        raise ValueError('Cannot remove unknown drawable path IDs')
    result = copy.deepcopy(results)
    for group in result:
        removed = [r['id'] for r in group.get('rows', []) if r['id'] in removed_ids]
        group['rows'] = [r for r in group.get('rows', []) if r['id'] not in removed_ids]
        if removed:
            group['scope_review_removed'] = removed
    return result


def run(clean, results, out, model, locks):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(canonical_rows(results), key=lambda r: r['id'])
    write(out / 'before_results.json', results)
    if not rows:
        write(out / 'audit.json', dict(state='empty', calls=0, removed=[], uncertain=[]))
        return copy.deepcopy(results), []
    inventory, names = [], {}
    for i, row in enumerate(rows, 1):
        pid = f'P{i:03d}'
        row['short_id'] = pid
        names[pid] = row['id']
        p = fit.points(row)
        inventory.append(dict(id=pid, kind=row['kind'],
                              bbox=np.round(np.r_[p.min(axis=0), p.max(axis=0)], 1).tolist()))
    clean.save(out / 'original.png')
    with locks['cpu']:
        coverage_visual.render(clean, rows, ev.identity_view(clean.size), out / 'recovered.png')
    write(out / 'id_map.json', names)
    try:
        answer = model.ask(out / 'request', prompt(clean.size, inventory),
                           [('ORIGINAL', out / 'original.png'), ('RECOVERED with drawable path IDs', out / 'recovered.png')],
                           lambda a: validate(a, names))
        write(out / 'answer.json', answer)
        removed = [names[v['id']] for v in answer['remove']]
        updated = apply(results, removed)
        uncertain = [dict(id=names[v['id']], reason=v['reason']) for v in answer['uncertain']]
        write(out / 'after_results.json', updated)
        write(out / 'audit.json', dict(state='reviewed', calls=1, removed=removed,
                                       removal_reasons=answer['remove'], uncertain=uncertain,
                                       kept_paths_geometry_style_unchanged=True))
        with locks['cpu']:
            coverage_visual.render(clean, [r for r in rows if r['id'] not in removed],
                                   ev.identity_view(clean.size), out / 'after.png')
        issues = [dict(stage='scope_review_uncertain', paths=uncertain)] if uncertain else []
        return updated, issues
    except Exception as exc:
        write(out / 'audit.json', dict(state='failed', calls=1, removed=[], retained_all=True, error=str(exc)))
        write(out / 'after_results.json', results)
        return copy.deepcopy(results), [dict(stage='scope_review_failed', error=str(exc), retained_all=True)]
