"""Bounded part edits: source-guided changes, programmatically preserved siblings."""
import copy
from pathlib import Path

from ..common import read
from .contracts import part_refs, validate_observation


def plan_revision(folder, review):
    local = read(folder / 'fit_local.json')
    base = local['observation']
    parts = {p['id']: p for p in base['parts']}
    editable = {p['id'] for p in local['parts'] if p['status'] != 'ok'}
    removed, issues = set(), []
    additions = local['fit_status'] != 'ok' or base['status'] != 'ok'
    for issue in (review or {}).get('issues', []):
        ids = issue.get('part_ids') or [review.get('part_id_map', {}).get(p, p) for p in issue['parts']]
        owners = set(ids) & parts.keys()
        if issue['kind'] == 'extra':
            removed.update(owners)
        else:
            editable.update(owners)
        additions |= issue['kind'] == 'missing'
        issues.append({k: copy.deepcopy(issue[k]) for k in ['kind', 'evidence', 'action']})
        issues[-1]['part_ids'] = ids
    editable -= removed
    # Shared vertices define one editable dependency group; never move half a
    # shared face/edge while claiming to have preserved the other half.
    while True:
        refs = {pid for key in editable for pid in part_refs(parts[key])}
        linked = {key for key, p in parts.items() if key not in removed and refs.intersection(part_refs(p))}
        if linked <= editable:
            break
        editable.update(linked)
    protected = parts.keys() - editable - removed
    return {'source_revision': str(folder), 'base_observation': copy.deepcopy(base),
            'base_ownership': copy.deepcopy(local.get('declared_ownership')),
            'frame': copy.deepcopy(local['frame']),
            'editable_ids': sorted(editable), 'remove_ids': sorted(removed),
            'preserved_ids': sorted(protected), 'allow_new_parts': bool(additions), 'issues': issues,
            'previous_display_part_ids': copy.deepcopy((review or {}).get('part_id_map', {}))}


def revision_prompt(context, size, mode):
    base = context['base_observation']
    editable = [p for p in base['parts'] if p['id'] in context['editable_ids']]
    refs = {pid for p in editable for pid in part_refs(p)}
    points = copy.deepcopy([p for p in base['points'] if p['id'] in refs])
    if mode == 'norm1000':
        for p in points:
            p['xy'] = [v * 1000 / (size[i] - 1) for i, v in enumerate(p['xy'])]
    return {'edit_part_ids': context['editable_ids'], 'remove_part_ids': context['remove_ids'],
            'preserved_parts': [{'id': p['id'], 'kind': p['kind']} for p in base['parts']
                                if p['id'] in context['preserved_ids']],
            'previous_editable_observations': {'points': points, 'parts': editable},
            'previous_observation_notes': base.get('reason', ''),
            'previous_display_part_ids': context['previous_display_part_ids'],
            'allow_new_parts_for_missing_geometry': context['allow_new_parts'],
            'issues': context['issues'],
            'rule': 'Return ONLY replacements for edit_part_ids, using those EXACT part IDs, plus '
                    'source-confirmed missing parts only when additions are allowed. Omit remove_part_ids; '
                    'the backend removes them. Do not return or re-interpret preserved_parts; the backend '
                    'copies them unchanged. Previous coordinates/notes are fallible context, not ground truth. '
                    'Observe corrections from the clean source. Include the points referenced by your returned '
                    'parts. Keep shared vertex IDs within the editable group. For removal-only edits return '
                    'status=ok, points=[], parts=[]. Do not add letters or labels during a geometry correction.'}


def reframe_context(context, frame):
    """Transport one bounded edit to a larger crop; source geometry stays fixed."""
    from .frames import FitFrame
    old=FitFrame.from_metadata(context['frame']);new=FitFrame.from_metadata(frame)
    if old.original_size!=new.original_size or old.global_size!=new.global_size:
        raise ValueError('Part-edit reframe requires the same original image')
    result=copy.deepcopy(context)
    if old==new:return result
    for point in result['base_observation']['points']:
        point['xy']=new.to_local(old.to_original(point['xy']))
        if any(v<0 or v>new.input_size[i]-1 for i,v in enumerate(point['xy'])):
            raise ValueError('Expanded edit crop clips existing supported geometry')
    result['frame']=new.metadata()
    result['source_frame']=copy.deepcopy(context.get('source_frame',context['frame']))
    result['geometry_reframed_without_refitting']=True
    return result


def merge_revision(patch, context, object_id, size, *, audit):
    """Merge by stable part identity; unrelated model redraws cannot change siblings."""
    if patch['status'] in ('needs_context', 'unsupported'):
        return patch
    base = context['base_observation']
    old_parts = {p['id']: p for p in base['parts']}
    editable, removed = set(context['editable_ids']), set(context['remove_ids'])
    selected = [p for p in patch['parts'] if p['id'] in editable or (
        context['allow_new_parts'] and p['id'] not in old_parts)]
    chosen = {p['id']: copy.deepcopy(p) for p in selected}
    missing = sorted(editable - chosen.keys())
    # Keep a previous valid component if a patch omits it. Incompleteness stays
    # visible to the existing critic/controller instead of erasing geometry.
    retained = [p for p in base['parts'] if p['id'] not in removed and p['id'] not in chosen]
    retained_refs = {pid for p in retained for pid in part_refs(p)}
    table = {p['id']: copy.deepcopy(p) for p in base['points'] if p['id'] in retained_refs}
    incoming_refs = {pid for p in selected for pid in part_refs(p)}
    remap = {}
    reserved = set(table) | incoming_refs
    for point in patch['points']:
        if point['id'] not in incoming_refs:
            continue
        pid = point['id']
        if pid in table and table[pid]['xy'] != point['xy']:
            replacement = 'edit_' + pid
            while replacement in reserved:
                replacement = 'edit_' + replacement
            remap[pid] = replacement; reserved.add(replacement)
        table[remap.get(pid, pid)] = {**copy.deepcopy(point), 'id': remap.get(pid, pid)}
    for part in chosen.values():
        for item in part.get('segments', [part]):
            item['point_ids'] = [remap.get(pid, pid) for pid in item['point_ids']]
    parts = [chosen.get(p['id'], copy.deepcopy(p)) for p in base['parts'] if p['id'] not in removed]
    parts.extend(chosen[p['id']] for p in selected if p['id'] not in old_parts)
    used = {pid for p in parts for pid in part_refs(p)}
    out = {**copy.deepcopy(patch), 'parts': parts, 'points': [p for k, p in table.items() if k in used]}
    if missing:
        out['status'] = 'partial'
        out['reason'] = (out['reason'] + ' Unrepaired part IDs retained: ' + ', '.join(missing)).strip()
    if not parts:
        out.update(status='unsupported', reason='Critic-requested removals leave no in-scope geometry.')
    validate_observation(out, object_id, size)
    audit.update(source_revision=context['source_revision'], preserved_ids=context['preserved_ids'],
                 edited_ids=sorted(chosen.keys() & editable), removed_ids=sorted(removed),
                 added_ids=sorted(chosen.keys() - old_parts.keys()), unrepaired_ids=missing,
                 ignored_unrequested_ids=[p['id'] for p in patch['parts'] if p['id'] not in chosen],
                 renamed_point_ids=remap)
    return out


def preserve_fits(local, context):
    """Retain successful protected commands/parameters exactly, not by re-fitting."""
    previous = read(Path(context['source_revision']) / 'fit_local.json')
    from .frames import FitFrame
    source_frame=FitFrame.from_metadata(previous['frame']);target_frame=FitFrame.from_metadata(local['frame'])
    reframed=source_frame!=target_frame
    if reframed and not context.get('geometry_reframed_without_refitting'):
        raise ValueError('Part revision requires the same crop coordinate frame')
    old = {p['id']: p for p in previous['parts']}
    for part in local['parts']:
        if part['id'] in context['preserved_ids'] and old[part['id']]['status'] == 'ok':
            key = part['key']
            current_parameters=copy.deepcopy(part.get('parameters',{}))
            current_diagnostics=copy.deepcopy(part.get('diagnostics',{}))
            part.update(copy.deepcopy(old[part['id']]))
            part['key'] = key
            if reframed:
                from .adapter import bounds
                from .scene import native_parameters
                preserved=old[part['id']]
                original_commands=[[command[0],*[v for i in range(1,len(command),2)
                    for v in source_frame.to_original(command[i:i+2])]] for command in preserved['commands']]
                part['commands']=[[command[0],*[v for i in range(1,len(command),2)
                    for v in target_frame.to_local(command[i:i+2])]] for command in original_commands]
                part['bbox_2d']=bounds(part['commands'])
                # Keep exact original commands and primitive parameters for
                # export, rather than refitting protected geometry after resize.
                original_part={**preserved,'parameters_local':preserved['parameters'],'commands':original_commands}
                part['_preserved_original_commands']=original_commands
                part['_preserved_native_parameters_original']=native_parameters(original_part,source_frame.metadata())
                # Numerical diagnostics and parameters describe the new local
                # frame, while the actual source contour remains exact.
                part['parameters']=current_parameters
                part['diagnostics']=current_diagnostics
                part['observed_part']=copy.deepcopy(next(p for p in local['observation']['parts'] if p['id']==part['id']))
    # Bounds and diagnostics depend on the successful parts, not their namespace.
    from .adapter import union_bounds, shared_point_diagnostics
    local['bbox_2d'] = union_bounds(local['parts'])
    succeeded = sum(p['status'] == 'ok' for p in local['parts'])
    failed = len(local['parts']) - succeeded
    local['part_counts'] = {'observed': len(local['parts']), 'succeeded': succeeded, 'failed': failed}
    local['fit_status'] = ('not_run' if not local['parts'] else 'failed' if not succeeded else
                           'partial' if failed or local['observation_status'] != 'ok' else 'ok')
    local['completeness_status'] = 'unreviewed' if local['fit_status'] == 'ok' else 'incomplete'
    local['shared_point_diagnostics'] = shared_point_diagnostics(
        local['parts'], {p['id']: p['xy'] for p in local['observation']['points']})
    return [p['id'] for p in local['parts'] if p['status'] != 'ok'
            and p['id'] in old and old[p['id']]['status'] == 'ok']
