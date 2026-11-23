"""Retire an invented occurrence only against an already verified source keeper.

This extends the reference backend's existing full-figure recovery request.
Local ambiguity, spelling equality and spatial proximity never delete text.
"""
from copy import deepcopy

PROTOCOL = 'verified_source_occurrence_recovery.2'


def ordinary(row):
    return (row.get('kind', 'text') == 'text'
            and all(r.get('script', 'normal') == 'normal' for r in row.get('runs', [])))


def verified_occurrences(updates, completed, recovery_ids=()):
    trusted=set(completed)-set(recovery_ids)
    return {parent: deepcopy(parts[0]) for parent, parts in updates.items()
            if parent in trusted and len(parts) == 1
            and parts[0]['id'] == parent and ordinary(parts[0])
            and parts[0].get('visual_line_count', 1) == 1}


def context_rows(rows, verified):
    return [verified.get(row['id'], row) for row in rows]


def recovery_prompt(prompt, verified, targets):
    result = deepcopy(prompt)
    keepers = {ident for ident in verified if ident not in targets}
    result['protocol'] += '_' + PROTOCOL
    result['verified_source_occurrence_ids'] = sorted(keepers)
    for row in result['context_read_only']:
        row['source_occurrence_verified'] = row['id'] in keepers
    result['rules'] = [r for r in result['rules'] if not r.startswith((
        'Preserve the physical occurrence', 'FULL-FIGURE RECOVERY:',
        'EXPANDED-NEIGHBORHOOD RECOVERY:'))]
    result['rules'].extend([
        'EXPANDED-NEIGHBORHOOD RECOVERY: locate each supplied physical occurrence '
        'within the supplied CLEAN view, using its original position and neighboring structure. '
        'Rough positions and earlier transcription are fallible. Never move a target '
        'onto a different repeated label. Distinct visible instances stay distinct, '
        'including repeated words, numbers and ellipses.',
        'A verified_source_occurrence_id has already passed source geometry and '
        'fixed-box attribute reading. Its corrected box is in context_read_only. '
        'If CLEAN establishes that an UNVERIFIED target has no independent glyph '
        'occurrence and the intended source ink is already completely represented '
        'by ONE such read-only keeper, return status=duplicate, units=[], '
        'duplicate_of=that keeper ID, and a reason explaining both the false target '
        'location and the actual source occurrence. Keep the keeper unchanged. '
        'Matching spelling, proximity, repetition or an empty rough box alone '
        'is insufficient. Different transcriptions can denote the same source ink.',
        'If any separate intended occurrence might exist, identity is uncertain, '
        'or no verified keeper represents all intended ink, return ambiguous '
        'with units=[]. Formulas and mathematical scripts cannot be retired as duplicates. '
        'Do not retire, replace or relocalize a verified keeper.'])
    result['response_format']['items'][0].update(
        status='resolved/ambiguous/duplicate',
        duplicate_of='Only for duplicate: an allowed verified read-only keeper ID')
    return result


def validate_geometry(raw, ids, size, *, original, targets, verified, partial=False,
                      wide=False, local_graphics=None, target_boxes=None):
    from shape_integration.texts.response_validation import isolated
    if partial:
        return isolated(raw, ids, lambda item: validate_geometry(
            {'items': [item]}, [item['id']], size, original=original,
            targets=targets, verified=verified, wide=wide, local_graphics=local_graphics,
            target_boxes=target_boxes)['items'][0])
    if isinstance(raw, list):
        raw = {'items': raw}
    values = raw.get('items') if isinstance(raw, dict) else None
    if (not isinstance(values, list) or not all(isinstance(v, dict) for v in values)
            or sorted(str(v.get('id', '')) for v in values) != sorted(ids)):
        raise ValueError('Return each supplied geometry target exactly once')
    result = []
    for item in values:
        if item.get('status') in ('non_text', 'non_text_candidate', 'confirmed_non_text'):
            from .text_graphic_evidence import validate
            if item['id'] in verified:
                raise ValueError('A verified keeper cannot be retired as a graphic')
            result.append(validate(item, size, (target_boxes or {}).get(item['id']),
                                   wide=wide, prior=(local_graphics or {}).get(item['id'])))
            continue
        if item.get('status') != 'duplicate':
            result.extend(original({'items': [item]}, [item['id']], size)['items'])
            continue
        target = item['id']; keeper = item.get('duplicate_of')
        if target in verified or not ordinary(targets[target]):
            raise ValueError('Only an unverified ordinary text target can be retired')
        if not isinstance(keeper,str) or keeper not in verified or keeper in targets:
            raise ValueError('duplicate_of must name a verified_source_occurrence_id outside the targets; '
                             'this keeper is unverified or unavailable. Keep ambiguous or locate an independent occurrence.')
        if (not ordinary(verified[keeper]) or item.get('units',[]) != []
                or not isinstance(item.get('reason'),str) or not item['reason'].strip()):
            raise ValueError('Duplicate requires no new geometry and a concrete source-ink explanation')
        result.append({'id': target, 'status': 'duplicate', 'units': [],
                       'duplicate_of': keeper, 'reason': item['reason'], 'protocol': PROTOCOL})
    return {'items': result}


def adapt(source):
    """Adapt a NEW backend copy; executed reference freezes stay immutable."""
    replacements = [
        ("    def job(index,group,wide=False):",
         "    from full_recovery import source_text_occurrences as occurrences\n"
         "    verified_source_occurrences={}\n"
         "    local_graphic_evidence={}\n"
         "    retired_graphic_occurrences=[]\n"
         "    source_occurrence_aliases=[]\n"
         "    def job(index,group,wide=False):"),
        ("        visible=[r for r in rows if intersect(r['bbox_2d'],f['box'])>0]",
         "        context_source=occurrences.context_rows(rows,verified_source_occurrences) if wide else rows\n"
         "        visible=[r for r in context_source if intersect(r['bbox_2d'],f['box'])>0]"),
        ("            save(folder/'geometry_prompt.json',prompt)",
         "            prompt=occurrences.recovery_prompt(prompt,verified_source_occurrences,ids)\n"
         "            save(folder/'geometry_prompt.json',prompt)"),
        ("        errors=[];updates={};excluded=[];completed=[]",
         "        from full_recovery.text_graphic_evidence import prompt as graphic_prompt\n"
         "        prompt=graphic_prompt(prompt)\n"
         "        save(folder/'geometry_prompt.json',prompt)\n"
         "        target_boxes={r['id']:r['rough_bbox_2d'] for r in targets}\n"
         "        errors=[];updates={};excluded=[];completed=[]"),
        ("validate_geometry(raw,ids,f['size'],partial=True)",
         "occurrences.validate_geometry(raw,ids,f['size'],original=validate_geometry,targets=originals,verified=verified_source_occurrences if wide else {},partial=True,wide=wide,local_graphics=local_graphic_evidence,target_boxes=target_boxes)"),
        ("validate_geometry(raw,pending,f['size'],partial=True)",
         "occurrences.validate_geometry(raw,pending,f['size'],original=validate_geometry,targets=originals,verified=verified_source_occurrences if wide else {},partial=True,wide=wide,local_graphics=local_graphic_evidence,target_boxes=target_boxes)"),
        ("        for item in geometry['items']:\n            if item['status']!='resolved':",
         "        aliases=[item for item in geometry['items'] if item['status']=='duplicate']\n"
         "        save(folder/'source_occurrence_aliases.json',{'protocol':occurrences.PROTOCOL,'retired':aliases})\n"
         "        for item in geometry['items']:\n"
         "            if not wide and item['status']=='non_text_candidate':\n"
         "                local_graphic_evidence[item['id']]=item\n"
         "            if item['status']=='confirmed_non_text':\n"
         "                retired_graphic_occurrences.append({**item,'group':name})\n"
         "                updates[item['id']]=[];completed.append(item['id']);continue\n"
         "            if item['status']=='duplicate':\n"
         "                source_occurrence_aliases.append({**item,'group':name})\n"
         "                updates[item['id']]=[];completed.append(item['id']);continue\n"
         "            if item['status']!='resolved':"),
        ("        if not unit_sources:return name,{},[],[],errors",
         "        if not unit_sources:\n"
         "            save(folder/'updates.json',{'updates':updates,'errors':errors,'completed_ids':completed})\n"
         "            return name,updates,excluded,completed,errors"),
        ("        recovery_groups=compact_groups(recovery_rows,size,6)",
         "        verified_source_occurrences=occurrences.verified_occurrences(updates,completed,recovery_ids)\n"
         "        recovery_groups=compact_groups(recovery_rows,size,6)"),
        ("        'excluded_image_text_ids':excluded,'regions_needing_review':errors,'unresolved_multiline_ids':unresolved_multiline,",
         "        'retired_source_duplicate_ids':[item['id'] for item in source_occurrence_aliases],\n"
         "        'source_occurrence_aliases':source_occurrence_aliases,\n"
         "        'retired_graphic_occurrences':retired_graphic_occurrences,\n"
         "        'excluded_image_text_ids':excluded,'regions_needing_review':errors,'unresolved_multiline_ids':unresolved_multiline,"),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('Reference occurrence recovery anchor changed: ' + before.splitlines()[0])
        source = source.replace(before, after)
    return source
