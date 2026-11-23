"""Bounded visual recall and source-confirmed occurrence repair before sizing.

Geometric/text containment only proposes review candidates. It never deletes
text. Only an explicit, validated same-occurrence response can retire an ID.
"""
from copy import deepcopy
import re

from PIL import Image

from ..common import intersect, save
from .controller import frame, local_hints, mapped_box, merge
from .layout_integrity import overlap
from .local_geometry import union
from .occurrence_dedup import key


def duplicate_pairs(rows):
    """Find overlapping literal token containment, never formula/script pairs."""
    pairs = []
    for index, a in enumerate(rows):
        for b in rows[index + 1:]:
            if any(r.get('kind') == 'formula' or any(
                    s.get('script', 'normal') != 'normal' for s in r.get('runs', []))
                   for r in (a, b)):
                continue
            if abs((a.get('rotation_deg', 0) - b.get('rotation_deg', 0) + 180) % 360 - 180) > 5:
                continue
            ka, kb = key(a), key(b)
            if not ka or not kb or overlap(a['bbox_2d'], b['bbox_2d']) < .8:
                continue
            # Word boundaries matter: 'a' inside 'data' is not a duplicate tag.
            shorter, longer = sorted((a['text'], b['text']), key=len)
            if re.search(r'(?<!\w)' + re.escape(shorter.strip()) + r'(?!\w)', longer):
                pairs.append([a['id'], b['id']])
    return pairs


def recall_regions(rows, size, limit=4):
    """Partition dense/small/rotated diagrams independently of model regions[]."""
    if not rows:
        return [[0, 0, size[0] - 1, size[1] - 1]][:limit]
    small = sum(min(r['bbox_2d'][2] - r['bbox_2d'][0],
                    r['bbox_2d'][3] - r['bbox_2d'][1]) < max(size) * .015 for r in rows)
    rotated = sum(abs(r.get('rotation_deg', 0)) > 10 for r in rows)
    if len(rows) < 16 and small < 4 and rotated < 2:
        return []
    nx, ny = (2, 2) if len(rows) >= 48 else ((2, 1) if size[0] >= size[1] else (1, 2))
    boxes = [[size[0] * x / nx, size[1] * y / ny,
              min(size[0] - 1, size[0] * (x + 1) / nx),
              min(size[1] - 1, size[1] * (y + 1) / ny)]
             for y in range(ny) for x in range(nx)]
    return boxes[:limit]


def validate_reply(raw, size, opt, known, pairs):
    from .staged_formula import inventory_validate
    if not isinstance(raw, dict):
        raise ValueError('Expected inventory audit object')
    checked = inventory_validate(raw, size, opt, local=True, known=known)
    removals = raw.get('duplicate_ids', [])
    allowed = {frozenset(pair) for pair in pairs}
    if not isinstance(removals, list):
        raise ValueError('duplicate_ids must be a list')
    retired = set()
    for item in removals:
        if (not isinstance(item, dict) or not isinstance(item.get('id'), str) or
                not isinstance(item.get('duplicate_of'), str) or item.get('id') in retired or
                not isinstance(item.get('reason'), str) or not item['reason'].strip() or
                frozenset((item.get('id'), item.get('duplicate_of'))) not in allowed):
            raise ValueError('Deletion requires a supplied physical-overlap candidate and visible reason')
        retired.add(item['id'])
    if any(item['duplicate_of'] in retired for item in removals):
        raise ValueError('Duplicate retirement cannot form chains or cycles')
    if any(retired.intersection(row['replaces']) for row in checked['texts']):
        raise ValueError('Cannot both replace and retire the same occurrence')
    checked['duplicate_ids'] = deepcopy(removals)
    return checked


def apply_reply(rows, reply, f, history, opt):
    """Apply atomically; a retired glyph must remain in an explicit keeper."""
    result = deepcopy(rows)
    old = {r['id']: r for r in result}
    entries = reply['texts']
    replacements = {r['replaces'][0]: r for r in entries if r['replaces']}
    for item in reply['duplicate_ids']:
        keeper = replacements.get(item['duplicate_of'])
        removed = old[item['id']]
        if (keeper is None or keeper.get('legibility') != 'clear' or
                not key(removed) or key(removed) not in key(keeper)):
            raise ValueError('Retired text must remain in an explicitly reread keeper')
        box = mapped_box(keeper['bbox_2d'], f, inverse=True)
        if overlap(box, removed['bbox_2d']) < .5:
            raise ValueError('Retirement cannot move glyphs to a different occurrence')
    changes = deepcopy(history)
    history_length = len(changes)
    merge(result, entries, f, 'inventory_visual_audit', changes, opt)
    retired = {item['id'] for item in reply['duplicate_ids']}
    result = [r for r in result if r['id'] not in retired]
    if len({r['id'] for r in result}) != len(result):
        raise ValueError('Inventory audit produced duplicate IDs')
    history.extend(changes[history_length:])
    history.extend({'action': 'retire_source_confirmed_duplicate', **item}
                   for item in reply['duplicate_ids'])
    return result


def audit(inventory, source, folder, runtime, opt, progress=lambda *a, **k: None):
    """Add missed text and resolve contained occurrences, with bounded requests."""
    if not opt.get('inventory_audit_enabled', True) or runtime is None:
        return deepcopy(inventory)
    result = deepcopy(inventory)
    folder.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        size = list(image.size)
    plans = recall_regions(result['texts'], size, opt.get('inventory_recall_max_regions', 4))
    # Small diagrams can still contain a duplicate; review that neighborhood.
    if not plans:
        by_id = {r['id']: r for r in result['texts']}
        plans = [union([by_id[a], by_id[b]]) for a, b in duplicate_pairs(result['texts'])][
            :opt.get('inventory_recall_max_regions', 4)]
    save(folder/'plan.json', {'regions': plans, 'policy': 'visual_recall_and_overlap_review_v1'})
    history = result.setdefault('merge_history', [])
    reports = []
    for index, box in enumerate(plans):
        root = folder/f'R{index:02d}'
        progress('inventory_visual_audit', region=index + 1, total=len(plans))
        f = frame(source, root/'input', {**opt, 'crop_min_padding_px': 32,
                  'crop_padding': 0, 'crop_scale': 100}, box)
        rows = result['texts']
        visible = [r for r in rows if intersect(r['bbox_2d'], f['box']) > 0]
        known = [r['id'] for r in visible]
        pairs = duplicate_pairs(visible)
        local_hints(root/'input/clean.png', visible, f, root/'input/ids.png')
        from .staged_formula import inventory_prompt
        from . import model
        prompt = inventory_prompt(f['size'], opt, local=True, known=known)
        prompt.update(task='Audit text inventory against CLEAN: add missing occurrences and correct mislocalized or duplicated units.',
            existing_occurrences=[{'id': r['id'], 'text': r['text'],
                'rough_bbox_2d': mapped_box(r['bbox_2d'], f)} for r in visible],
            suspected_duplicate_pairs=pairs)
        prompt['rules'] += [
            'Search the ENTIRE CLEAN region for missing text, including WHITE or colored text, small rotated labels, repeated labels at different positions, ellipses and bullets. Existing boxes are NOT a coverage mask: inspect their interiors and gaps too. Never invent text from repeated structure.',
            'Return new missing occurrences with replaces=[]. Correct an existing occurrence with replaces=[its ID]. Omit already-correct occurrences. Source pixels determine the physical position; rough boxes can be wrong. Never relocate a repeated word to a different occurrence.',
            'Each physical glyph occurrence must be represented only once. For a label plus an overlapping substring, read actual baselines: a nearby list number may be a SEPARATE label, not part of the word. Correct their independent boxes/transcriptions, preserving every visible number and word.',
            'Only suspected_duplicate_pairs may be retired. To retire an actual duplicate return duplicate_ids=[{id: redundant ID, duplicate_of: keeper ID, reason: visible evidence}], AND explicitly reread the complete keeper in texts with replaces=[keeper ID]. The keeper must preserve the redundant visible characters. Otherwise duplicate_ids=[]. Never retire distinct physical occurrences, formulas or mathematical scripts.',
            'For text cut by this audit crop, do not shorten or replace a complete existing entry with a partial fragment. Report uncertainty instead. Background stays transparent; do not output background attributes.']
        prompt['example']['duplicate_ids'] = []
        try:
            reply = model.ask(runtime, root, prompt,
                [('CLEAN source; the only glyph evidence.', root/'input/clean.png'),
                 ('Existing rough IDs only; not source ink.', root/'input/ids.png')],
                lambda raw: validate_reply(raw, f['size'], opt, known, pairs), opt)
            before = len(result['texts'])
            result['texts'] = apply_reply(rows, reply, f, history, opt)
            reports.append({'region': index, 'before': before, 'after': len(result['texts']),
                            'returned_units': len(reply['texts']), 'retired': reply['duplicate_ids']})
        except Exception as exc:
            warning = {'stage': 'inventory_visual_audit', 'region': index,
                       'reason': str(exc), 'fallback': 'previous complete inventory retained'}
            result.setdefault('regions_needing_review', []).append(warning)
            reports.append(warning)
        save(root/'application.json', reports[-1])
    save(folder/'summary.json', {'regions': len(plans), 'reports': reports,
         'remaining_duplicate_pairs': duplicate_pairs(result['texts'])})
    remaining=duplicate_pairs(result['texts'])
    if remaining:
        result.setdefault('regions_needing_review',[]).append({
            'stage':'inventory_duplicate_review','pairs':remaining,
            'reason':'Overlapping token-containment remains unconfirmed; no automatic deletion'})
    save(folder/'inventory.json', result)
    return result
