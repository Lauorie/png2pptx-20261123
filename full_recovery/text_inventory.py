"""Source-confirmed aliases in the reference's existing inventory review.

OCR spelling is fallible. Spatial collisions nominate review; neither a
collision nor uncertainty alone retires an occurrence.
"""
from copy import deepcopy
import json

PROTOCOL = 'source_confirmed_inventory_occurrences.6'


def duplicate_pairs(rows):
    from shape_integration.texts.layout_integrity import overlap
    pairs = []
    for index, a in enumerate(rows):
        for b in rows[index + 1:]:
            if any(r.get('kind') == 'formula' or any(
                    s.get('script', 'normal') != 'normal' for s in r.get('runs', []))
                   for r in (a, b)):
                continue
            if abs((a.get('rotation_deg', 0) - b.get('rotation_deg', 0) + 180) % 360 - 180) > 5:
                continue
            # The reference already uses .20 to nominate text collisions.
            # This only proposes source review, never an automatic deletion.
            if overlap(a['bbox_2d'], b['bbox_2d']) >= .20:
                pairs.append([a['id'], b['id']])
    return pairs


def apply_reply(rows, reply, f, history, opt):
    from shape_integration.texts.controller import mapped_box, merge
    from shape_integration.texts.layout_integrity import overlap
    from shape_integration.texts.occurrence_dedup import key
    result = deepcopy(rows)
    old = {r['id']: r for r in result}
    entries = reply['texts']
    replacements = {r['replaces'][0]: r for r in entries if r['replaces']}
    proofs, held = [], []
    for item in reply['duplicate_ids']:
        keeper = replacements.get(item['duplicate_of'])
        removed = old[item['id']]
        if keeper is None or keeper.get('legibility') != 'clear' or not key(keeper):
            held.append({**item, 'hold_reason': 'No complete, clearly reread source keeper'})
            continue
        box = mapped_box(keeper['bbox_2d'], f, inverse=True)
        old_keeper = old[item['duplicate_of']]
        # A clipped audit view cannot authorize retiring a complete occurrence.
        if any(old_keeper['bbox_2d'][i] < f['box'][i] - 1e-6 if i < 2 else
               old_keeper['bbox_2d'][i] > f['box'][i] + 1e-6 for i in range(4)):
            held.append({**item, 'hold_reason': 'Keeper crosses the source audit crop'})
            continue
        if overlap(box, removed['bbox_2d']) < .5:
            held.append({**item, 'hold_reason': 'Reread box does not establish the same occurrence'})
            continue
        proofs.append((item, keeper, box))
    approved_keepers = {item['duplicate_of'] for item, _, _ in proofs}
    held_keepers = {item['duplicate_of'] for item in held} - approved_keepers
    # A rejected alias does not authorize moving its keeper. Other independent
    # source corrections and fully proved aliases keep their own evidence.
    entries = [row for row in entries if not held_keepers.intersection(row['replaces'])]
    changes = deepcopy(history)
    start = len(changes)
    merge(result, entries, f, 'inventory_visual_audit', changes, opt)
    applied = {r['id']: r for r in result}
    for item, reread, box in proofs:
        keeper = applied[item['duplicate_of']]
        if key(keeper) != key(reread) or overlap(keeper['bbox_2d'], box) < .99:
            raise ValueError('Source-confirmed keeper was not committed')
    accepted = [item for item, _, _ in proofs]
    retired = {item['id'] for item in accepted}
    result = [r for r in result if r['id'] not in retired]
    if len({r['id'] for r in result}) != len(result):
        raise ValueError('Inventory audit produced duplicate IDs')
    history.extend(changes[start:])
    history.extend({'action': 'retire_source_confirmed_duplicate', **item,
                    'protocol': PROTOCOL, 'spelling_is_not_occurrence_identity': True}
                   for item in accepted)
    history.extend({'action': 'hold_source_alias', **item, 'protocol': PROTOCOL}
                   for item in held)
    # The reference reports this normalized application result separately from
    # the immutable raw/model response stored in the request directory.
    reply['duplicate_ids'] = deepcopy(accepted)
    reply['held_duplicate_ids'] = deepcopy(held)
    return result


def install():
    from shape_integration.texts import inventory_audit
    if getattr(inventory_audit, '_source_occurrence_protocol', None) == PROTOCOL:
        return
    original = inventory_audit.validate_reply

    def validate_reply(raw, size, opt, known, pairs):
        value = deepcopy(raw)
        repairs = []
        if isinstance(value, dict) and isinstance(value.get('uncertain'), list):
            for index, note in enumerate(value['uncertain']):
                if isinstance(note, dict):
                    # These are diagnostic notes, never editable occurrences.
                    # Preserve every field without rejecting valid text edits
                    # because the provider used structured instead of plain notes.
                    value['uncertain'][index] = json.dumps(note, ensure_ascii=False, sort_keys=True)
                    repairs.append({'index': index, 'field': 'uncertain',
                                    'before': deepcopy(note), 'after': value['uncertain'][index]})
        if isinstance(value, dict) and isinstance(value.get('texts'), list):
            for index, row in enumerate(value['texts']):
                if not isinstance(row, dict):
                    continue
                status = row.get('legibility')
                normalized = {'cut': 'partial', 'cut_by_crop': 'partial',
                              'unclear': 'unreadable'}.get(status)
                if normalized:
                    row['legibility'] = normalized
                    repairs.append({'index': index, 'field': 'legibility',
                                    'before': status, 'after': normalized})
        # Validate observations independently: one malformed occurrence must
        # not discard unrelated source corrections. The existing audit call
        # gives rejected observations one bounded local reread.
        if isinstance(value, dict) and isinstance(value.get('texts'), list):
            accepted, rejected = [], []
            for index, row in enumerate(value['texts']):
                try:
                    unit = original({**value, 'texts': [row], 'duplicate_ids': []},
                                    size, opt, known, pairs)
                    accepted.extend(unit['texts'])
                except (ValueError, TypeError, KeyError, IndexError) as exc:
                    rejected.append({'index': index, 'candidate': deepcopy(row),
                                     'reason': str(exc)})
            # Keep strict cross-entry and retirement validation; isolation
            # never authorizes unknown IDs, duplicate replacements or cycles.
            checked = original({**value, 'texts': accepted}, size, opt, known, pairs)
            checked['rejected_texts'] = rejected
        else:
            checked = original(value, size, opt, known, pairs)
        checked.setdefault('format_repairs', []).extend(repairs)
        return checked

    from shape_integration.texts import prompt_protocol
    boundary = ('GEOMETRY BOUNDARY: recover characters, mathematical notation and actual '
                'omission/list punctuation. Repeated graphical node bodies and port markers '
                'remain geometry, whether hollow or filled; do not transcribe their outlines '
                'as font glyphs or recode their fill as a character. Use their attachment, '
                'arrangement and role in CLEAN to distinguish them from punctuation. '
                'When uncertain, report uncertainty rather than add a second representation.')
    if not any(rule.startswith('GEOMETRY BOUNDARY:') for rule in prompt_protocol.RECOVERY_SCOPE):
        prompt_protocol.RECOVERY_SCOPE.append(boundary)
    connector_boundary = (
        'CONNECTOR BOUNDARY: inspect the surrounding route before transcribing an isolated mark. '
        'A segment or dash belonging to a continuous or regularly dashed connector is line geometry, '
        'including segments visible inside or beside nodes. Do not turn fragments of that route into '
        'letters, mathematical symbols, punctuation, or rotated text merely because a tight crop '
        'resembles a glyph. Use alignment, continuation and dash cadence in the wider CLEAN context '
        'to establish line ownership. Preserve genuine isolated labels and mathematical signs when '
        'the source shows an independent text occurrence; short length alone never excludes text. '
        'If ownership is ambiguous, report the uncertainty instead of inventing a transcription.')
    if not any(rule.startswith('CONNECTOR BOUNDARY:') for rule in prompt_protocol.RECOVERY_SCOPE):
        prompt_protocol.RECOVERY_SCOPE.append(connector_boundary)

    inventory_audit.duplicate_pairs = duplicate_pairs
    inventory_audit.apply_reply = apply_reply
    inventory_audit.validate_reply = validate_reply
    inventory_audit._source_occurrence_protocol = PROTOCOL
