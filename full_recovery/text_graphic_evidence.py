"""Two source views must agree before retiring a graphic misread as text.

Uses the existing local and expanded geometry calls. Ambiguity, a blank crop,
spelling, character length, and a single model verdict never remove a target.
"""
from copy import deepcopy

PROTOCOL = 'two_view_graphic_ownership.1'


def prompt(value):
    value = deepcopy(value)
    value['protocol'] += '_' + PROTOCOL
    value['rules'].append(
        'GRAPHIC OWNERSHIP: independently inspect CLEAN around every target. '
        'If ALL intended ink belongs to a visible connector, border, node outline '
        'or illustration, with no independent glyph occurrence, return '
        'status=non_text, units=[], and graphic_evidence. Identify the enclosing '
        'graphic in owner_bbox_2d and explain its visible continuation/attachment '
        'beyond the target. In particular inspect dash cadence and line continuation '
        'through nodes before treating a fragment as a letter or mathematical sign. '
        'The old kind/text is fallible, including kind=formula. A real standalone '
        'letter, sign, bullet, ellipsis, or complete formula must remain text. '
        'An empty, clipped or uncertain location is ambiguous, never non_text. '
        'Any independent glyph mixed with graphics also prevents non_text. '
        'Do not infer this decision from spelling, size or length. The local and '
        'expanded source views are checked independently before any retirement.')
    schema = value['response_format']['items'][0]
    schema['status'] += '/non_text'
    schema['graphic_evidence'] = {
        'graphic_type': 'connector/border/node_outline/illustration',
        'owner_bbox_2d': [0, 0, 100, 100],
        'independent_glyphs_present': False,
        'continuation': 'Visible graphic structure beyond the target; only for non_text'}
    return value


def validate(item, size, target_box, *, wide, prior=None):
    from shape_integration.texts.response_validation import numeric_box
    from shape_integration.common import valid_box
    evidence = item.get('graphic_evidence')
    if (item.get('units') != [] or not isinstance(evidence, dict)
            or evidence.get('graphic_type') not in
            {'connector', 'border', 'node_outline', 'illustration'}
            or evidence.get('independent_glyphs_present') is not False
            or not isinstance(item.get('reason'), str) or not item['reason'].strip()
            or not isinstance(evidence.get('continuation'), str)
            or not evidence['continuation'].strip()):
        raise ValueError('non_text requires explicit complete graphic ownership evidence')
    owner = numeric_box(evidence.get('owner_bbox_2d'))
    if not valid_box(owner, size) or not valid_box(target_box, size):
        raise ValueError('A clipped/invalid target or owner cannot authorize retirement')
    if not (owner[0] <= target_box[0] and owner[1] <= target_box[1]
            and owner[2] >= target_box[2] and owner[3] >= target_box[3]):
        raise ValueError('The graphic owner must cover the complete original target')
    if owner == target_box:
        raise ValueError('Show graphic context beyond the target, not just a glyph-sized box')
    value = deepcopy(item)
    value.update(status='non_text_candidate', units=[], protocol=PROTOCOL)
    value['graphic_evidence']['owner_bbox_2d'] = owner
    if (wide and prior and prior.get('graphic_evidence', {}).get('graphic_type')
            == evidence['graphic_type']):
        value.update(status='confirmed_non_text', local_evidence=deepcopy(prior))
    return value
