from copy import deepcopy
import pytest

from full_recovery.source_text_occurrences import validate_geometry, adapt
from full_recovery.text_graphic_evidence import validate


def graphic():
    return {'id': 'mark', 'status': 'non_text', 'units': [],
            'reason': 'A continuous dashed route crosses this node.',
            'graphic_evidence': {'graphic_type': 'connector',
                'owner_bbox_2d': [20, 0, 70, 180],
                'independent_glyphs_present': False,
                'continuation': 'Aligned equally spaced dashes above and below the node.'}}


def check(item, wide=False, prior=None, box=None):
    return validate(item, [200, 200], box or [40, 70, 50, 90], wide=wide, prior=prior)


def test_one_view_never_retires_and_two_views_preserve_both_proofs():
    local = check(graphic())
    assert local['status'] == 'non_text_candidate'
    assert check(graphic(), wide=True)['status'] == 'non_text_candidate'
    wide = check(graphic(), wide=True, prior=local)
    assert wide['status'] == 'confirmed_non_text'
    assert wide['local_evidence'] == local
    # Normalized cached results must pass the same validation on resumption.
    assert check(wide, wide=True, prior=local) == wide


def test_disagreement_or_partial_context_cannot_retire():
    local = check(graphic())
    changed = graphic(); changed['graphic_evidence']['graphic_type'] = 'border'
    assert check(changed, wide=True, prior=local)['status'] == 'non_text_candidate'
    for box in ([-1, 70, 50, 90], [40, 70, 201, 90], [80, 70, 90, 90]):
        with pytest.raises(ValueError): check(graphic(), wide=True, prior=local, box=box)


@pytest.mark.parametrize('field,value', [
    ('independent_glyphs_present', True), ('independent_glyphs_present', 'false'),
    ('continuation', ''), ('graphic_type', 'unknown'),
    ('owner_bbox_2d', [40, 70, 50, 90])])
def test_graphics_mixed_with_glyphs_or_without_positive_evidence_cannot_retire(field, value):
    item = graphic(); item['graphic_evidence'][field] = value
    with pytest.raises(ValueError): check(item, wide=True, prior=check(graphic()))


def passthrough(raw, ids, size):
    assert all(i['status'] in ('resolved', 'ambiguous') for i in raw['items'])
    return deepcopy(raw)


@pytest.mark.parametrize('kind,text', [('text', 'i'), ('formula', '−'), ('text', '…')])
def test_real_short_labels_formulas_and_uncertainty_remain(kind, text):
    for status, units in [('resolved', [{'bbox_2d': [40,70,50,90]}]), ('ambiguous', [])]:
        raw = {'items': [{'id': 'mark', 'status': status, 'units': units, 'reason': 'Visible label'}]}
        assert validate_geometry(raw, ['mark'], [200,200], original=passthrough,
            targets={'mark': {'kind': kind, 'text': text}}, verified={}, wide=True,
            local_graphics={'mark': check(graphic())}, target_boxes={'mark': [40,70,50,90]}) == raw


def test_false_formula_can_be_retired_only_by_two_source_proofs_not_old_kind():
    kwargs = dict(original=passthrough, targets={'mark': {'kind': 'formula', 'text': 'i'}},
                  verified={}, target_boxes={'mark': [40,70,50,90]})
    local = validate_geometry({'items': [graphic()]}, ['mark'], [200,200], **kwargs)['items'][0]
    result = validate_geometry({'items': [graphic()]}, ['mark'], [200,200], **kwargs,
                               wide=True, local_graphics={'mark': local})
    assert result['items'][0]['status'] == 'confirmed_non_text'


def test_one_bad_graphic_does_not_discard_independent_text():
    bad = graphic(); bad['graphic_evidence']['independent_glyphs_present'] = True
    good = {'id': 'label', 'status': 'resolved', 'units': [{'bbox_2d': [90,20,120,40]}]}
    result = validate_geometry({'items': [bad, good]}, ['mark', 'label'], [200,200],
        original=passthrough, targets={}, verified={}, partial=True,
        target_boxes={'mark': [40,70,50,90]})
    assert next(i for i in result['items'] if i['id'] == 'label') == good
    assert next(i for i in result['items'] if i['id'] == 'mark')['status'] == 'invalid'
    assert 'ownership evidence' in next(i for i in result['items'] if i['id'] == 'mark')['reason']
